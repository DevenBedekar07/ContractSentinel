from __future__ import annotations
import json, re, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, Any

import httpx
import pymupdf
import networkx as nx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.ollama_service import generate_text

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
DB_PATH = DATA / "reviews.json"

app = FastAPI(title="ContractSentinel API", version="0.1.0", description="Contract review MVP: extraction, clause classification, policy retrieval, risk assessment and human review.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

POLICIES = [
 {"id":"POL-LIAB-01","category":"liability","title":"Liability cap","text":"Vendor liability should be capped at fees paid or payable in the preceding 12 months, except for agreed carve-outs.","risk_guidance":"Uncapped or unlimited liability needs legal review."},
 {"id":"POL-IND-01","category":"indemnity","title":"Mutual indemnity","text":"Indemnity obligations should be clearly scoped, proportionate, and include appropriate mutual protections where applicable.","risk_guidance":"One-sided or unlimited indemnity may create material exposure."},
 {"id":"POL-TERM-01","category":"termination","title":"Termination rights","text":"The agreement should provide a reasonable termination-for-convenience or cure mechanism where commercially appropriate.","risk_guidance":"No exit path or an unusually long cure period may create operational risk."},
 {"id":"POL-REN-01","category":"renewal","title":"Auto-renewal notice","text":"Automatic renewal should include a clear renewal period and a reasonable advance notice window for non-renewal.","risk_guidance":"Automatic renewal without a clear notice window may cause unwanted commitments."},
 {"id":"POL-PAY-01","category":"payment","title":"Payment terms","text":"Payment terms, due dates, disputed-invoice handling, and late fees should be explicit and commercially reasonable.","risk_guidance":"Ambiguous or unusually short payment deadlines need review."},
 {"id":"POL-CONF-01","category":"confidentiality","title":"Confidential information","text":"Confidentiality obligations should define protected information, permitted disclosures, exclusions, and survival period.","risk_guidance":"Overbroad or undefined confidentiality obligations need review."},
]

CATEGORY_PATTERNS = {
 "liability": [r"liabilit", r"damages", r"limitation of liability", r"consequential"],
 "indemnity": [r"indemnif", r"indemnity", r"hold harmless"],
 "termination": [r"terminat", r"cure period", r"terminate for convenience"],
 "renewal": [r"auto.?renew", r"renewal", r"non.?renew"],
 "payment": [r"payment", r"invoice", r"late fee", r"net \d+"],
 "confidentiality": [r"confidential", r"non.?disclosure", r"proprietary information"],
}
RISK_RULES = {
 "high": [r"unlimited liability", r"without limitation", r"perpetual and irrevocable", r"sole discretion"],
 "medium": [r"automatically renew", r"automatic renewal", r"within 5 days", r"waive all claims", r"indemnify.*any and all"],
}

class ReviewDecision(BaseModel):
    review_id: str
    clause_id: str
    decision: str = Field(pattern="^(approved|rejected)$")
    reviewer_note: str = ""

class LLMTestRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=4000
    )

class WorkflowState(TypedDict, total=False):
    text: str
    clauses: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    stage_log: list[str]


def read_store():
    if not DB_PATH.exists(): return []
    try: return json.loads(DB_PATH.read_text(encoding="utf-8"))
    except Exception: return []

def write_store(items):
    DB_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

def extract_pdf(data: bytes) -> str:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            text = "\n".join(page.get_text("text") for page in doc)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")
    if not text.strip():
        raise HTTPException(status_code=422, detail="No selectable text found. Scanned PDF OCR is not implemented in Day 1 MVP.")
    return text

def split_clauses(text: str) -> list[str]:
    # A pragmatic baseline: split on numbered clause headings and paragraph boundaries.
    normalized = re.sub(r"\r", "", text)
    chunks = re.split(r"\n\s*(?=(?:\d+(?:\.\d+)*[.)]?\s+[A-Z]|(?:ARTICLE|SECTION)\s+[IVX0-9]+) )", normalized, flags=re.I)
    if len(chunks) < 2:
        chunks = re.split(r"\n\s*\n+", normalized)
    clauses=[]
    for chunk in chunks:
        chunk=re.sub(r"\s+", " ", chunk).strip()
        if len(chunk) >= 35:
            clauses.append(chunk)
    if not clauses and normalized.strip(): clauses=[re.sub(r"\s+", " ", normalized).strip()]
    return clauses[:100]

def classify_clause(text: str) -> tuple[str, float]:
    scores={cat: sum(len(re.findall(p, text, re.I)) for p in pats) for cat,pats in CATEGORY_PATTERNS.items()}
    cat=max(scores, key=scores.get)
    if scores[cat] == 0: return "general", 0.35
    confidence=min(0.96, 0.55 + 0.12*scores[cat])
    return cat, round(confidence,2)

def retrieve_policy(category: str, clause: str) -> dict:
    # Day 1 deterministic baseline. Replace/augment with embeddings and graph-enhanced retrieval in Module 2.
    matches=[p for p in POLICIES if p["category"] == category]
    if matches: return matches[0]
    # keyword overlap fallback
    words=set(re.findall(r"[a-z]{4,}", clause.lower()))
    def score(p): return len(words & set(re.findall(r"[a-z]{4,}", p["text"].lower())))
    return max(POLICIES, key=score)

def assess_risk(text: str, category: str, policy: dict) -> dict:
    risk="low"; reasons=[]
    for level, patterns in RISK_RULES.items():
        if any(re.search(p, text, re.I) for p in patterns):
            risk=level; reasons.append("Matched a configured risk phrase: " + next(p for p in patterns if re.search(p,text,re.I)))
            break
    if category == "liability" and re.search(r"unlimited|without limitation|no cap", text, re.I):
        risk="high"; reasons.append("Liability language appears uncapped.")
    if category == "renewal" and re.search(r"automatically renew|auto.?renew", text, re.I) and not re.search(r"days prior|days' notice|written notice", text, re.I):
        if risk == "low": risk="medium"
        reasons.append("Renewal language detected without an obvious notice window.")
    if not reasons: reasons.append("No configured high-risk phrase matched. This is not a legal conclusion; review against the cited policy.")
    action={"high":"Escalate for legal review and negotiate a clearer limitation.","medium":"Review the clause against the policy and clarify obligations or notice periods.","low":"No configured trigger found; verify context and policy alignment."}[risk]
    return {"risk_level":risk,"reasons":reasons,"recommended_action":action,"policy":policy}

def analyze_pipeline(text: str) -> dict:
    clauses=[]; findings=[]; stage_log=["document_extracted"]
    for i, clause_text in enumerate(split_clauses(text), start=1):
        category, confidence=classify_clause(clause_text)
        clause={"id":f"C-{i:03d}","text":clause_text,"category":category,"classification_confidence":confidence}
        clauses.append(clause)
        policy=retrieve_policy(category, clause_text)
        finding=assess_risk(clause_text, category, policy)
        finding.update({"clause_id":clause["id"],"clause_text":clause_text,"category":category,"policy_id":policy["id"],"suggested_redline":generate_redline(clause_text, category, policy)})
        findings.append(finding)
    stage_log += ["clauses_segmented","clauses_classified","policy_retrieval_completed","risk_assessment_completed"]
    return {"clauses":clauses,"findings":findings,"stage_log":stage_log}

def generate_redline(clause: str, category: str, policy: dict) -> str:
    # Safe transparent template; not represented as an LLM-generated legal amendment.
    if category == "liability":
        return "Suggested discussion draft: Subject to mutually agreed exceptions, each party's aggregate liability should be capped at fees paid or payable under this agreement during the preceding twelve (12) months. Counsel should define appropriate carve-outs."
    if category == "renewal":
        return "Suggested discussion draft: Any renewal should be for a clearly specified period, with either party permitted to give written notice of non-renewal at least thirty (30) days before the current term expires. Confirm the period with counsel."
    if category == "termination":
        return "Suggested discussion draft: Include a written notice and reasonable opportunity to cure material breach, together with clearly defined termination rights. Counsel should tailor timing and exceptions."
    if category == "indemnity":
        return "Suggested discussion draft: Define the covered claims, procedures, exclusions, and scope of indemnity in a proportionate manner, with appropriate reciprocal protections where commercially suitable."
    if category == "payment":
        return "Suggested discussion draft: State invoice requirements, payment due dates, good-faith dispute procedures, and any applicable late charges explicitly. Confirm commercial terms before use."
    return f"Review the clause against policy {policy['id']} and propose a narrowly tailored amendment. No automatic legal amendment was generated for this category."

# LangGraph is used when installed: explicit nodes with shared state. Day 1 nodes wrap real deterministic functions.
def _extract_node(state: WorkflowState): return {"stage_log": state.get("stage_log", []) + ["extract_node_completed"]}
def _classify_node(state: WorkflowState):
    clauses=[]
    for i, txt in enumerate(split_clauses(state["text"]),1):
        cat, conf=classify_clause(txt); clauses.append({"id":f"C-{i:03d}","text":txt,"category":cat,"classification_confidence":conf})
    return {"clauses":clauses,"stage_log":state.get("stage_log",[])+["classification_agent_completed"]}
def _risk_node(state: WorkflowState):
    findings=[]
    for c in state.get("clauses",[]):
        policy=retrieve_policy(c["category"],c["text"]); risk=assess_risk(c["text"],c["category"],policy)
        risk.update({"clause_id":c["id"],"clause_text":c["text"],"category":c["category"],"policy_id":policy["id"],"suggested_redline":generate_redline(c["text"],c["category"],policy)})
        findings.append(risk)
    return {"findings":findings,"stage_log":state.get("stage_log",[])+["risk_analysis_agent_completed","redlining_agent_completed"]}

def build_graph():
    if not LANGGRAPH_AVAILABLE: return None
    g=StateGraph(WorkflowState)
    g.add_node("document_processing",_extract_node)
    g.add_node("clause_classification_agent",_classify_node)
    g.add_node("risk_and_redlining_agents",_risk_node)
    g.set_entry_point("document_processing")
    g.add_edge("document_processing","clause_classification_agent")
    g.add_edge("clause_classification_agent","risk_and_redlining_agents")
    g.add_edge("risk_and_redlining_agents",END)
    return g.compile()

GRAPH=build_graph()

@app.post("/api/llm-test")
async def llm_test(body: LLMTestRequest):
    """
    Test the connection between FastAPI and local Ollama.
    """

    try:
        answer = await generate_text(body.prompt)

        return {
            "success": True,
            "model": "qwen3:4b",
            "response": answer
        }

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Ollama timed out. Try again or check model performance."
        )

    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Check that Ollama is running."
        )

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned HTTP {exc.response.status_code}."
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc)
        )

@app.get("/api/health")
def health():
    return {"status":"ok","service":"ContractSentinel API","langgraph_enabled":LANGGRAPH_AVAILABLE,"workflow":"LangGraph" if GRAPH else "deterministic fallback","version":"0.1.0"}

@app.get("/api/policies")
def policies(): return POLICIES

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Upload a PDF file.")
    data=await file.read()
    if len(data)>15*1024*1024: raise HTTPException(status_code=413, detail="File exceeds the 15 MB MVP limit.")
    text=extract_pdf(data)
    if GRAPH:
        result=GRAPH.invoke({"text":text,"stage_log":["pdf_extracted"]})
        clauses=result.get("clauses",[]); findings=result.get("findings",[]); stages=result.get("stage_log",[])
    else:
        result=analyze_pipeline(text); clauses=result["clauses"]; findings=result["findings"]; stages=result["stage_log"]
    review_id=str(uuid.uuid4())
    graph_nodes=[]; graph_edges=[]
    for c in clauses:
        graph_nodes.append({"id":c["id"],"label":c["category"].title(),"type":"clause","detail":c["text"][:180]})
    seen=set()
    for f in findings:
        pid=f["policy_id"]
        p=f["policy"]
        if pid not in seen:
            graph_nodes.append({"id":pid,"label":p["title"],"type":"policy","detail":p["text"]}); seen.add(pid)
        graph_edges.append({"source":f["clause_id"],"target":pid,"label":"checked against"})
    review={"review_id":review_id,"filename":file.filename,"created_at":datetime.now(timezone.utc).isoformat(),"clauses":clauses,"findings":findings,"graph":{"nodes":graph_nodes,"edges":graph_edges},"stage_log":stages,"decisions":[],"disclaimer":"Prototype for educational research only. Findings and suggested wording require review by qualified counsel."}
    items=read_store(); items.insert(0,review); write_store(items)
    return review

@app.get("/api/reviews")
def reviews():
    return [{"review_id":r["review_id"],"filename":r["filename"],"created_at":r["created_at"],"finding_count":len(r["findings"]),"high_risk_count":sum(f["risk_level"]=="high" for f in r["findings"]),"medium_risk_count":sum(f["risk_level"]=="medium" for f in r["findings"])} for r in read_store()]

@app.get("/api/reviews/{review_id}")
def review_detail(review_id: str):
    r=next((x for x in read_store() if x["review_id"]==review_id),None)
    if not r: raise HTTPException(status_code=404,detail="Review not found")
    return r

@app.post("/api/reviews/{review_id}/decision")
def save_decision(review_id: str, body: ReviewDecision):
    items=read_store(); review=next((x for x in items if x["review_id"]==review_id),None)
    if not review: raise HTTPException(status_code=404,detail="Review not found")
    if body.review_id != review_id: raise HTTPException(status_code=400,detail="Review ID mismatch")
    if not any(f["clause_id"]==body.clause_id for f in review["findings"]): raise HTTPException(status_code=404,detail="Clause not found in review")
    decision={"clause_id":body.clause_id,"decision":body.decision,"reviewer_note":body.reviewer_note,"timestamp":datetime.now(timezone.utc).isoformat()}
    review["decisions"]=[d for d in review.get("decisions",[]) if d["clause_id"]!=body.clause_id]+[decision]
    write_store(items)
    return {"ok":True,"decision":decision}

@app.get("/api/reviews/{review_id}/report")
def report(review_id: str):
    r=next((x for x in read_store() if x["review_id"]==review_id),None)
    if not r: raise HTTPException(status_code=404,detail="Review not found")
    return {"title":"ContractSentinel Review Report","review_id":r["review_id"],"filename":r["filename"],"created_at":r["created_at"],"disclaimer":r["disclaimer"],"findings":r["findings"],"decisions":r.get("decisions",[])}
