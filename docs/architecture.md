# ContractSentinel architecture — Day 1 baseline

## Four planned technical modules
1. Contract Intelligence: PDF text extraction, segmentation, clause categorization; planned trained TF-IDF/SVM baseline.
2. Graph RAG: policy retrieval and clause-policy graph. Day 1 retrieval is category/keyword-based; graph represents actual returned links. Embedding-based retrieval and graph-enhanced retrieval comparison remain future work.
3. Multi-Agent Risk Engine: LangGraph currently coordinates document-processing, clause-classification, and risk/redlining nodes. The current nodes use deterministic baseline logic, not autonomous LLM agents.
4. Redlining & Governance: suggested discussion drafts, approve/reject decisions, persisted review records, JSON report export.

## Current limitations
- Text-based PDFs only; no OCR.
- Heuristic categorization and risk phrases are not validated legal classifiers.
- Suggested amendments are templates, not model-generated legal language.
- Graph is a clause-policy relationship view, not yet a complete Graph RAG implementation.
- No real contract data should be uploaded without permission. Use synthetic/sample contracts.
