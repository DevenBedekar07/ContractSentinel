import type { ReactNode } from 'react';
import { motion } from 'framer-motion';
import { ArrowDown, ArrowRight, ArrowUpRight, Fingerprint, FileCheck2, FileSearch, GitBranch, Network, ScanSearch, ShieldCheck, Workflow } from 'lucide-react';
import KnowledgeGraphBackground from '../components/KnowledgeGraphBackground';

const steps = [
  { n:'01', title:'Upload the contract', text:'Start with a text-based PDF and create a review workspace.', icon:FileSearch },
  { n:'02', title:'Inspect the findings', text:'Examine detected clauses, risk signals, and supporting text.', icon:ScanSearch },
  { n:'03', title:'Connect the evidence', text:'Explore clause-to-policy relationships available in the current review.', icon:Network },
  { n:'04', title:'Review proposed changes', text:'Evaluate discussion drafts and record a human decision.', icon:FileCheck2 },
];
const features = [
  { n:'01', title:'Contract risk review', text:'Bring clause text, risk signals, and review rationale into one focused workspace.', icon:ShieldCheck },
  { n:'02', title:'Policy matching', text:'Inspect the policy reference associated with a finding and understand the potential gap.', icon:GitBranch },
  { n:'03', title:'Relationship explorer', text:'Visualise clause-to-policy connections from the current review in an interactive graph.', icon:Network },
  { n:'04', title:'Human-led redlining', text:'Compare original wording with proposed discussion drafts and keep decisions explicit.', icon:FileCheck2 },
];
function Label({children}:{children:ReactNode}) { return <div className="cs-section-label"><span />{children}</div>; }

export default function LandingPage() {
  return <main className="cs-landing">
    <header className="cs-nav">
      <a className="cs-brand" href="/" aria-label="ContractSentinel home"><span className="cs-brand-mark"><Fingerprint size={21}/></span><span className="cs-brand-copy"><strong>CONTRACT<span>SENTINEL</span></strong><small>CONTRACT INTELLIGENCE</small></span></a>
      <nav className="cs-nav-links" aria-label="Main navigation"><a href="#overview">Overview</a><a href="#workflow">How it works</a><a href="#features">Features</a><a href="#technology">Technology</a></nav>
      <a className="cs-nav-cta" href="/app">Launch application <ArrowUpRight size={15}/></a>
    </header>

    <section className="cs-hero">
      <div className="cs-hero-grid" aria-hidden="true"/><KnowledgeGraphBackground/>
      <div className="cs-hero-content">
        <motion.div className="cs-kicker" initial={{opacity:0,y:10}} animate={{opacity:1,y:0}} transition={{duration:.55}}><span className="cs-status-dot"/> AI-ASSISTED CONTRACT INTELLIGENCE</motion.div>
        <motion.h1 initial={{opacity:0,y:18}} animate={{opacity:1,y:0}} transition={{duration:.65,delay:.08}}>Clarity before<br/><em>commitment.</em></motion.h1>
        <motion.p className="cs-hero-copy" initial={{opacity:0,y:16}} animate={{opacity:1,y:0}} transition={{duration:.65,delay:.18}}>Understand contractual exposure, connect clauses to policies, and review proposed changes through a focused, human-led workflow.</motion.p>
        <motion.div className="cs-hero-actions" initial={{opacity:0,y:12}} animate={{opacity:1,y:0}} transition={{duration:.6,delay:.27}}><a className="cs-button-primary" href="/app">Launch application <ArrowRight size={16}/></a><a className="cs-button-quiet" href="#overview">Explore the platform <ArrowDown size={15}/></a></motion.div>
        <div className="cs-hero-footnote"><ShieldCheck size={14}/> AI-assisted review. Human decisions remain essential.</div>
      </div>
      <div className="cs-hero-side-note"><span>01 / CONTRACT</span><i/><span>02 / EVIDENCE</span><i/><span>03 / DECISION</span></div>
      <a className="cs-scroll-cue" href="#overview"><span>SCROLL TO EXPLORE</span><ArrowDown size={14}/></a>
    </section>

    <section className="cs-section cs-overview" id="overview">
      <div className="cs-overview-copy"><Label>THE PLATFORM</Label><h2>Contract review,<br/><em>with context.</em></h2><p>ContractSentinel brings clause-level findings, policy references, relationship visualisation, and proposed discussion drafts into one review experience.</p><a className="cs-inline-link" href="/app">Explore the workspace <ArrowUpRight size={16}/></a></div>
      <div className="cs-overview-panel"><div className="cs-panel-top"><span>REVIEW WORKFLOW</span><span className="cs-panel-indicator">HUMAN-IN-THE-LOOP</span></div>
        <div className="cs-evidence-row"><span className="cs-evidence-icon"><FileSearch size={17}/></span><div><b>Clause evidence</b><small>Inspect the source wording</small></div><span className="cs-row-state">01</span></div><div className="cs-evidence-connector"/>
        <div className="cs-evidence-row"><span className="cs-evidence-icon"><GitBranch size={17}/></span><div><b>Policy relationship</b><small>Review the matched reference</small></div><span className="cs-row-state">02</span></div><div className="cs-evidence-connector"/>
        <div className="cs-evidence-row"><span className="cs-evidence-icon"><ShieldCheck size={17}/></span><div><b>Reviewer decision</b><small>Approve or reject the suggestion</small></div><span className="cs-row-state">03</span></div>
        <div className="cs-panel-bottom"><span className="cs-mini-dot"/> EVIDENCE FIRST · DECISIONS BY PEOPLE</div>
      </div>
    </section>

    <section className="cs-section cs-workflow" id="workflow"><div className="cs-section-heading"><div><Label>THE PROCESS</Label><h2>From document<br/>to <em>informed review.</em></h2></div><p>A clear sequence keeps evidence visible and leaves final decisions with the reviewer.</p></div>
      <div className="cs-steps">{steps.map((s,i)=>{const Icon=s.icon;return <motion.article className="cs-step" key={s.n} initial={{opacity:0,y:22}} whileInView={{opacity:1,y:0}} viewport={{once:true,amount:.25}} transition={{duration:.5,delay:i*.08}}><div className="cs-step-top"><span>{s.n}</span><Icon size={20}/></div><h3>{s.title}</h3><p>{s.text}</p><div className="cs-step-line"><span/></div></motion.article>;})}</div>
    </section>

    <section className="cs-section cs-features" id="features"><div className="cs-section-heading"><div><Label>BUILT FOR REVIEW</Label><h2>Evidence over<br/><em>black boxes.</em></h2></div><p>Designed to help reviewers inspect the reasons behind a flag—not just receive a score.</p></div>
      <div className="cs-feature-grid">{features.map((f,i)=>{const Icon=f.icon;return <motion.article className="cs-feature-card" key={f.n} initial={{opacity:0,y:18}} whileInView={{opacity:1,y:0}} viewport={{once:true,amount:.2}} transition={{duration:.45,delay:i*.07}}><div className="cs-feature-top"><span>{f.n}</span><Icon size={20}/></div><h3>{f.title}</h3><p>{f.text}</p><div className="cs-feature-arrow"><ArrowUpRight size={16}/></div></motion.article>;})}</div>
    </section>

    <section className="cs-section cs-technology" id="technology"><div className="cs-tech-copy"><Label>UNDER THE HOOD</Label><h2>A connected<br/><em>review workflow.</em></h2><p>The interface is built around a local API and a modular analysis workflow. This is a high-level product view; verify individual components in the running implementation.</p></div>
      <div className="cs-architecture"><div className="cs-arch-node"><span><FileSearch size={17}/></span><div><b>Contract input</b><small>PDF document</small></div></div><div className="cs-arch-line"><span/></div>
        <div className="cs-arch-node cs-arch-core"><span><Workflow size={17}/></span><div><b>Analysis workflow</b><small>API · clause analysis · policy matching</small></div></div><div className="cs-arch-branches"><i/><i/><i/></div>
        <div className="cs-arch-output"><div className="cs-arch-node"><span><ShieldCheck size={16}/></span><div><b>Risk findings</b><small>Evidence and rationale</small></div></div><div className="cs-arch-node"><span><Network size={16}/></span><div><b>Relationships</b><small>Clause-to-policy graph</small></div></div><div className="cs-arch-node"><span><FileCheck2 size={16}/></span><div><b>Human review</b><small>Decisions and redlines</small></div></div></div>
        <div className="cs-arch-caption"><span className="cs-mini-dot"/> MODULAR BY DESIGN</div>
      </div>
    </section>

    <section className="cs-final-cta"><div className="cs-cta-glow" aria-hidden="true"/><Label>MAKE THE NEXT REVIEW CLEARER</Label><h2>See the evidence.<br/><em>Make the decision.</em></h2><p>Open the ContractSentinel workspace to review a contract and inspect the available findings.</p><a className="cs-button-primary" href="/app">Enter the workspace <ArrowRight size={16}/></a></section>
    <footer className="cs-footer"><a className="cs-brand" href="/"><span className="cs-brand-mark"><Fingerprint size={20}/></span><span className="cs-brand-copy"><strong>CONTRACT<span>SENTINEL</span></strong><small>CONTRACT INTELLIGENCE</small></span></a><p>AI-assisted contract review · Human judgement required.</p><div><a href="#overview">Platform</a><a href="#workflow">Workflow</a><a href="#features">Features</a><a href="/app">Launch app <ArrowUpRight size={13}/></a></div><span className="cs-footer-build">PROJECT PREVIEW · BUILD 0.1.0</span></footer>
  </main>;
}
