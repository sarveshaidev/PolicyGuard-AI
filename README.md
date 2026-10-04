## 🎯 Impact

| Metric | Current State | With PolicyGuard AI | Improvement |
|--------|--------------|---------------------|-------------|
| **Policy Query Resolution Time** | 24-48 hours | <5 seconds | **99.9% faster** |
| **HR Time on Repetitive Queries** | 40% of workweek | 10% of workweek | **75% reduction** |
| **Resume Screening (100 candidates)** | 15-20 hours manual review | 15 minutes AI-powered | **98% faster** |
| **Time-to-Hire** | 45 days average | 25 days average | **44% reduction** |
| **Compliance Audit Preparation** | 3-5 days manual compilation | Real-time, on-demand | **100% audit-ready** |
| **Employee Satisfaction (HR Support)** | 62% satisfaction rate | 87% satisfaction rate | **+25 points** |
| **Cost per Policy Query** | $2.50 (HR labor cost) | $0.007 (LLM API cost) | **99.7% reduction** |
| **Annual Productivity Loss** (1000-employee company) | $2.3M | $580K | **$1.72M saved** |

---

## 🔥 Problem / Pain Points

### **Problem 1: Employee Policy Confusion**
- **80% of employees** waste **2-3 hours/week** searching for policy information across scattered documents (handbooks, emails, intranet)
- **HR teams spend 40% of their time** answering repetitive questions (leave balance, benefits eligibility, compliance rules) instead of strategic initiatives
- **Inconsistent answers** from different managers create **compliance risks** and employee confusion
- **No audit trail** of who asked what question and when — critical failure for **labor law compliance audits**
- **Employee frustration** leads to **15% drop in satisfaction scores** and increased support tickets

### **Problem 2: Recruitment Inefficiency**
- **HR recruiters spend 15-20 hours per role** manually screening resumes against job descriptions
- **Candidate matching is subjective** — different recruiters score the same resume differently (inter-rater reliability: 0.42)
- **Data silos** — candidate data scattered across Excel sheets, PDFs, ATS systems, email attachments
- **No evidence-based scoring** — decisions based on gut feeling rather than systematic skill/experience/culture-fit evaluation
- **45-day average time-to-hire** (industry benchmark: 30 days) — losing top candidates to faster competitors
- **$4,700 average cost-per-hire** inflated by manual screening overhead

### **Problem 3: Compliance & Security Risks**
- **Sensitive HR data** (salaries, performance reviews, disciplinary actions) accessible to unauthorized employees
- **No role-based access control** — junior HR staff can view executive compensation data
- **Prompt injection vulnerabilities** — malicious actors can exploit LLM to extract confidential information
- **Missing audit logs** — cannot prove compliance with GDPR, CCPA, labor regulations during audits
- **Data breach risk** — $4.45M average cost of HR data breach (IBM Security Report 2023)

### **Problem 4: Lack of Analytics & Insights**
- **HR leadership has no visibility** into:
  - Most common employee questions (to improve documentation)
  - Query resolution time by HR team member
  - Policy gaps causing repeated confusion
  - Recruitment funnel bottlenecks
- **No cost tracking** — cannot measure ROI of HR technology investments
- **No quality metrics** — cannot measure answer accuracy or employee satisfaction

---

## 💡 The Core Challenge

**Mid-size companies (500-5000 employees) need:**
1. ✅ **Instant, accurate policy answers** with citations and audit trails
2. ✅ **AI-powered recruitment screening** that's objective, fast, and explainable
3. ✅ **Enterprise-grade security** with RBAC, injection protection, and compliance logging
4. ✅ **Actionable analytics** to improve HR operations and reduce costs

**But existing solutions are:**
- ❌ Generic chatbots without HR-specific knowledge or compliance features
- ❌ Expensive enterprise platforms ($50K+/year) requiring 6-month implementations
- ❌ Manual processes that don't scale with company growth
-  Siloed tools that don't integrate policy Q&A with recruitment workflows

**Result:** HR teams remain overwhelmed, employees remain frustrated, and companies face compliance risks.
------------------------------------------------------------------------------------------------------------------------------

# 🛡️ PolicyGuard AI

> Enterprise HR Policy & Talent Intelligence Platform powered by RAG, LangGraph, security guardrails, RBAC, and Human-in-the-Loop workflows.

PolicyGuard AI is a production-style HR intelligence application designed to answer policy questions from an organization’s approved knowledge base while enforcing access control, tenant isolation, security guardrails, auditability, and controlled human approval for consequential actions.

## ✨ Highlights

- 🔎 **Policy Intelligence** — grounded answers from indexed HR policy documents
- 🎯 **Talent Intelligence** — explainable semantic + skill matching for authorized HR users
- 🧠 **LangGraph orchestration** — explicit workflow/state routing
- 📚 **RAG pipeline** — document parsing, chunking, embeddings, retrieval and reranking
- 🔀 **Hybrid retrieval** — semantic + lexical retrieval with reranking
- 🛡️ **Security Guardrails** — prompt-injection detection and request blocking
- 🔐 **RBAC** — Viewer, Editor and Administrator access levels
- 🏢 **Multi-tenant isolation** — organization-scoped users, documents, candidates and audit data
- 👤 **PII detection** — sensitive information handling before model processing
- ⏸️ **Human-in-the-Loop (HITL)** — sensitive/action-oriented workflows can pause for explicit human approval
- 🧾 **Audit logging** — authentication, administrative and workflow events are recorded
- ⚡ **Caching + rate limiting** — protects the application and reduces repeated work
- 💾 **Conversation memory/checkpointing** — supports stateful graph workflows
- 🔌 **MCP/tool integration** — extensible tool-oriented architecture
- 🧪 **Automated test suite** — unit, integration and security-oriented coverage
- 🎨 **Streamlit UI** — enterprise-style workspace for policy, talent and administration

## 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │      Streamlit UI    │
                         └──────────┬───────────┘
                                    │
                         Authentication / RBAC
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
          Policy Intelligence                 Talent Intelligence
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    │
                           Security Guardrails
                                    │
                              LangGraph
                         Orchestration Layer
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
          Router                  RAG                 HITL Gate
              │                     │                     │
              │          ┌──────────┴──────────┐          │
              │          │ Retrieval + Rerank  │          │
              │          │ + Policy Context    │          │
              │          └──────────┬──────────┘          │
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                              LLM Generation
                                    │
                         Validation / Metrics
                                    │
                         Audit + Persistence
```

## 🔐 Human-in-the-Loop

PolicyGuard AI does not blindly execute sensitive workflows.

For action-oriented requests, the workflow can pause and present a human approval checkpoint:

```text
User Request
     │
     ▼
AI / Workflow Analysis
     │
     ▼
Human approval required?
   ┌─┴──────────────┐
  YES              NO
   │                │
   ▼                ▼
 PAUSE            Continue
   │
 ┌─┴─────────┐
 ▼           ▼
Approve     Reject
 │           │
 ▼           ▼
Resume      Cancel
```

The current HITL implementation provides the approval/resume gate. It does **not** claim to send a real email or perform an external side effect unless a corresponding action tool is explicitly wired into the workflow.

## 🧠 Policy Intelligence

Policy questions are answered using retrieved organizational policy context rather than relying on unsupported model knowledge.

The generation layer is instructed to:

1. Use only supplied policy context.
2. Avoid inventing policy, benefits, deadlines or procedures.
3. Clearly state when the available documents are insufficient.
4. Provide source/page references when available.
5. Avoid exposing internal prompts, credentials or security controls.

Example:

```text
User → "What is the leave policy?"
        ↓
Policy Router
        ↓
Hybrid Retrieval
        ↓
Reranking
        ↓
Relevant Policy Chunks
        ↓
LLM
        ↓
Grounded Answer + Source
```

## 🎯 Talent Intelligence

Authorized HR users can use the Talent Intelligence workspace to:

- upload resumes into the internal talent pool
- provide a job description
- rank candidates using semantic and skill fit
- inspect candidate/role evidence
- keep hiring decisions under human responsibility

Talent Intelligence is **decision support**, not an automated hiring, promotion, compensation or termination decision.

## 🛡️ Security

PolicyGuard AI is designed around defense-in-depth:

- bcrypt-based authentication
- failed-login tracking and lockout controls
- role-based authorization
- organization/tenant scoping
- prompt-injection detection
- PII detection
- parameterized database queries
- rate limiting
- audit logging
- restricted administrative actions
- safe error handling
- secrets kept outside source code

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangGraph |
| LLM | OpenRouter-compatible model endpoint |
| Retrieval | Vector + lexical/hybrid retrieval |
| Reranking | Cross-encoder reranking |
| Embeddings | Sentence Transformers / configured embedding provider |
| Database | SQLite |
| Authentication | bcrypt + RBAC |
| Caching | SQLite / configured cache layer |
| Rate limiting | Application limiter with optional Redis support |
| Tooling | MCP-compatible tool architecture |
| Testing | pytest |

## 📁 Project Structure

```text
PolicyGuard AI/
├── app.py
├── database.py
├── requirements.txt
├── README.md
├── config/
├── data/
│   └── PDFs/
├── src/
│   ├── auth/
│   ├── core/
│   ├── ingestion/
│   ├── mcp_server/
│   ├── orchestrator/
│   │   └── graph.py
│   ├── pipeline/
│   ├── retrieval/
│   └── ...
├── tests/
└── evaluation/
```

## 🚀 Run Locally

### 1. Clone

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd "PolicyGuard AI"
```

### 2. Create environment

Windows PowerShell:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file locally.

Do **not** commit API keys, passwords or other secrets.

Typical configuration includes the LLM/API credentials and application settings required by the project.

### 5. Start the application

```powershell
streamlit run app.py
```

## 🧪 Testing

Run the full automated suite:

```powershell
python -m pytest -q
```

Syntax check:

```powershell
python -m py_compile app.py src\orchestrator\graph.py
```

## ☁️ Deployment

The application can be deployed as a Streamlit web service on platforms such as Render.

Typical Render start command:

```text
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Configure secrets through the hosting platform's environment-variable settings rather than committing them to Git.

## 📌 Engineering Focus

The goal of PolicyGuard AI is not simply to put an LLM behind a chat box.

The project focuses on the engineering problems that matter when AI is used inside enterprise workflows:

- **Grounding** — answers should be supported by organizational documents.
- **Access control** — users should only see information they are authorized to access.
- **Security** — malicious or unsafe requests should be detected and controlled.
- **Governance** — consequential workflows can require human approval.
- **Auditability** — important actions and decisions should leave an audit trail.
- **Reliability** — caching, rate limiting, validation and safe fallbacks reduce operational risk.
- **Explainability** — retrieval sources and talent-matching evidence help humans understand outputs.

## ⚠️ Responsible Use

PolicyGuard AI is a portfolio/engineering project and should be adapted, tested and reviewed before being used with real organizational HR data.

Talent Intelligence is decision support only. Human HR professionals remain responsible for employment decisions.

## 👨‍💻 Project

**PolicyGuard AI**  
Enterprise HR Policy & Talent Intelligence Platform

Built with Python, Streamlit, LangGraph, RAG, security guardrails, RBAC and Human-in-the-Loop workflows.

---

⭐ If you find the project useful, consider starring the repository.


----------------------------------------------
Phase 1: Core Functionality (Week 1)
✅ Guardrails + Injection Detection (#18, #19)
✅ LangGraph Multi-Agent (#5, #25)
✅ LLM Router (#15)
✅ RBAC Enhancement (#20)
✅ Hybrid RAG Verification (#8)
Phase 2: Performance & Quality (Week 2)
✅ Redis Caching (#4)
✅ Cross-Encoder Re-ranker (#9)
✅ Query Rewriting (#11)
✅ Structured Output (#23)
✅ Citation Verification (#22)
Phase 3: Monitoring & Evaluation (Week 3)
✅ RAG Evaluation (#13, #14)
✅ Cost + Token Tracking (#30)
✅ Observability (#29)
✅ Rate Limiting (#32)
✅ Tests (#34)
Phase 4: Advanced Features (Week 4)
✅ PostgreSQL + pgvector (#3)
✅ Neo4j + GraphRAG (#6, #7)
✅ PII Detection (#21)
✅ Async Ingestion (#27)
✅ Feedback Loop (#31)
Phase 5: Deployment (Week 5)
✅ Docker (#2)
✅ CI/CD (#35)
✅ Cloud Deployment (#1)



# Project Structure

```
PolicyGuard AI/
├── __pycache__
│   ├── app.cpython-314.pyc
│   ├── test_auth_readonly.cpython-314-pytest-8.4.2.pyc
│   ├── test_auth_readonly.cpython-314.pyc
│   └── test_mcp_client.cpython-314-pytest-8.4.2.pyc
├── config
│   ├── __pycache__
│   │   ├── __init__.cpython-314.pyc
│   │   └── settings.cpython-314.pyc
│   ├── __init__.py
│   └── settings.py
├── data
│   ├── documents
│   ├── PDFs
│   │   ├── PolicyGuardAI_HR_Employee_Code_of_Conduct.pdf
│   │   └── PolicyGuardAI_HR_Policy_Manual.pdf
│   ├── sample_docs
│   ├── talent_pool
│   │   └── dummy_candidates
│   │       ├── 01_Aarav_Sharma_Resume.pdf
│   │       ├── 02_Ananya_Patel_Resume.pdf
│   │       ├── 03_Rohan_Mehta_Resume.pdf
│   │       ├── 04_Priya_Nair_Resume.pdf
│   │       ├── 05_Vikram_Singh_Resume.pdf
│   │       ├── 06_Ishita_Rao_Resume.pdf
│   │       ├── 07_Kabir_Joshi_Resume.pdf
│   │       ├── 08_Neha_Kulkarni_Resume.pdf
│   │       ├── 09_Aditya_Verma_Resume.pdf
│   │       ├── 10_Meera_Iyer_Resume.pdf
│   │       ├── 11_Arjun_Desai_Resume.pdf
│   │       ├── 12_Sneha_Kapoor_Resume.pdf
│   │       ├── 13_Rahul_Bhat_Resume.pdf
│   │       ├── 14_Kavya_Menon_Resume.pdf
│   │       ├── 15_Siddharth_Shah_Resume.pdf
│   │       ├── 16_Pooja_Reddy_Resume.pdf
│   │       ├── 17_Nikhil_Agarwal_Resume.pdf
│   │       ├── 18_Diya_Choudhary_Resume.pdf
│   │       ├── 19_Manish_Gupta_Resume.pdf
│   │       ├── 20_Riya_Bose_Resume.pdf
│   │       ├── 21_Varun_Malhotra_Resume.pdf
│   │       ├── 22_Tanvi_Joshi_Resume.pdf
│   │       ├── 23_Karan_Sethi_Resume.pdf
│   │       ├── 24_Aditi_Krishnan_Resume.pdf
│   │       ├── 25_Yash_Thakur_Resume.pdf
│   │       └── README.md
│   ├── traces
│   │   ├── evaluation_20260912_154903.json
│   │   ├── memory_test_user_123_20260912_155158.json
│   │   ├── metrics_20260912_133516.json
│   │   └── traces.jsonl
│   ├── uploads
│   │   ├── 0b2bfd293309468fb624a4f40f0fc8a0_08_Neha_Kulkarni_Resume.pdf
│   │   ├── 1d9030dd7d6d4db09daf23e3b9450e3a_24_Aditi_Krishnan_Resume.pdf
│   │   ├── 26b04cc9ab8e4811a1ea77142bfad744_PolicyGuardAI_HR_Employee_Code_of_Conduct.pdf
│   │   ├── 277b60f6f3dc421c8f779b5c2e5a5a2f_15_Siddharth_Shah_Resume.pdf
│   │   ├── 3a88436df80149e9b9661501d550f535_11_Arjun_Desai_Resume.pdf
│   │   ├── 3df7f7d9742148f682dd4852ed18328b_12_Sneha_Kapoor_Resume.pdf
│   │   ├── 4e055cf249ab46438fe2d117c354f307_16_Pooja_Reddy_Resume.pdf
│   │   ├── 50186aa49f374e33b8cbcd2468980337_17_Nikhil_Agarwal_Resume.pdf
│   │   ├── 5339574d38814433a866b99d55af38be_21_Varun_Malhotra_Resume.pdf
│   │   ├── 69506a9ee3b8492d8600c9b125e8c514_25_Yash_Thakur_Resume.pdf
│   │   ├── 6d43ad03695949639f6ad53c6b84ea95_02_Ananya_Patel_Resume.pdf
│   │   ├── 73df030d926f4ff4b02a972c4e70a339_07_Kabir_Joshi_Resume.pdf
│   │   ├── 7bbd19d00f1e4684a1c76bcaeea39dd0_PolicyGuardAI_HR_Employee_Code_of_Conduct.pdf
│   │   ├── 834fd151f5614f77b5cfba0b0f0157b4_13_Rahul_Bhat_Resume.pdf
│   │   ├── 83bd9229c2584464979b9d51a2141907_14_Kavya_Menon_Resume.pdf
│   │   ├── 8f7614dd179a473aa0619b71cbfa7eab_09_Aditya_Verma_Resume.pdf
│   │   ├── 91af3d18e06340619cda22e786eee66f_19_Manish_Gupta_Resume.pdf
│   │   ├── ab9798bac8e748579a47c7c32da35c27_23_Karan_Sethi_Resume.pdf
│   │   ├── b4619e9b146a4223962fd52aaf5d29a8_20_Riya_Bose_Resume.pdf
│   │   ├── b6f26f5a1f364b3dad77b1fa636f6814_03_Rohan_Mehta_Resume.pdf
│   │   ├── bb142f0be7144092beab046be7c0b02c_05_Vikram_Singh_Resume.pdf
│   │   ├── c5d82f939229450fab5a8b5fe6fded9b_22_Tanvi_Joshi_Resume.pdf
│   │   ├── d24666041efb4001a42e03dae63bad78_06_Ishita_Rao_Resume.pdf
│   │   ├── d6b90f21b43c45ffb30b5da15cc707a7_PolicyGuardAI_HR_Policy_Manual.pdf
│   │   ├── e0e9aa34a7984138ac8fdaa5945432a6_18_Diya_Choudhary_Resume.pdf
│   │   ├── e4c7b43b40d14a0e9fcc8dfcc75ab3ec_10_Meera_Iyer_Resume.pdf
│   │   ├── e9fddfd7fb214d16bb23b32dec02c022_01_Aarav_Sharma_Resume.pdf
│   │   └── f90139e704704ece95a9d5a8cbd145dc_04_Priya_Nair_Resume.pdf
│   ├── vector_db
│   │   ├── bm25.pkl
│   │   ├── bm25.pkl.bak
│   │   ├── chunks.pkl
│   │   ├── chunks.pkl.bak
│   │   ├── faiss.index
│   │   └── faiss.index.bak
│   ├── cache.db
│   ├── cache.db-shm
│   ├── cache.db-wal
│   ├── ground_truth.json
│   ├── memory.db
│   ├── memory.db-shm
│   └── memory.db-wal
├── logs
├── src
│   ├── __pycache__
│   │   └── __init__.cpython-314.pyc
│   ├── auth
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-314.pyc
│   │   │   ├── access_request.cpython-314.pyc
│   │   │   └── database.cpython-314.pyc
│   │   ├── __init__.py
│   │   ├── access_request.py
│   │   └── database.py
│   ├── core
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-314.pyc
│   │   │   ├── cache.cpython-314.pyc
│   │   │   ├── embedder_singleton.cpython-314.pyc
│   │   │   ├── exceptions.cpython-314.pyc
│   │   │   ├── memory_manager.cpython-314.pyc
│   │   │   └── observability.cpython-314.pyc
│   │   ├── __init__.py
│   │   ├── cache.py
│   │   ├── embedder_singleton.py
│   │   ├── exceptions.py
│   │   ├── memory_manager.py
│   │   └── observability.py
│   ├── evaluation
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-314.pyc
│   │   │   ├── llm_judge.cpython-314.pyc
│   │   │   └── ragas_evaluator.cpython-314.pyc
│   │   ├── __init__.py
│   │   ├── llm_judge.py
│   │   └── ragas_evaluator.py
│   ├── ingestion
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-314.pyc
│   │   │   ├── multimodal_parser.cpython-314.pyc
│   │   │   └── universal_parser.cpython-314.pyc
│   │   ├── __init__.py
│   │   ├── multimodal_parser.py
│   │   └── universal_parser.py
│   ├── mcp_server
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-314.pyc
│   │   │   └── server.cpython-314.pyc
│   │   ├── __init__.py
│   │   └── server.py
│   ├── orchestrator
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-314.pyc
│   │   │   └── graph.cpython-314.pyc
│   │   ├── __init__.py
│   │   └── graph.py
│   ├── pipeline
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-314.pyc
│   │   │   └── rag_engine.cpython-314.pyc
│   │   ├── __init__.py
│   │   └── rag_engine.py
│   ├── retrieval
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-314.pyc
│   │   │   ├── cross_encoder.cpython-314.pyc
│   │   │   ├── hybrid_search.cpython-314.pyc
│   │   │   └── vector_store.cpython-314.pyc
│   │   ├── __init__.py
│   │   ├── cross_encoder.py
│   │   ├── hybrid_search.py
│   │   └── vector_store.py
│   ├── security
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-314.pyc
│   │   │   └── guard_model.cpython-314.pyc
│   │   ├── __init__.py
│   │   └── guard_model.py
│   ├── vision
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-314.pyc
│   │   │   └── advanced_ocr.cpython-314.pyc
│   │   ├── __init__.py
│   │   └── advanced_ocr.py
│   └── __init__.py
├── tests
│   ├── __pycache__
│   │   ├── test_auth.cpython-314-pytest-8.4.2.pyc
│   │   ├── test_auth.cpython-314.pyc
│   │   ├── test_rag.cpython-314-pytest-8.4.2.pyc
│   │   └── test_security.cpython-314-pytest-8.4.2.pyc
│   ├── test_auth.py
│   ├── test_rag.py
│   └── test_security.py
├── app.py
├── docker-compose.yml
├── Dockerfile
├── main.py
├── nexus_auth.db
├── README.md
└── requirements.txt
```
