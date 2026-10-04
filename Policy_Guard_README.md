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