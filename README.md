# OmniMind — Agentic RAG v2

A production-grade **Universal Document Intelligence Hub** — upload PDFs, Word docs, web URLs, or YouTube videos into isolated workspaces, then chat with your data using hybrid retrieval, cross-encoder reranking, and conversation memory.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange)](https://github.com/langchain-ai/langgraph)

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph v2 Pipeline                     │
│                                                              │
│  ┌──────────┐   route=rag    ┌──────────┬──────────────────┐ │
│  │  Router  │──────────────►│ Retrieve │  Vector + BM25   │ │
│  │ (heuristic)│             │  (hybrid)│  RRF Fusion       │ │
│  └────┬─────┘               └────┬─────┴──────────────────┘ │
│       │ route=chat               │                            │
│       │ route=web                ▼                            │
│       │               ┌──────────────────┐                   │
│       │               │    Reranker      │ Cross-Encoder     │
│       │               │  (ms-marco CPU)  │ ~80ms, no API     │
│       │               └────────┬─────────┘                   │
│       │                        │                              │
│       └────────────────────────▼                             │
│                        ┌──────────────────┐                  │
│                        │    Generate      │ + History inject  │
│                        │  Primary → Backup│ + Citation enforce│
│                        └────────┬─────────┘                  │
│                                 │                             │
│                        ┌────────▼─────────┐                  │
│                        │ Citation Validate │                  │
│                        └──────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
    │                     │
    ▼                     ▼
Memory Store          Workspace Manager
    (10 turns/session)    (JSON + Qdrant filter)
```

---

## Stack

| Layer | Technology | Notes |
|---|---|---|
| Orchestration | LangGraph | Stateful agentic graph, conditional routing |
| LLM | OpenRouter / Gemini | Primary + backup auto-failover |
| Vector DB | Qdrant (local) | No Docker; workspace_id payload filters |
| Embeddings | `all-MiniLM-L6-v2` | Local CPU, ~90 MB |
| Sparse Retrieval | BM25Okapi | Static (default) + dynamic (workspaces) |
| Fusion | Reciprocal Rank Fusion | Parameter-free, rank-based merge |
| Reranker | `ms-marco-MiniLM-L-6-v2` | Cross-encoder, ~80ms on CPU |
| API | FastAPI + Uvicorn | Full OpenAPI docs at `/docs` |
| Memory | In-process dict | 10 messages/session, thread-safe |
| Cache | LRU (100 entries) | Exact-match, keyed by (question, workspace, mode) |
| Web Search | Tavily (optional) | 1,000 free searches/month |
| PDF parser | pypdf | Text-based PDFs |
| DOCX parser | python-docx | Headings + tables |
| YouTube | youtube-transcript-api | Auto-generated or manual captions |

---

## Project Structure

```
src/
├── v1/                  ← Frozen baseline (linear graph, no memory)
└── v2/                  ← Active (agentic, workspace-based)
    ├── api/
    │   ├── main.py      ← FastAPI server (< 340 lines)
    │   └── schemas.py   ← All Pydantic request/response models
    ├── generation/
    │   ├── chain.py     ← Graph assembler + ask() entry point
    │   ├── state.py     ← RAGState TypedDict + make_initial_state()
    │   ├── prompt.py    ← Three system prompts (rag/chat/web)
    │   ├── citation_validator.py
    │   └── nodes/
    │       ├── router.py    ← Heuristic query classifier
    │       ├── retrieve.py  ← Hybrid retrieval (static+dynamic BM25)
    │       ├── rerank.py    ← Cross-encoder reranking
    │       ├── generate.py  ← LLM generation with history injection
    │       └── validate.py  ← Citation stripping
    ├── retrieval/
    │   ├── models.py        ← RetrievedChunk dataclass
    │   ├── vector_retriever.py  ← Qdrant + workspace filter
    │   ├── bm25_retriever.py    ← Static pickle BM25
    │   ├── dynamic_bm25.py      ← On-the-fly BM25 for workspaces
    │   ├── fusion.py            ← Reciprocal Rank Fusion
    │   └── reranker.py
    ├── ingestion/
    │   ├── dispatcher.py    ← Parse → chunk → embed → upsert
    │   └── parsers/
    │       ├── url_parser.py
    │       ├── pdf_parser.py
    │       ├── docx_parser.py
    │       └── youtube_parser.py
    ├── memory/
    │   ├── models.py    ← ChatMessage, Session
    │   └── store.py     ← MemoryStore (10 msgs/session, thread-safe)
    ├── workspaces/
    │   ├── models.py    ← Workspace, WorkspaceSource, WorkspaceStore
    │   └── manager.py   ← CRUD + JSON persistence
    ├── tools/
    │   ├── web_search.py  ← Tavily wrapper
    │   └── registry.py    ← Tool registry + availability flags
    └── core/
        ├── retry.py     ← Exponential backoff decorator
        └── cache.py     ← LRU query cache (100 entries)

static/
├── index.html       ← Thin shell, loads ES modules
├── css/main.css     ← All styles
└── js/
    ├── api.js        ← Fetch wrappers
    ├── chat.js       ← Message rendering, sources panel
    ├── memory.js     ← Session persistence (localStorage)
    └── workspace.js  ← Sidebar + upload overlay
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <your-repo>
cd rag-portfolio
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure

```bash
copy .env.example .env   # Windows
# Edit .env — at minimum set GOOGLE_API_KEY or OPENROUTER_API_KEY
```

### 3. Index the default LangChain Docs workspace

```bash
python scripts/ingest.py
# Scrapes 20 LangChain pages, embeds locally (~90 MB download, CPU only)
# Builds Qdrant index + BM25 pickle — takes ~2 min on first run
```

### 4. Start the v2 server

```bash
uvicorn src.v2.api.main:app --reload --port 8000
```

Open **http://localhost:8000** for the chat UI, **http://localhost:8000/docs** for the API.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Pipeline readiness + workspace count |
| `GET` | `/metrics` | Cache stats, memory stats, tool availability |
| `POST` | `/ask` | Q&A with session memory + workspace scope |
| `GET` | `/session/{id}` | Get conversation history |
| `DELETE` | `/session/{id}` | Clear a session |
| `GET` | `/workspaces` | List all workspaces |
| `POST` | `/workspaces` | Create a workspace |
| `DELETE` | `/workspaces/{id}` | Delete workspace + purge vectors |
| `POST` | `/workspaces/{id}/ingest/url` | Ingest URL or YouTube link |
| `POST` | `/workspaces/{id}/ingest/file` | Upload PDF / DOCX / TXT |

### Ask with session memory

```bash
# First request — no session_id needed
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LCEL?", "workspace_id": "default"}'

# Response includes session_id — pass it back for multi-turn memory
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Show me an example",
    "session_id": "RETURNED_SESSION_ID",
    "workspace_id": "default",
    "search_mode": "combined"
  }'
```

### Create a workspace and ingest a PDF

```bash
# 1. Create workspace
curl -X POST http://localhost:8000/workspaces \
  -H "Content-Type: application/json" \
  -d '{"name": "Q3 Research"}'

# 2. Upload a PDF
curl -X POST http://localhost:8000/workspaces/WORKSPACE_ID/ingest/file \
  -F "file=@./my_paper.pdf"

# 3. Chat with your document
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the key findings", "workspace_id": "WORKSPACE_ID", "search_mode": "workspace"}'
```

---

## Enabling Web Search

Add a free Tavily API key ([tavily.com](https://tavily.com)) to `.env`:

```
TAVILY_API_KEY=tvly-...
```

The router will automatically detect queries like *"latest langchain release"* or *"search the web for…"* and fetch live results instead of hitting the document index.

---

## Evaluation

```bash
# Run LLM-as-a-judge evaluation on 10 samples
python scripts/evaluate_v2.py --samples 10

# CI quality gate (fails if faithfulness < 0.75 or relevance < 0.70)
pytest tests/ -v
```

---

## Key Design Decisions

**Workspace isolation via payload filters, not separate collections** — One Qdrant collection with `workspace_id` in every vector payload. Filtering is O(log n) and avoids the management overhead of per-user collections.

**Dynamic BM25 at query time** — For user-uploaded documents, BM25 is built in-memory from Qdrant payloads at query time (~30ms for 1,000 chunks). The static pre-built index is only used for the default LangChain Docs workspace.

**Heuristic router, not LLM router** — Zero-latency regex classification saves 200-400ms and one API call per query. Swap `classify_query()` in `nodes/router.py` for an LLM call if precision needs to improve.

**LRU cache keyed on (question, workspace, mode)** — Repeated identical queries (common in demos) return instantly without touching the LLM or retrieval stack.

**Primary → backup LLM failover** — If the primary model (e.g. free OpenRouter tier) hits rate limits, the backup (Gemini) is tried transparently.

**v1 baseline preserved in `src/v1/`** — The original linear pipeline is frozen as a reference. Switch the server back to `src.v1.api.main:app` to A/B compare.


