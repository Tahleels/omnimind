"""
main.py — FastAPI Server v2

Serves the v2 RAG pipeline with:
  - Conversation memory (session_id in every /ask request)
  - Workspace management (CRUD for document partitions)
  - Dynamic ingestion (file upload + URL scraping endpoints)
  - All v1 endpoints maintained at /ask, /health for backwards compat

Endpoint map:
  GET  /              — Serves the frontend UI
  GET  /health        — Pipeline readiness + workspace count
  POST /ask           — Main Q&A endpoint (with session + workspace)
  GET  /session/{id}  — Retrieve conversation history for a session
  DELETE /session/{id}— Clear a session's history
  GET  /workspaces    — List all workspaces
  POST /workspaces    — Create a new workspace
  DELETE /workspaces/{id}  — Delete a workspace (purges Qdrant vectors)
  POST /workspaces/{id}/ingest/url  — Ingest a web URL into a workspace
  POST /workspaces/{id}/ingest/file — Upload a PDF/DOCX/TXT file

Intentionally kept under 300 lines — route handlers delegate to
service layers (chain, memory, workspace manager, ingestion).
"""

from __future__ import annotations

import logging
import os
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

# Fix MIME type detection on Windows for JavaScript module scripts
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


from src.v2.api.schemas import (
    AskRequest, AskResponse, HealthResponse, RetrievalStats, SourceReference,
    SessionResponse, ChatMessageOut,
    WorkspaceCreateRequest, WorkspaceOut, WorkspaceListResponse,
    WorkspaceSourceOut, IngestUrlRequest, IngestResponse,
)
from src.v2.generation.chain import RAGPipelineV2
from src.v2.memory.store import MemoryStore
from src.v2.workspaces.manager import WorkspaceManager
from src.v2.workspaces.models import WorkspaceSource, SourceType
from src.v2.ingestion.dispatcher import ingest_source

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
pipeline: RAGPipelineV2 | None = None
pipeline_ready: bool = False
memory_store: MemoryStore = MemoryStore()
workspace_mgr: WorkspaceManager = WorkspaceManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, pipeline_ready

    qdrant_path = os.getenv("QDRANT_PATH", "./data/qdrant_storage")
    bm25_path = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
    collection_name = os.getenv("COLLECTION_NAME", "langchain_docs")
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    reranker_model = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    if not Path(bm25_path).exists():
        logger.warning("⚠ BM25 index missing — run: python scripts/ingest.py")
        pipeline_ready = False
    else:
        try:
            pipeline = RAGPipelineV2(
                qdrant_path=qdrant_path,
                bm25_path=bm25_path,
                collection_name=collection_name,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
                memory_store=memory_store,
            )
            pipeline_ready = True
            logger.info("✓ RAGPipelineV2 ready")
        except Exception as exc:
            logger.error(f"Pipeline init failed: {exc}")
            pipeline_ready = False

    yield
    logger.info("=== Shutting down RAGPipelineV2 ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OmniMind RAG v2",
    description=(
        "Agentic RAG with conversation memory, workspace isolation, "
        "dynamic ingestion, and hybrid retrieval."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent.parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "Frontend not found. See /docs for the API."}
    return FileResponse(str(index_path))


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    ws_summary = workspace_mgr.summary()
    return HealthResponse(
        status="ok",
        message=(
            "RAG Pipeline v2 is ready."
            if pipeline_ready
            else "⚠ Pipeline not ready — run: python scripts/ingest.py"
        ),
        indexes_loaded=pipeline_ready,
        workspace_count=ws_summary["workspace_count"],
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """Runtime metrics: cache, memory, workspaces, tool availability."""
    data = {
        "pipeline_ready": pipeline_ready,
        "workspaces": workspace_mgr.summary(),
        "memory": memory_store.stats(),
    }
    if pipeline_ready and pipeline is not None:
        data["cache"] = pipeline.cache.stats()
        data["tools"] = pipeline.tools.summary()
    return data


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest):
    """Ask a question. Pass session_id for multi-turn memory."""
    if not pipeline_ready or pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")

    try:
        result = pipeline.ask(
            question=request.question,
            session_id=request.session_id or "",
            workspace_id=request.workspace_id,
            search_mode=request.search_mode,
        )
    except Exception as exc:
        logger.error(f"/ask error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    sources = [SourceReference(**s) for s in result["sources"]]
    stats = RetrievalStats(**result["retrieval_stats"])
    return AskResponse(
        answer=result["answer"],
        sources=sources,
        retrieval_stats=stats,
        session_id=result["session_id"],
        route=result["route"],
    )


# ---------------------------------------------------------------------------
# Session / Memory
# ---------------------------------------------------------------------------
@app.get("/session/{session_id}", response_model=SessionResponse, tags=["Memory"])
async def get_session(session_id: str):
    """Retrieve conversation history for a session (for UI reload)."""
    session = memory_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionResponse(
        session_id=session_id,
        workspace_id=session.workspace_id,
        messages=[
            ChatMessageOut(
                role=m.role if isinstance(m.role, str) else m.role.value,
                content=m.content,
                timestamp=m.timestamp.isoformat(),
                cited_sources=m.cited_sources,
            )
            for m in session.messages
        ],
    )


@app.delete("/session/{session_id}", tags=["Memory"])
async def clear_session(session_id: str):
    """Clear a session's conversation history."""
    cleared = memory_store.clear_session(session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"message": f"Session {session_id} cleared."}


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------
def _workspace_to_out(ws) -> WorkspaceOut:
    return WorkspaceOut(
        workspace_id=ws.workspace_id,
        name=ws.name,
        description=ws.description,
        is_default=ws.is_default,
        total_chunks=ws.total_chunks,
        source_count=ws.source_count,
        sources=[
            WorkspaceSourceOut(
                source_id=s.source_id,
                source_type=s.source_type,
                name=s.name,
                origin=s.origin,
                chunk_count=s.chunk_count,
                ingested_at=s.ingested_at.isoformat(),
            )
            for s in ws.sources
        ],
        created_at=ws.created_at.isoformat(),
    )


@app.get("/workspaces", response_model=WorkspaceListResponse, tags=["Workspaces"])
async def list_workspaces():
    workspaces = workspace_mgr.list_workspaces()
    return WorkspaceListResponse(workspaces=[_workspace_to_out(w) for w in workspaces])


@app.post("/workspaces", response_model=WorkspaceOut, tags=["Workspaces"])
async def create_workspace(request: WorkspaceCreateRequest):
    ws = workspace_mgr.create_workspace(
        name=request.name, description=request.description
    )
    return _workspace_to_out(ws)


@app.delete("/workspaces/{workspace_id}", tags=["Workspaces"])
async def delete_workspace(workspace_id: str):
    if not pipeline_ready or pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    deleted = workspace_mgr.delete_workspace(workspace_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Workspace not found or cannot be deleted.",
        )
    # Purge Qdrant vectors for this workspace
    pipeline.vector_retriever.delete_by_workspace(workspace_id)
    return {"message": f"Workspace {workspace_id} deleted."}


# ---------------------------------------------------------------------------
# Ingestion endpoints (stubs — parsers wired in Phase 3)
# ---------------------------------------------------------------------------
@app.post(
    "/workspaces/{workspace_id}/ingest/url",
    response_model=IngestResponse,
    tags=["Ingestion"],
)
async def ingest_url(workspace_id: str, request: IngestUrlRequest):
    """Ingest a web URL or YouTube link into a workspace."""
    if not pipeline_ready or pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    ws = workspace_mgr.get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")
    try:
        result = ingest_source(
            source=request.url,
            filename=request.url,
            workspace_id=workspace_id,
            vector_retriever=pipeline.vector_retriever,
            embedding_model=pipeline.vector_retriever.model,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Register source with workspace manager
    from src.v2.workspaces.models import WorkspaceSource, SourceType
    source = WorkspaceSource(
        source_id=result.source_id,
        source_type=result.source_type,
        name=request.name or result.name,
        origin=result.origin,
        chunk_count=result.chunk_count,
    )
    workspace_mgr.add_source(workspace_id, source)

    return IngestResponse(
        workspace_id=workspace_id,
        source_id=result.source_id,
        name=source.name,
        source_type=result.source_type,
        chunks_indexed=result.chunk_count,
    )


@app.post(
    "/workspaces/{workspace_id}/ingest/file",
    response_model=IngestResponse,
    tags=["Ingestion"],
)
async def ingest_file(workspace_id: str, file: UploadFile = File(...)):
    """Upload a PDF, DOCX, or TXT file into a workspace."""
    if not pipeline_ready or pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready.")
    ws = workspace_mgr.get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found.")

    content = await file.read()
    filename = file.filename or "upload"

    try:
        result = ingest_source(
            source=content,
            filename=filename,
            workspace_id=workspace_id,
            vector_retriever=pipeline.vector_retriever,
            embedding_model=pipeline.vector_retriever.model,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    from src.v2.workspaces.models import WorkspaceSource
    source = WorkspaceSource(
        source_id=result.source_id,
        source_type=result.source_type,
        name=result.name,
        origin=filename,
        chunk_count=result.chunk_count,
        page_count=result.page_count,
        word_count=result.word_count,
    )
    workspace_mgr.add_source(workspace_id, source)

    return IngestResponse(
        workspace_id=workspace_id,
        source_id=result.source_id,
        name=result.name,
        source_type=result.source_type,
        chunks_indexed=result.chunk_count,
    )
