"""
schemas.py — Pydantic Request/Response Models for the v2 FastAPI Server.

Additions over v1:
  - AskRequest now accepts session_id, workspace_id, search_mode
  - AskResponse returns session_id and route (for frontend state)
  - New SessionResponse for GET /session/{id}
  - New WorkspaceResponse family for workspace CRUD endpoints
  - New IngestResponse for file/URL ingestion endpoints
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core Q&A schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """POST /ask — Question with optional session and workspace context."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question.",
        examples=["How do I use LCEL to chain prompts together?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation memory. Auto-generated if omitted.",
    )
    workspace_id: str = Field(
        default="default",
        description="Workspace to search. Use 'default' for LangChain Docs.",
    )
    search_mode: str = Field(
        default="combined",
        description="Search scope: 'docs' | 'workspace' | 'combined'",
    )


class SourceReference(BaseModel):
    """A single cited source returned with an answer."""
    citation_number: int
    title: str
    url: str
    excerpt: str


class RetrievalStats(BaseModel):
    """Pipeline performance metadata (for the UI stats panel)."""
    vector_results: int
    bm25_results: int
    fused_results: int
    final_chunks: int
    invalid_citations_stripped: int


class AskResponse(BaseModel):
    """POST /ask — Full response with answer, citations, and pipeline metadata."""
    answer: str
    sources: list[SourceReference] = Field(default_factory=list)
    retrieval_stats: RetrievalStats
    session_id: str = Field(description="Session ID — store this in the frontend.")
    route: str = Field(description="Which pipeline path was used: rag | chat | web")


# ---------------------------------------------------------------------------
# Health & system schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """GET /health — Pipeline readiness."""
    status: str = "ok"
    message: str = "RAG Pipeline v2 is ready."
    indexes_loaded: bool = False
    workspace_count: int = 0


# ---------------------------------------------------------------------------
# Session / memory schemas
# ---------------------------------------------------------------------------

class ChatMessageOut(BaseModel):
    """A single message returned in a session history response."""
    role: str
    content: str
    timestamp: str
    cited_sources: list[str] = Field(default_factory=list)


class SessionResponse(BaseModel):
    """GET /session/{session_id} — Returns conversation history."""
    session_id: str
    workspace_id: str
    messages: list[ChatMessageOut]


# ---------------------------------------------------------------------------
# Workspace schemas
# ---------------------------------------------------------------------------

class WorkspaceCreateRequest(BaseModel):
    """POST /workspaces — Create a new workspace."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="")


class WorkspaceSourceOut(BaseModel):
    """A single source within a workspace (for list responses)."""
    source_id: str
    source_type: str
    name: str
    origin: str
    chunk_count: int
    ingested_at: str


class WorkspaceOut(BaseModel):
    """Full workspace detail response."""
    workspace_id: str
    name: str
    description: str
    is_default: bool
    total_chunks: int
    source_count: int
    sources: list[WorkspaceSourceOut]
    created_at: str


class WorkspaceListResponse(BaseModel):
    """GET /workspaces — List of all workspaces (summary only)."""
    workspaces: list[WorkspaceOut]


# ---------------------------------------------------------------------------
# Ingestion schemas
# ---------------------------------------------------------------------------

class IngestUrlRequest(BaseModel):
    """POST /workspace/{id}/ingest/url — Ingest a URL or YouTube link."""
    url: str = Field(..., description="Web URL or YouTube link to ingest.")
    name: Optional[str] = Field(
        default=None,
        description="Optional human-readable display name for this source.",
    )


class IngestResponse(BaseModel):
    """Response after a successful ingestion (file or URL)."""
    workspace_id: str
    source_id: str
    name: str
    source_type: str
    chunks_indexed: int
    message: str = "Ingestion complete."
