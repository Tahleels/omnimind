"""
schemas.py — Pydantic Request/Response Models for the FastAPI Server

All API input/output is typed and validated by Pydantic. This ensures:
- Clear API contracts
- Automatic OpenAPI docs generation
- Runtime input validation
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """POST /ask — Question from the user."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to ask about LangChain documentation.",
        examples=["How do I use LCEL to chain prompts together?"],
    )


class SourceReference(BaseModel):
    """A single cited source from the LangChain docs."""
    citation_number: int = Field(..., description="The [n] citation number in the answer.")
    title: str = Field(..., description="Title of the source page.")
    url: str = Field(..., description="URL of the source page.")
    excerpt: str = Field(..., description="Short excerpt from the cited chunk.")


class RetrievalStats(BaseModel):
    """Debug metadata about the retrieval pipeline."""
    vector_results: int
    bm25_results: int
    fused_results: int
    final_chunks: int
    invalid_citations_stripped: int


class AskResponse(BaseModel):
    """POST /ask — Response with answer, citations, and sources."""
    answer: str = Field(..., description="The generated answer with [n] citation markers.")
    sources: list[SourceReference] = Field(
        default_factory=list,
        description="List of cited sources referenced in the answer.",
    )
    retrieval_stats: RetrievalStats = Field(
        ...,
        description="Pipeline performance metadata (for debugging/demo).",
    )


class HealthResponse(BaseModel):
    """GET /health — Simple health check response."""
    status: str = "ok"
    message: str = "RAG Pipeline is up and running."
    indexes_loaded: bool = False
