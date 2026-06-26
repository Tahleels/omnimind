"""
models.py — Shared data models for the v2 retrieval layer.

Separating the RetrievedChunk dataclass here avoids circular imports
between vector_retriever.py, bm25_retriever.py, dynamic_bm25.py, and
fusion.py — all of which need this type.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """
    A single retrieved text chunk with relevance metadata.

    Fields:
        chunk_id:       Unique identifier matching the Qdrant payload.
        text:           The raw chunk text passed to the LLM as context.
        source_url:     Origin URL of the document.
        source_title:   Human-readable document title.
        score:          Relevance score (meaning depends on retriever field).
        retriever:      Which retriever produced this result:
                          "vector"    — cosine similarity from Qdrant
                          "bm25"      — BM25Okapi score
                          "rrf_fused" — combined RRF score
                          "reranked"  — cross-encoder score
        workspace_id:   Workspace this chunk belongs to.
        source_id:      Source document ID within the workspace.
    """
    chunk_id: str
    text: str
    source_url: str
    source_title: str
    score: float
    retriever: str = "vector"
    workspace_id: str = "default"
    source_id: str = ""
