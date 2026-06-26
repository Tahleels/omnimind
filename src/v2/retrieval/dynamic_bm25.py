"""
dynamic_bm25.py — On-the-Fly BM25 Retrieval for Workspace Documents.

Problem: The v1 BM25Retriever loads a pre-built pickle index that was
created at ingestion time.  This works for the static LangChain Docs
workspace, but breaks for dynamically uploaded user documents.

Solution: When a user asks a question against a custom workspace, fetch
all text chunks for that workspace from Qdrant and build a BM25Okapi
index in memory on the fly.

Performance:
  - Fetching 1,000 chunks from Qdrant (local):  ~10 ms
  - Building BM25Okapi over 1,000 chunks:        ~20 ms
  - Total overhead vs static BM25:               ~30 ms (acceptable)

For the default LangChain Docs workspace, the static BM25Retriever
(src/v2/retrieval/bm25_retriever.py) is used instead — it's faster
because the index is already built.

Usage:
    retriever = DynamicBM25Retriever(vector_retriever)
    results = retriever.retrieve("my query", workspace_id="workspace-uuid")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

from src.v2.retrieval.models import RetrievedChunk

if TYPE_CHECKING:
    from src.v2.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


class DynamicBM25Retriever:
    """
    Workspace-scoped BM25 retriever that builds its index at query time.

    The retriever depends on VectorRetriever.fetch_workspace_chunks()
    to pull chunk texts from Qdrant so we don't duplicate storage.

    One DynamicBM25Retriever instance is shared for all workspaces —
    each call to retrieve() builds a fresh index for the requested
    workspace.  This is intentional: workspace chunk counts are small
    enough that caching the index provides negligible benefit while
    adding stale-cache risk after new ingestion.
    """

    def __init__(self, vector_retriever: "VectorRetriever") -> None:
        self._vector = vector_retriever
        logger.info("DynamicBM25Retriever ready")

    def retrieve(
        self,
        query: str,
        workspace_id: str,
        top_k: int = 25,
    ) -> list[RetrievedChunk]:
        """
        Build a BM25 index for the workspace and return top-k results.

        Args:
            query:        User query string.
            workspace_id: Target workspace.
            top_k:        Max results to return.

        Returns:
            List of RetrievedChunk sorted by descending BM25 score,
            with zero-scoring chunks filtered out.
        """
        # Step 1: fetch all chunks for this workspace from Qdrant
        raw_chunks = self._vector.fetch_workspace_chunks(workspace_id)
        if not raw_chunks:
            logger.warning(
                f"DynamicBM25: no chunks found for workspace '{workspace_id}'"
            )
            return []

        # Step 2: build BM25 index in memory
        tokenized = [
            chunk.get("text", "").lower().split()
            for chunk in raw_chunks
        ]
        bm25 = BM25Okapi(tokenized)

        # Step 3: score the query
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        # Step 4: collect top-k non-zero results
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break  # Sorted descending — all remaining are also 0
            chunk = raw_chunks[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.get("chunk_id", f"dyn_{idx}"),
                    text=chunk.get("text", ""),
                    source_url=chunk.get("source_url", ""),
                    source_title=chunk.get("source_title", ""),
                    score=float(scores[idx]),
                    retriever="bm25",
                    workspace_id=chunk.get("workspace_id", workspace_id),
                    source_id=chunk.get("source_id", ""),
                )
            )

        logger.debug(
            f"DynamicBM25: {len(results)} results "
            f"(workspace='{workspace_id}', chunks={len(raw_chunks)})"
        )
        return results
