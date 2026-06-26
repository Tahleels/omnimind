"""
vector_retriever.py — Dense Vector Retrieval via Qdrant (v2)

Changes over v1:
  - Accepts an optional `workspace_id` filter so the same Qdrant
    collection serves multiple isolated document workspaces.
  - `workspace_id=None` means "search all vectors" (used internally
    for the default LangChain Docs workspace).
  - Uses the shared RetrievedChunk from retrieval.models to avoid
    circular imports.
  - All config values come from env vars with sensible defaults.

Architecture note:
  VectorRetriever is initialised ONCE at startup (heavy — loads the
  ~90 MB embedding model into RAM) and reused for every query.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.v2.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Dense retriever backed by Qdrant local file storage.

    Encodes queries on-the-fly with the same embedding model used
    during ingestion and retrieves nearest neighbours via cosine
    similarity, optionally filtered by workspace_id.
    """

    def __init__(
        self,
        qdrant_path: str = "./data/qdrant_storage",
        collection_name: str = "langchain_docs",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        logger.info(
            f"Initialising VectorRetriever (collection={collection_name})"
        )
        self.collection_name = collection_name

        # Local Qdrant — no Docker required
        self.client = QdrantClient(path=qdrant_path)

        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)
        logger.info("✓ VectorRetriever ready")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 25,
        workspace_id: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k semantically similar chunks.

        Args:
            query:        User query string.
            top_k:        Max number of results.
            workspace_id: If given, restrict search to this workspace.
                          Pass None to search across all vectors.

        Returns:
            List of RetrievedChunk sorted by descending cosine score.
        """
        query_vector = self._encode(query)
        qdrant_filter = self._build_filter(workspace_id)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=qdrant_filter,
        )

        chunks = [
            RetrievedChunk(
                chunk_id=r.payload.get("chunk_id", str(r.id)),
                text=r.payload.get("text", ""),
                source_url=r.payload.get("source_url", ""),
                source_title=r.payload.get("source_title", ""),
                score=float(r.score),
                retriever="vector",
                workspace_id=r.payload.get("workspace_id", "default"),
                source_id=r.payload.get("source_id", ""),
            )
            for r in response.points
        ]

        logger.debug(
            f"Vector retrieval: {len(chunks)} results "
            f"(workspace={workspace_id or 'all'}, query='{query[:50]}')"
        )
        return chunks

    def upsert_chunks(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
    ) -> None:
        """
        Upsert pre-embedded chunk vectors into Qdrant.

        Used by the dynamic ingestion pipeline after embedding new
        documents.  Each chunk dict must include: chunk_id, text,
        source_url, source_title, workspace_id, source_id.

        Args:
            chunks:  List of chunk metadata dicts.
            vectors: Parallel list of embedding vectors.
        """
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )

        # Qdrant point IDs must be unsigned ints or UUIDs.  We use the
        # chunk list index offset by the current collection size.
        existing = self.client.count(self.collection_name).count
        points = [
            PointStruct(
                id=existing + i,
                vector=vec,
                payload=chunk,
            )
            for i, (chunk, vec) in enumerate(zip(chunks, vectors))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Upserted {len(points)} vectors into '{self.collection_name}'")

    def delete_by_workspace(self, workspace_id: str) -> int:
        """
        Delete all vectors belonging to a workspace.

        Returns:
            Number of deleted points.
        """
        from qdrant_client.models import FilterSelector

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="workspace_id",
                            match=MatchValue(value=workspace_id),
                        )
                    ]
                )
            ),
        )
        count = result.result.deleted if hasattr(result, "result") else 0
        logger.info(
            f"Deleted vectors for workspace '{workspace_id}' ({count} points)"
        )
        return count

    def fetch_workspace_chunks(self, workspace_id: str) -> list[dict]:
        """
        Fetch all chunk payloads for a workspace (used by DynamicBM25).

        Returns lightweight payload dicts — no vectors returned.
        """
        results, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="workspace_id",
                        match=MatchValue(value=workspace_id),
                    )
                ]
            ),
            limit=10_000,   # Practical cap; increase for huge workspaces
            with_payload=True,
            with_vectors=False,
        )
        return [r.payload for r in results]

    def embed_query(self, query: str) -> list[float]:
        """Encode a query string and return the vector (public helper)."""
        return self._encode(query)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> list[float]:
        """Encode text with normalised embeddings for cosine similarity."""
        return (
            self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            .tolist()
        )

    def _build_filter(
        self, workspace_id: Optional[str]
    ) -> Optional[Filter]:
        """Build a Qdrant Filter for workspace scoping, or None for all."""
        if workspace_id is None:
            return None
        return Filter(
            must=[
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=workspace_id),
                )
            ]
        )
