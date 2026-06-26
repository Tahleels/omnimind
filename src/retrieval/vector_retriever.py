"""
vector_retriever.py — Dense Vector Retrieval via Qdrant

Connects to the local Qdrant collection and retrieves the top-k most
semantically similar chunks for a given query embedding.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its relevance score."""
    chunk_id: str
    text: str
    source_url: str
    source_title: str
    score: float
    retriever: str = "vector"  # "vector" | "bm25" | "reranked"


class VectorRetriever:
    """
    Dense retriever backed by Qdrant local file storage.

    Encodes queries on-the-fly with the same embedding model used during
    ingestion and retrieves nearest neighbors via cosine similarity.
    """

    def __init__(
        self,
        qdrant_path: str = "./data/qdrant_storage",
        collection_name: str = "langchain_docs",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        logger.info(f"Initializing VectorRetriever (collection: {collection_name})")
        self.collection_name = collection_name

        # Load local Qdrant (no Docker needed)
        self.client = QdrantClient(path=qdrant_path)

        # Load the same embedding model used for indexing
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.model = SentenceTransformer(embedding_model_name)
        logger.info("✓ VectorRetriever ready")

    def retrieve(self, query: str, top_k: int = 25) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most similar chunks for a query.

        Args:
            query: The user's search query.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk objects sorted by descending score.
        """
        # Encode query (normalized for cosine similarity)
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).tolist()

        # Search Qdrant using the current API (query_points in qdrant-client >= 1.9)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        chunks = [
            RetrievedChunk(
                chunk_id=r.payload.get("chunk_id", str(r.id)),
                text=r.payload.get("text", ""),
                source_url=r.payload.get("source_url", ""),
                source_title=r.payload.get("source_title", ""),
                score=float(r.score),
                retriever="vector",
            )
            for r in response.points
        ]

        logger.debug(f"Vector retrieval: {len(chunks)} results for query: '{query[:60]}...'")
        return chunks
