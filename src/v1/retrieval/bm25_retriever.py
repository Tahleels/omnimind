"""
bm25_retriever.py — Sparse BM25 Keyword Retrieval

Loads the pre-built BM25Okapi index from disk and retrieves the top-k
best keyword-matched chunks for a given query.

BM25 excels at exact-match queries — things like API names, error codes,
class names, and acronyms that dense embeddings often miss.
"""

import logging
import pickle
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.retrieval.vector_retriever import RetrievedChunk

logger = logging.getLogger(__name__)


class BM25Retriever:
    """
    Sparse retriever using BM25Okapi keyword matching.

    Loads the pickled index built during ingestion and performs fast
    in-memory term-frequency/inverse-document-frequency scoring.
    """

    def __init__(self, bm25_path: str = "./data/bm25_index.pkl"):
        logger.info(f"Loading BM25 index from: {bm25_path}")

        with open(bm25_path, "rb") as f:
            index_data = pickle.load(f)

        self.bm25: BM25Okapi = index_data["bm25"]
        self.chunks: list[dict] = index_data["chunks"]

        logger.info(f"✓ BM25Retriever ready ({len(self.chunks)} chunks indexed)")

    def retrieve(self, query: str, top_k: int = 25) -> list[RetrievedChunk]:
        """
        Retrieve the top-k keyword-matching chunks for a query.

        Args:
            query: The user's search query.
            top_k: Number of results to return.

        Returns:
            List of RetrievedChunk objects sorted by descending BM25 score.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices by score (descending)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        chunks = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue  # Skip zero-score results
            chunk_data = self.chunks[idx]
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_data["chunk_id"],
                    text=chunk_data["text"],
                    source_url=chunk_data["source_url"],
                    source_title=chunk_data["source_title"],
                    score=float(scores[idx]),
                    retriever="bm25",
                )
            )

        logger.debug(f"BM25 retrieval: {len(chunks)} results for query: '{query[:60]}'")
        return chunks
