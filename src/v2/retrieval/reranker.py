"""
reranker.py — Cross-Encoder Neural Reranking

After hybrid retrieval produces a coarse set of ~50 candidates, this
cross-encoder scores each (query, chunk) pair TOGETHER — unlike the
bi-encoder (which encodes query and document independently), giving
dramatically higher precision.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Size: ~90 MB
- Runs entirely on CPU (~50-100ms for 50 candidates)
- No API key required
"""

import logging
from dataclasses import replace

from sentence_transformers.cross_encoder import CrossEncoder

from src.v2.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Neural cross-encoder reranker for post-retrieval precision boost.

    The cross-encoder reads the full (query + chunk) pair jointly,
    allowing it to model fine-grained relevance that bi-encoders miss
    (e.g., negation, exact entity matching, multi-sentence reasoning).
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        logger.info(f"Loading cross-encoder reranker: {model_name}")
        # CrossEncoder downloads the model on first use (~90 MB)
        self.model = CrossEncoder(model_name, max_length=512)
        logger.info("✓ CrossEncoderReranker ready")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Rerank candidate chunks using the cross-encoder.

        Args:
            query: The user's original query.
            candidates: Candidate chunks from RRF fusion.
            top_k: Number of top results to return after reranking.

        Returns:
            Top-k RetrievedChunk objects sorted by cross-encoder score,
            with retriever field set to "reranked".
        """
        if not candidates:
            return []

        # Build (query, text) pairs for the cross-encoder
        pairs = [(query, chunk.text) for chunk in candidates]

        # Score all pairs (runs on CPU, no API call)
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores and sort
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk, score in scored[:top_k]:
            # Create a new RetrievedChunk with the reranker score
            reranked = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source_url=chunk.source_url,
                source_title=chunk.source_title,
                score=float(score),
                retriever="reranked",
            )
            results.append(reranked)

        logger.debug(
            f"Reranking: {len(candidates)} candidates → top {len(results)} "
            f"(best score: {results[0].score:.3f})"
        )
        return results
