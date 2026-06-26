"""
fusion.py — Reciprocal Rank Fusion (RRF)

Merges ranked results from multiple retrievers (vector + BM25) into a
single unified ranking without needing to normalize scores across
different mathematical spaces.

Why RRF over weighted score averaging?
- BM25 scores are term-frequency-based integers (e.g., 12.3)
- Vector scores are normalized cosine similarities (e.g., 0.87)
- These spaces are INCOMPATIBLE — you cannot directly add them.
- RRF only uses the *rank position* of each result, making it
  mathematically clean, parameter-free, and empirically strong.

Formula: score(d) = Σ 1 / (k + rank(d, retriever_i))
where k=60 is a smoothing constant (standard default).
"""

import logging
from collections import defaultdict

from src.retrieval.vector_retriever import RetrievedChunk

logger = logging.getLogger(__name__)

RRF_K = 60  # Standard smoothing constant; higher = less penalty for lower ranks


def reciprocal_rank_fusion(
    *ranked_lists: list[RetrievedChunk],
    top_k: int = 50,
) -> list[RetrievedChunk]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        *ranked_lists: Variable number of ranked result lists (e.g., vector
                       results and BM25 results). Each list should be sorted
                       by descending relevance.
        top_k: Number of results to return from the fused ranking.

    Returns:
        Merged and re-ranked list of RetrievedChunk objects, sorted by
        descending RRF score.
    """
    # Map chunk_id → RetrievedChunk (keep the first occurrence found)
    chunk_map: dict[str, RetrievedChunk] = {}
    rrf_scores: dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            # Accumulate RRF score: higher rank = higher contribution
            rrf_scores[chunk.chunk_id] += 1.0 / (RRF_K + rank)

            # Store chunk data (prefer whichever retriever found it)
            if chunk.chunk_id not in chunk_map:
                chunk_map[chunk.chunk_id] = chunk

    # Sort by RRF score descending and take top_k
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    top_ids = sorted_ids[:top_k]

    fused_results = []
    for cid in top_ids:
        chunk = chunk_map[cid]
        # Replace the original retriever score with the RRF score
        fused = RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source_url=chunk.source_url,
            source_title=chunk.source_title,
            score=rrf_scores[cid],
            retriever="rrf_fused",
        )
        fused_results.append(fused)

    logger.debug(
        f"RRF fusion: {sum(len(r) for r in ranked_lists)} candidates → "
        f"{len(fused_results)} fused results (top {top_k})"
    )
    return fused_results
