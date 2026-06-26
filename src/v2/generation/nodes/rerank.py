"""
rerank.py — LangGraph Cross-Encoder Rerank Node (v2)

Wraps the CrossEncoderReranker as a LangGraph node factory.
Only runs when route == "rag" AND fused_results is non-empty.

No logic changes from v1 — purely structural refactor into its
own file to keep chain.py thin and this node unit-testable.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from src.v2.generation.state import RAGState
from src.v2.generation.prompt import build_context_block

if TYPE_CHECKING:
    from src.v2.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


def build_rerank_node(reranker: "CrossEncoderReranker"):
    """
    Factory returning the rerank node with the reranker injected.

    Args:
        reranker: Shared CrossEncoderReranker instance.

    Returns:
        A node function: (state: RAGState) -> dict
    """

    def node_rerank(state: RAGState) -> dict:
        """
        LangGraph node: cross-encoder reranking of fused candidates.

        Skips silently when route != "rag" or there are no fused results.
        """
        if state.get("route") != "rag":
            return {}

        fused = state.get("fused_results", [])
        if not fused:
            logger.warning("Rerank node: no fused results to rerank")
            return {"final_chunks": [], "context_block": ""}

        top_k = int(os.getenv("RERANKER_TOP_K", "5"))
        final_chunks = reranker.rerank(
            query=state["query"],
            candidates=fused,
            top_k=top_k,
        )
        context_block = build_context_block(final_chunks)

        logger.info(
            f"Rerank: {len(fused)} candidates → {len(final_chunks)} final chunks"
        )
        return {
            "final_chunks": final_chunks,
            "context_block": context_block,
        }

    return node_rerank
