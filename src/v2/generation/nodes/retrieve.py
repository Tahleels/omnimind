"""
retrieve.py — LangGraph Hybrid Retrieval Node (v2)

Handles two retrieval modes based on state["workspace_id"] and
state["search_mode"]:

    search_mode == "docs"      → Search default workspace only
    search_mode == "workspace" → Search user workspace only
    search_mode == "combined"  → Search both, merge via RRF

For the default workspace ("default"), the static BM25 pickle is used
(fast, pre-built).  For custom workspaces, DynamicBM25Retriever builds
the BM25 index on the fly from Qdrant payloads.

This node is ONLY reached when state["route"] == "rag".
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from src.v2.generation.state import RAGState
from src.v2.retrieval.models import RetrievedChunk

if TYPE_CHECKING:
    from src.v2.retrieval.vector_retriever import VectorRetriever
    from src.v2.retrieval.dynamic_bm25 import DynamicBM25Retriever
    from src.v2.retrieval.bm25_retriever import BM25Retriever

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE = "default"
DEFAULT_TOP_K = 25


def _get_top_k() -> int:
    return int(os.getenv("VECTOR_TOP_K", str(DEFAULT_TOP_K)))


def _pick_bm25_retriever(
    workspace_id: str,
    static_bm25: "BM25Retriever",
    dynamic_bm25: "DynamicBM25Retriever",
    query: str,
    top_k: int,
) -> list[RetrievedChunk]:
    """
    Route to the right BM25 retriever based on workspace.

    - Default workspace: static pre-built index (fast).
    - Custom workspace: dynamic on-the-fly index.
    """
    if workspace_id == DEFAULT_WORKSPACE:
        return static_bm25.retrieve(query, top_k=top_k)
    return dynamic_bm25.retrieve(query, workspace_id=workspace_id, top_k=top_k)


def build_retrieve_node(
    vector_retriever: "VectorRetriever",
    static_bm25: "BM25Retriever",
    dynamic_bm25: "DynamicBM25Retriever",
):
    """
    Factory that returns a LangGraph-compatible node function with
    retrievers injected via closure.

    Args:
        vector_retriever: Shared VectorRetriever instance.
        static_bm25:      Pre-built BM25Retriever for the default workspace.
        dynamic_bm25:     DynamicBM25Retriever for custom workspaces.

    Returns:
        A node function: (state: RAGState) -> dict
    """
    from src.v2.retrieval.fusion import reciprocal_rank_fusion

    def node_retrieve(state: RAGState) -> dict:
        """
        LangGraph node: run hybrid retrieval and fuse results.

        Skips retrieval when route is "chat" or "web" — those paths
        don't need document context.
        """
        if state["route"] != "rag":
            logger.debug(
                f"Skipping retrieval — route is '{state['route']}'"
            )
            return {}

        query = state["query"]
        workspace_id = state["workspace_id"]
        search_mode = state.get("search_mode", "combined")
        top_k = _get_top_k()

        vector_results: list[RetrievedChunk] = []
        bm25_results: list[RetrievedChunk] = []

        # --- Vector search ---
        vec_workspace = (
            workspace_id
            if search_mode in ("workspace", "combined")
            else None   # None = search all (for "docs" mode we still filter)
        )
        if search_mode == "docs":
            vec_workspace = DEFAULT_WORKSPACE
        elif search_mode == "workspace":
            vec_workspace = workspace_id
        # "combined" → search both in one call (no filter) then tag later
        # For simplicity we run two filtered calls
        if search_mode == "combined" and workspace_id != DEFAULT_WORKSPACE:
            vec_default = vector_retriever.retrieve(
                query, top_k=top_k, workspace_id=DEFAULT_WORKSPACE
            )
            vec_custom = vector_retriever.retrieve(
                query, top_k=top_k, workspace_id=workspace_id
            )
            vector_results = vec_default + vec_custom
        else:
            vector_results = vector_retriever.retrieve(
                query, top_k=top_k, workspace_id=vec_workspace
            )

        # --- BM25 search ---
        if search_mode in ("docs", "combined"):
            bm25_default = _pick_bm25_retriever(
                DEFAULT_WORKSPACE, static_bm25, dynamic_bm25, query, top_k
            )
            bm25_results.extend(bm25_default)

        if search_mode in ("workspace", "combined") and workspace_id != DEFAULT_WORKSPACE:
            bm25_custom = _pick_bm25_retriever(
                workspace_id, static_bm25, dynamic_bm25, query, top_k
            )
            bm25_results.extend(bm25_custom)

        # --- RRF Fusion ---
        fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=50)

        logger.info(
            f"Retrieve: vector={len(vector_results)}, bm25={len(bm25_results)}, "
            f"fused={len(fused)} (mode={search_mode}, workspace={workspace_id})"
        )

        return {
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "fused_results": fused,
            "retrieval_method": f"hybrid_rrf_{search_mode}",
        }

    return node_retrieve
