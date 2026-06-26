"""
state.py — RAGState TypedDict for the LangGraph v2 pipeline.

Separating the state schema into its own file keeps chain.py thin and
makes it trivial to inspect, test, or extend the state independently.

Key additions over v1:
  - messages:     Conversation history injected into the prompt
  - workspace_id: Scopes retrieval to the active workspace
  - search_mode:  "docs" | "workspace" | "combined" | "web"
  - route:        Decision made by the router node
  - web_results:  Raw snippets from web search (when route == "web")
"""

from __future__ import annotations

from typing import TypedDict, Optional

from src.v2.memory.models import ChatMessage
from src.v2.retrieval.models import RetrievedChunk


class RAGState(TypedDict):
    """
    The state dictionary threaded through every node in the LangGraph.

    Every field must have a defined initial value in RAGPipeline.ask()
    before the graph is invoked — missing keys cause runtime KeyError.

    Retrieval flow:
        vector_results + bm25_results
            → (RRF) fused_results
                → (Cross-Encoder) final_chunks
                    → context_block
                        → raw_answer
                            → validated_answer
    """

    # --- Query ---
    query: str
    session_id: str
    workspace_id: str                   # Qdrant payload filter key
    search_mode: str                    # "docs" | "workspace" | "combined" | "web"

    # --- Routing ---
    route: str                          # Set by router node: "rag" | "web" | "chat"

    # --- Conversation history ---
    messages: list[ChatMessage]         # Last N messages from MemoryStore

    # --- Retrieval ---
    vector_results: list[RetrievedChunk]
    bm25_results: list[RetrievedChunk]
    fused_results: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]  # After reranking

    # --- Web search (used when route == "web") ---
    web_results: list[dict]             # Raw snippets from Tavily/Brave

    # --- Generation ---
    context_block: str
    raw_answer: str
    validated_answer: str

    # --- Citations ---
    cited_indices: list[int]
    invalid_citations: list[int]
    sources: list[dict]

    # --- Metadata ---
    retrieval_method: str
    error: Optional[str]


def make_initial_state(
    query: str,
    session_id: str = "",
    workspace_id: str = "default",
    search_mode: str = "combined",
    messages: Optional[list[ChatMessage]] = None,
) -> RAGState:
    """
    Build a fully initialised RAGState with safe defaults.

    Call this in RAGPipeline.ask() instead of constructing the dict
    manually — it prevents KeyError from missing TypedDict fields.

    Args:
        query:        The user's question.
        session_id:   Active session (used to fetch/store history).
        workspace_id: Qdrant partition to search.
        search_mode:  Which sources to search.
        messages:     Pre-fetched conversation history.

    Returns:
        A RAGState dict ready for graph.invoke().
    """
    return RAGState(
        query=query,
        session_id=session_id,
        workspace_id=workspace_id,
        search_mode=search_mode,
        route="",
        messages=messages or [],
        vector_results=[],
        bm25_results=[],
        fused_results=[],
        final_chunks=[],
        web_results=[],
        context_block="",
        raw_answer="",
        validated_answer="",
        cited_indices=[],
        invalid_citations=[],
        sources=[],
        retrieval_method="",
        error=None,
    )
