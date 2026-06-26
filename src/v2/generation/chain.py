"""
chain.py — LangGraph v2 RAG Pipeline Assembler

Responsibilities of THIS file:
  1. Initialise all heavy components (models, retrievers) ONCE at startup.
  2. Wire the LangGraph state machine from the node factories in nodes/.
  3. Expose a clean ask() method that:
       a. Fetches conversation history from MemoryStore
       b. Runs the graph
       c. Persists the new turns to MemoryStore

Pipeline graph:
    START
      └─► router
            ├─ route=="chat"  ──────────────────────────────────► generate ──► validate ──► END
            ├─ route=="web"   ──────────────────────────────────► generate ──► validate ──► END
            └─ route=="rag"   ──► retrieve ──► rerank ──► generate ──► validate ──► END

This file should stay THIN — all node logic lives in nodes/*.py.
Target: < 200 lines.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from src.v2.generation.state import RAGState, make_initial_state
from src.v2.generation.nodes.router import node_router
from src.v2.generation.nodes.retrieve import build_retrieve_node
from src.v2.generation.nodes.rerank import build_rerank_node
from src.v2.generation.nodes.generate import build_generate_node
from src.v2.generation.nodes.validate import node_validate_citations
from src.v2.retrieval.vector_retriever import VectorRetriever
from src.v2.retrieval.bm25_retriever import BM25Retriever
from src.v2.retrieval.dynamic_bm25 import DynamicBM25Retriever
from src.v2.retrieval.reranker import CrossEncoderReranker
from src.v2.memory.store import MemoryStore
from src.v2.tools.registry import ToolRegistry
from src.v2.core.cache import QueryCache

logger = logging.getLogger(__name__)


def _make_llm(provider: str, model: str, role: str):
    """Initialise an LLM based on provider string."""
    if provider == "openrouter":
        logger.info(f"Initialising OpenRouter {role} LLM: {model}")
        return ChatOpenAI(
            model=model,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.1,
            max_tokens=2048,
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": f"OmniMind RAG ({role})",
            },
        )
    else:
        logger.info(f"Initialising Gemini {role} LLM: {model}")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_output_tokens=2048,
        )


class RAGPipelineV2:
    """
    The v2 RAG pipeline: agentic graph with memory, routing, and
    workspace-scoped hybrid retrieval.

    Initialised ONCE at FastAPI startup; reused for all queries.
    """

    def __init__(
        self,
        qdrant_path: str = "./data/qdrant_storage",
        bm25_path: str = "./data/bm25_index.pkl",
        collection_name: str = "langchain_docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        memory_store: Optional[MemoryStore] = None,
    ) -> None:
        logger.info("=== Initialising RAGPipelineV2 ===")

        # --- Retrievers ---
        self.vector_retriever = VectorRetriever(
            qdrant_path=qdrant_path,
            collection_name=collection_name,
            embedding_model_name=embedding_model,
        )
        self.static_bm25 = BM25Retriever(bm25_path=bm25_path)
        self.dynamic_bm25 = DynamicBM25Retriever(self.vector_retriever)
        self.reranker = CrossEncoderReranker(model_name=reranker_model)

        # --- LLMs ---
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        primary_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        backup_provider = os.getenv(
            "LLM_BACKUP_PROVIDER",
            "openrouter" if provider == "gemini" else "gemini",
        )
        backup_model = os.getenv(
            "LLM_BACKUP_MODEL",
            "google/gemini-2.5-flash:free"
            if backup_provider == "openrouter"
            else "gemini-2.5-flash",
        )
        self.primary_llm = _make_llm(provider, primary_model, "primary")
        self.backup_llm = _make_llm(backup_provider, backup_model, "backup")

        # --- Memory ---
        self.memory = memory_store or MemoryStore()

        # --- Tools ---
        self.tools = ToolRegistry()

        # --- Cache ---
        self.cache = QueryCache(
            max_size=int(os.getenv("CACHE_MAX_SIZE", "100"))
        )

        # --- Graph ---
        self.app = self._build_graph()
        logger.info("✓ RAGPipelineV2 ready")

    # ------------------------------------------------------------------

    def _build_graph(self):
        """Assemble the LangGraph state machine."""
        retrieve_node = build_retrieve_node(
            self.vector_retriever, self.static_bm25, self.dynamic_bm25
        )
        rerank_node = build_rerank_node(self.reranker)
        generate_node = build_generate_node(self.primary_llm, self.backup_llm)

        builder = StateGraph(RAGState)
        builder.add_node("router", node_router)
        builder.add_node("retrieve", retrieve_node)
        builder.add_node("rerank", rerank_node)
        builder.add_node("generate", generate_node)
        builder.add_node("validate_citations", node_validate_citations)

        builder.add_edge(START, "router")

        # Conditional fork after router
        def route_decision(state: RAGState) -> str:
            return "retrieve" if state["route"] == "rag" else "generate"

        builder.add_conditional_edges("router", route_decision)
        builder.add_edge("retrieve", "rerank")
        builder.add_edge("rerank", "generate")
        builder.add_edge("generate", "validate_citations")
        builder.add_edge("validate_citations", END)

        return builder.compile()

    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        session_id: str = "",
        workspace_id: str = "default",
        search_mode: str = "combined",
    ) -> dict:
        """
        Run a question through the full v2 RAG pipeline.

        Args:
            question:     User's question.
            session_id:   Session ID for memory lookup.  Auto-created if empty.
            workspace_id: Qdrant partition to search.
            search_mode:  "docs" | "workspace" | "combined"

        Returns:
            dict with: answer, sources, retrieval_stats, session_id, route
        """
        # Ensure session exists
        if not self.memory.session_exists(session_id):
            session_id = self.memory.create_session(
                workspace_id=workspace_id,
                session_id=session_id or None,
            )

        # --- Cache check (skip for conversational turns) ---
        cached = self.cache.get(question, workspace_id, search_mode)
        if cached:
            cached["session_id"] = session_id
            cached["from_cache"] = True
            return cached

        # Fetch prior history
        history = self.memory.get_history(session_id)

        # Build initial state
        initial = make_initial_state(
            query=question,
            session_id=session_id,
            workspace_id=workspace_id,
            search_mode=search_mode,
            messages=history,
        )

        # --- Pre-fetch web results if web_search tool available ---
        # The router node will set route; we pre-warm web results so the
        # generate node has them available without a second graph pass.
        # We do a quick classify here to avoid fetching unnecessarily.
        from src.v2.generation.nodes.router import classify_query
        pre_route = classify_query(question)
        if pre_route == "web" and self.tools.web_search_available:
            try:
                web_results = self.tools.web_search.search(question)
                initial["web_results"] = web_results
            except Exception as exc:
                logger.warning(f"Web search pre-fetch failed: {exc} — falling back to RAG")
                initial["route"] = "rag"   # force RAG fallback

        # Run the graph
        final = self.app.invoke(initial)

        # Persist this turn to memory
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(
            session_id,
            "assistant",
            final["validated_answer"],
            cited_sources=[s.get("url", "") for s in final.get("sources", [])],
            retrieval_method=final.get("retrieval_method", ""),
        )

        result = {
            "answer": final["validated_answer"],
            "sources": final.get("sources", []),
            "retrieval_stats": {
                "vector_results": len(final.get("vector_results", [])),
                "bm25_results": len(final.get("bm25_results", [])),
                "fused_results": len(final.get("fused_results", [])),
                "final_chunks": len(final.get("final_chunks", [])),
                "invalid_citations_stripped": len(final.get("invalid_citations", [])),
            },
            "session_id": session_id,
            "route": final.get("route", "rag"),
            "from_cache": False,
        }

        # Cache RAG and web results (not chat — those change with context)
        if result["route"] in ("rag", "web"):
            self.cache.set(question, workspace_id, search_mode, result)

        return result
