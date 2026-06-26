"""
chain.py — LangGraph RAG State Machine

Orchestrates the full RAG pipeline as a LangGraph state machine with
clearly defined nodes for each stage. This gives us:
- Easy debugging (inspect state at each node)
- Modular components (swap any node independently)
- Full traceability of the pipeline

Pipeline:
  START → retrieve_hybrid → rerank → generate → validate_citations → END
"""

import logging
import os
from typing import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from src.retrieval.vector_retriever import VectorRetriever, RetrievedChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.prompt import build_prompt, build_context_block
from src.generation.citation_validator import (
    validate_citations,
    extract_cited_sources,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State Schema — passed between all graph nodes
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    """The state dictionary threaded through the LangGraph state machine."""
    query: str

    # Retrieval results
    vector_results: list[RetrievedChunk]
    bm25_results: list[RetrievedChunk]
    fused_results: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]       # After reranking

    # Generation
    context_block: str
    raw_answer: str
    validated_answer: str

    # Citations
    cited_indices: list[int]
    invalid_citations: list[int]
    sources: list[dict]

    # Metadata
    retrieval_method: str
    error: str | None


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

def node_retrieve_hybrid(state: RAGState, retrievers: dict) -> dict:
    """
    Node 1: Run vector and BM25 retrieval in parallel, then fuse with RRF.
    """
    query = state["query"]
    top_k_each = int(os.getenv("VECTOR_TOP_K", "25"))

    vector_results = retrievers["vector"].retrieve(query, top_k=top_k_each)
    bm25_results = retrievers["bm25"].retrieve(query, top_k=top_k_each)

    fused = reciprocal_rank_fusion(vector_results, bm25_results, top_k=50)

    return {
        "vector_results": vector_results,
        "bm25_results": bm25_results,
        "fused_results": fused,
        "retrieval_method": "hybrid_rrf",
    }


def node_rerank(state: RAGState, reranker: CrossEncoderReranker) -> dict:
    """
    Node 2: Cross-encoder reranking of fused candidates.
    """
    top_k = int(os.getenv("RERANKER_TOP_K", "5"))
    final_chunks = reranker.rerank(
        query=state["query"],
        candidates=state["fused_results"],
        top_k=top_k,
    )
    context_block = build_context_block(final_chunks)

    return {
        "final_chunks": final_chunks,
        "context_block": context_block,
    }


def node_generate(state: RAGState, primary_llm, backup_llm) -> dict:
    """Node 3: Generate an answer using the primary LLM, fallback to backup LLM on error.
    """
    prompt = build_prompt()
    chain = prompt | primary_llm
    try:
        response = chain.invoke({
            "context": state["context_block"],
            "question": state["query"],
        })
    except Exception as e:
        logger.warning(f"Primary LLM generation failed ({e}), falling back to backup LLM.")
        chain = prompt | backup_llm
        response = chain.invoke({
            "context": state["context_block"],
            "question": state["query"],
        })
    raw_answer = response.content if hasattr(response, "content") else str(response)
    return {"raw_answer": raw_answer}


def node_validate_citations(state: RAGState) -> dict:
    """
    Node 4: Validate all [n] citations in the generated answer.
    Strips any hallucinated references.
    """
    result = validate_citations(
        answer=state["raw_answer"],
        num_chunks=len(state["final_chunks"]),
        strip_invalid=True,
    )

    sources = extract_cited_sources(result.cited_indices, state["final_chunks"])

    return {
        "validated_answer": result.validated_text,
        "cited_indices": result.cited_indices,
        "invalid_citations": result.invalid_citations,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    The main RAG pipeline, implemented as a LangGraph state machine.

    Initializes all components once (heavy models loaded on startup)
    and serves queries efficiently.
    """

    def __init__(
        self,
        qdrant_path: str = "./data/qdrant_storage",
        bm25_path: str = "./data/bm25_index.pkl",
        collection_name: str = "langchain_docs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        logger.info("Initializing RAG Pipeline components...")

        # Load all models once at startup
        self.vector_retriever = VectorRetriever(
            qdrant_path=qdrant_path,
            collection_name=collection_name,
            embedding_model_name=embedding_model,
        )
        self.bm25_retriever = BM25Retriever(bm25_path=bm25_path)
        self.reranker = CrossEncoderReranker(model_name=reranker_model)

        # Initialize primary LLM based on provider
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        primary_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        if provider == "openrouter":
            logger.info("Initializing OpenRouter primary LLM...")
            api_key = os.getenv("OPENROUTER_API_KEY")
            self.primary_llm = ChatOpenAI(
                model=primary_model,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.1,
                max_tokens=2048,
                default_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "AskMyDocs RAG (primary)",
                },
            )
        else:
            logger.info("Initializing Google Gemini primary LLM...")
            api_key = os.getenv("GOOGLE_API_KEY")
            self.primary_llm = ChatGoogleGenerativeAI(
                model=primary_model,
                google_api_key=api_key,
                temperature=0.1,  # Low temperature for factual Q&A
                max_output_tokens=2048,
            )

        # For backward compatibility, expose primary LLM as .llm
        self.llm = self.primary_llm
        backup_provider = os.getenv("LLM_BACKUP_PROVIDER")
        backup_model = os.getenv("LLM_BACKUP_MODEL")
        if not backup_provider:
            backup_provider = "openrouter" if provider == "gemini" else "gemini"
        if not backup_model:
            backup_model = "gemini-2.5-flash" if backup_provider == "gemini" else "google/gemini-2.5-flash:free"

        # Initialize backup LLM
        if backup_provider == "openrouter":
            logger.info("Initializing OpenRouter backup LLM...")
            api_key = os.getenv("OPENROUTER_API_KEY")
            self.backup_llm = ChatOpenAI(
                model=backup_model,
                openai_api_key=api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.1,
                max_tokens=2048,
                default_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "AskMyDocs RAG (backup)",
                },
            )
        else:
            logger.info("Initializing Google Gemini backup LLM...")
            api_key = os.getenv("GOOGLE_API_KEY")
            self.backup_llm = ChatGoogleGenerativeAI(
                model=backup_model,
                google_api_key=api_key,
                temperature=0.1,
                max_output_tokens=2048,
            )

        # Build the LangGraph
        self.app = self._build_graph()
        logger.info("✓ RAG Pipeline ready")

    def _build_graph(self) -> StateGraph:
        """Assemble the LangGraph state machine."""
        retrievers = {
            "vector": self.vector_retriever,
            "bm25": self.bm25_retriever,
        }
        reranker = self.reranker
        llm = self.llm

        # Wrap nodes with their dependencies using closures
        def retrieve_node(state): return node_retrieve_hybrid(state, retrievers)
        def rerank_node(state): return node_rerank(state, reranker)
        def generate_node(state):
            return node_generate(state, self.primary_llm, self.backup_llm)
        def validate_node(state): return node_validate_citations(state)

        builder = StateGraph(RAGState)
        builder.add_node("retrieve_hybrid", retrieve_node)
        builder.add_node("rerank", rerank_node)
        builder.add_node("generate", generate_node)
        builder.add_node("validate_citations", validate_node)

        builder.add_edge(START, "retrieve_hybrid")
        builder.add_edge("retrieve_hybrid", "rerank")
        builder.add_edge("rerank", "generate")
        builder.add_edge("generate", "validate_citations")
        builder.add_edge("validate_citations", END)

        return builder.compile()

    def ask(self, question: str) -> dict:
        """
        Run a question through the full RAG pipeline.

        Args:
            question: The user's question.

        Returns:
            dict with:
                - answer: The validated, cited answer
                - sources: List of cited sources with title, url, excerpt
                - retrieval_stats: Counts from each retrieval stage
        """
        initial_state: RAGState = {
            "query": question,
            "vector_results": [],
            "bm25_results": [],
            "fused_results": [],
            "final_chunks": [],
            "context_block": "",
            "raw_answer": "",
            "validated_answer": "",
            "cited_indices": [],
            "invalid_citations": [],
            "sources": [],
            "retrieval_method": "",
            "error": None,
        }

        final_state = self.app.invoke(initial_state)

        return {
            "answer": final_state["validated_answer"],
            "sources": final_state["sources"],
            "retrieval_stats": {
                "vector_results": len(final_state["vector_results"]),
                "bm25_results": len(final_state["bm25_results"]),
                "fused_results": len(final_state["fused_results"]),
                "final_chunks": len(final_state["final_chunks"]),
                "invalid_citations_stripped": len(final_state["invalid_citations"]),
            },
        }
