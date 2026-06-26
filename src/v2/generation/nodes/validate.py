"""
validate.py — LangGraph Citation Validation Node (v2)

Wraps the citation_validator module as a LangGraph node.

For "chat" and "web" routes, citation validation is skipped — the
answer is passed through directly, and sources are left empty.

For "rag" routes, validates [n] references against len(final_chunks),
strips hallucinated citations, and builds the sources list for the API
response.

Unchanged from v1 logic; structural refactor only.
"""

from __future__ import annotations

import logging

from src.v2.generation.state import RAGState
from src.v2.generation.citation_validator import (
    validate_citations,
    extract_cited_sources,
)

logger = logging.getLogger(__name__)


def node_validate_citations(state: RAGState) -> dict:
    """
    LangGraph node: validate and strip hallucinated [n] citations.

    For non-RAG routes the raw_answer is promoted directly to
    validated_answer with empty sources.
    """
    route = state.get("route", "rag")

    if route != "rag":
        # No citations to validate for chat / web responses
        return {
            "validated_answer": state.get("raw_answer", ""),
            "cited_indices": [],
            "invalid_citations": [],
            "sources": [],
        }

    result = validate_citations(
        answer=state["raw_answer"],
        num_chunks=len(state.get("final_chunks", [])),
        strip_invalid=True,
    )

    sources = extract_cited_sources(
        cited_indices=result.cited_indices,
        chunks=state.get("final_chunks", []),
    )

    if result.invalid_citations:
        logger.warning(
            f"Stripped {len(result.invalid_citations)} hallucinated "
            f"citation(s): {result.invalid_citations}"
        )

    return {
        "validated_answer": result.validated_text,
        "cited_indices": result.cited_indices,
        "invalid_citations": result.invalid_citations,
        "sources": sources,
    }
