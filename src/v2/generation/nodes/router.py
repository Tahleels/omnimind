"""
router.py — LangGraph Router Node (v2)

The router is the first node in the v2 pipeline.  It classifies the
incoming query into one of three routes and sets state["route"]:

    "chat"  — Casual greeting / chitchat, no retrieval needed.
              The generate node will reply playfully without context.

    "rag"   — Factual question: run hybrid retrieval + reranking.
              This is the default for anything not clearly conversational.

    "web"   — User explicitly requests up-to-date or external info
              (e.g. "latest langchain release", "search the web for…")
              OR user pasted a URL for live ingestion.

Design choice: We use a lightweight heuristic classifier rather than
calling an LLM for routing.  This keeps latency at ~0 ms and avoids
spending API quota on routing decisions.  If the heuristics aren't
precise enough, swap classify_query() for an LLM call.
"""

from __future__ import annotations

import logging
import re

from src.v2.generation.state import RAGState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

# Casual/greeting signals — if the whole query matches, route to "chat"
_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|what'?s up|howdy|sup|yo|good (morning|afternoon|evening)|"
    r"how are you|thanks|thank you|great|awesome|cool|nice|ok|okay|got it|"
    r"try again|retry|please retry|can you retry|lol|lmao|haha|😊|😄)[\s!?.]*$",
    re.IGNORECASE,
)

# Web-search / external-info signals
_WEB_PATTERNS = re.compile(
    r"(search the web|google|latest|recent|news|as of \d{4}|current|"
    r"right now|today|this week|live|real.?time|http[s]?://|www\.)",
    re.IGNORECASE,
)

# URL detection (user pasted a link)
_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


def classify_query(query: str) -> str:
    """
    Classify a query string into a route label.

    Args:
        query: The raw user query.

    Returns:
        One of: "chat" | "rag" | "web"
    """
    stripped = query.strip()

    # Short casual message → chat
    if _GREETING_PATTERNS.match(stripped):
        return "chat"

    # Explicitly requests web search or contains a URL
    if _WEB_PATTERNS.search(stripped) or _URL_PATTERN.search(stripped):
        return "web"

    # Everything else → RAG
    return "rag"


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def node_router(state: RAGState) -> dict:
    """
    LangGraph node: classify the query and set the route.

    Returns a partial state update — only the "route" key is set here.
    Downstream nodes check state["route"] to decide what to do.

    Conditional edges in chain.py then fork the graph based on this value.
    """
    query = state["query"]
    route = classify_query(query)

    logger.info(f"Router → '{route}' for query: '{query[:60]}'")

    return {"route": route}
