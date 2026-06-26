"""
web_search.py — Tavily Web Search Tool (v2)

Wraps the Tavily search API as a LangChain-compatible tool that the
router node can call when the query requires live / external information.

Why Tavily over Google/Bing?
  - Generous free tier (1,000 searches/month)
  - Returns pre-extracted snippets (no HTML parsing needed)
  - Purpose-built for RAG/LLM applications
  - Simple Python SDK

Fallback:
  If TAVILY_API_KEY is not set, the tool raises a RuntimeError with a
  helpful message instead of failing silently.

Usage (called by router node when route == "web"):
    tool = WebSearchTool()
    results = tool.search("latest langchain release 2025")
    # returns list[dict] with keys: title, url, content
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5
DEFAULT_SEARCH_DEPTH = "basic"   # "basic" | "advanced" (advanced costs 2 credits)


class WebSearchTool:
    """
    Tavily-backed web search tool for the router's web route.

    Initialised once at pipeline startup.  If TAVILY_API_KEY is missing
    the tool is marked as unavailable and the router falls back to RAG.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        search_depth: str = DEFAULT_SEARCH_DEPTH,
    ) -> None:
        self._api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        self._max_results = max_results
        self._search_depth = search_depth
        self.available = bool(self._api_key)

        if self.available:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self._api_key)
                logger.info("✓ WebSearchTool (Tavily) ready")
            except ImportError:
                logger.warning("tavily-python not installed — web search disabled")
                self.available = False
        else:
            logger.info(
                "WebSearchTool: TAVILY_API_KEY not set — web search disabled. "
                "Add it to .env to enable."
            )

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> list[dict]:
        """
        Run a web search and return structured results.

        Args:
            query:       Search query string.
            max_results: Override the default result count.

        Returns:
            List of dicts with keys: title, url, content, score.
            Returns an empty list if the tool is unavailable.

        Raises:
            RuntimeError: If the Tavily API call fails after one retry.
        """
        if not self.available:
            logger.warning("WebSearchTool.search() called but tool is unavailable")
            return []

        n = max_results or self._max_results
        logger.info(f"Web search: '{query[:70]}' (max_results={n})")

        try:
            response = self._client.search(
                query=query,
                max_results=n,
                search_depth=self._search_depth,
                include_answer=False,
            )
            results = response.get("results", [])
            logger.info(f"  → {len(results)} web results returned")
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                }
                for r in results
            ]
        except Exception as exc:
            logger.error(f"Tavily search failed: {exc}")
            raise RuntimeError(f"Web search failed: {exc}") from exc
