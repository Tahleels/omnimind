"""
registry.py — Tool Registry for the v2 Pipeline

Central place to instantiate and expose all LangChain-compatible
tools used by the agentic pipeline.

Adding a new tool:
  1. Create the tool class in its own file (e.g. tools/my_tool.py)
  2. Import it here and instantiate it in ToolRegistry.__init__()
  3. Add it to get_tools() if it should be bound to the LLM
  4. Update router.py if it should affect routing decisions

The registry is created ONCE at pipeline startup and passed to
RAGPipelineV2 so all tools share a single initialisation cost.
"""

from __future__ import annotations

import logging

from src.v2.tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Manages all tools available to the agentic pipeline.

    Attributes:
        web_search: Tavily-backed web search (available if API key set).
    """

    def __init__(self) -> None:
        self.web_search = WebSearchTool()
        tools_available = [
            name for name, tool in self._tool_map().items()
            if getattr(tool, "available", True)
        ]
        logger.info(
            f"ToolRegistry: {len(tools_available)} tool(s) available: "
            f"{tools_available or ['none']}"
        )

    def _tool_map(self) -> dict:
        return {"web_search": self.web_search}

    @property
    def web_search_available(self) -> bool:
        return self.web_search.available

    def summary(self) -> dict:
        """Return availability dict for /health endpoint."""
        return {
            name: getattr(tool, "available", True)
            for name, tool in self._tool_map().items()
        }
