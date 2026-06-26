"""
generate.py — LangGraph Generation Node (v2)

Changes over v1:
  - Conversation history from MemoryStore is injected into the prompt
    via a MessagesPlaceholder so the LLM can reference prior turns.
  - Handles all three routes:
      "rag"  — standard cited answer using retrieved context
      "chat" — playful reply without document context
      "web"  — synthesise answer from web_results snippets
  - Primary → Backup LLM fallback retained from v1.
  - Uses build_prompt_v2() which accepts history tuples.

Node function signature stays as (state: RAGState) -> dict so
LangGraph wires it identically to v1.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.v2.generation.state import RAGState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route-specific prompt builders
# ---------------------------------------------------------------------------

def _build_rag_messages(state: RAGState) -> list[tuple[str, str]]:
    """
    Build the full message list for a RAG query:
        system → history turns → human (context + question)
    """
    from src.v2.generation.prompt import (
        RAG_SYSTEM_PROMPT,
        RAG_HUMAN_TEMPLATE,
    )

    messages: list[tuple[str, str]] = [("system", RAG_SYSTEM_PROMPT)]

    # Inject prior conversation turns (up to MAX_HISTORY already enforced
    # by MemoryStore — no trimming needed here)
    for msg in state.get("messages", []):
        messages.append((msg.role if isinstance(msg.role, str) else msg.role.value, msg.content))

    # Final human turn with numbered context
    # NOTE: We use str.replace() instead of .format() because context
    # and question may contain curly braces (code snippets, JSON, etc.)
    # which would crash Python's str.format() parser.
    human_text = RAG_HUMAN_TEMPLATE.replace("{context}", state["context_block"]).replace("{question}", state["query"])
    messages.append(("human", human_text))
    return messages


def _build_chat_messages(state: RAGState) -> list[tuple[str, str]]:
    """
    Build messages for a casual/chitchat query (no document context).
    """
    from src.v2.generation.prompt import CHAT_SYSTEM_PROMPT

    messages: list[tuple[str, str]] = [("system", CHAT_SYSTEM_PROMPT)]
    for msg in state.get("messages", []):
        messages.append((msg.role if isinstance(msg.role, str) else msg.role.value, msg.content))
    messages.append(("human", state["query"]))
    return messages


def _build_web_messages(state: RAGState) -> list[tuple[str, str]]:
    """
    Build messages for a web-search answer using external snippets.
    """
    from src.v2.generation.prompt import WEB_SYSTEM_PROMPT

    snippets = "\n\n".join(
        f"[{i+1}] {r.get('title','')}\n{r.get('content','')}"
        for i, r in enumerate(state.get("web_results", []))
    )
    messages: list[tuple[str, str]] = [("system", WEB_SYSTEM_PROMPT)]
    for msg in state.get("messages", []):
        messages.append((msg.role if isinstance(msg.role, str) else msg.role.value, msg.content))
    messages.append((
        "human",
        f"Web search results:\n\n{snippets}\n\n---\n\nQuestion: {state['query']}\nAnswer:",
    ))
    return messages


# ---------------------------------------------------------------------------
# LangGraph node factory
# ---------------------------------------------------------------------------

def build_generate_node(primary_llm: "BaseChatModel", backup_llm: "BaseChatModel"):
    """
    Factory returning the generate node with LLMs injected via closure.

    Args:
        primary_llm: First-choice language model.
        backup_llm:  Fallback model if the primary raises an exception.

    Returns:
        A node function: (state: RAGState) -> dict
    """
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    def _tuples_to_messages(tuples: list[tuple[str, str]]):
        """Convert (role, content) tuples to LangChain message objects.

        This bypasses ChatPromptTemplate entirely, which is critical because
        document content often contains {curly braces} (JSON, code, CSS)
        that ChatPromptTemplate incorrectly interprets as template variables.
        """
        msgs = []
        for role, content in tuples:
            if role == "system":
                msgs.append(SystemMessage(content=content))
            elif role in ("human", "user"):
                msgs.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                msgs.append(AIMessage(content=content))
            else:
                msgs.append(HumanMessage(content=content))
        return msgs

    def node_generate(state: RAGState) -> dict:
        """
        LangGraph node: generate an answer using the appropriate prompt.

        Selects the message format based on state["route"], then calls
        the primary LLM with a fallback to backup_llm on any exception.
        """
        route = state.get("route", "rag")

        if route == "chat":
            msg_tuples = _build_chat_messages(state)
        elif route == "web":
            msg_tuples = _build_web_messages(state)
        else:
            msg_tuples = _build_rag_messages(state)

        messages = _tuples_to_messages(msg_tuples)

        try:
            response = primary_llm.invoke(messages)
        except Exception as exc:
            logger.warning(
                f"Primary LLM failed ({exc}), trying backup LLM..."
            )
            response = backup_llm.invoke(messages)

        raw_answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )
        logger.info(
            f"Generated answer ({route} route, {len(raw_answer)} chars)"
        )
        return {"raw_answer": raw_answer}

    return node_generate

