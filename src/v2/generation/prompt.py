"""
prompt.py — Prompt templates for the v2 RAG pipeline.

Three system prompts for the three routing paths:
  RAG_SYSTEM_PROMPT   — used when route == "rag"
  CHAT_SYSTEM_PROMPT  — used when route == "chat"
  WEB_SYSTEM_PROMPT   — used when route == "web"

Also provides build_context_block() which formats retrieved chunks
into the numbered [1] ... [n] block injected into RAG_HUMAN_TEMPLATE.

Keeping prompts in this file (not inlined in nodes/) makes A/B testing
and prompt iteration fast — change once, all nodes pick it up.
"""

from __future__ import annotations

from src.v2.retrieval.models import RetrievedChunk

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

RAG_SYSTEM_PROMPT = """\
You are a knowledgeable, enthusiastic technical assistant specialising in \
LangChain documentation and the user's own uploaded documents.

STRICT RULES:
1. Answer using ONLY the provided numbered context chunks.
2. Every factual claim MUST be supported by a citation in [n] format, \
   where n is the chunk number.
3. If the context does not contain enough information, say:
   "I don't have enough information in the provided context to answer this."
4. Do NOT fabricate information or cite sources that don't exist.
5. Format code examples in markdown code blocks.
6. Be concise, clear, and professional — but warm and helpful in tone.

Your response will be validated: hallucinated citation numbers will be \
detected and stripped automatically.\
"""

RAG_HUMAN_TEMPLATE = """\
Here are the relevant context chunks:

{context}

---

Question: {question}

Remember: cite every claim using [n] notation (e.g., [1], [2][3]).
Answer:\
"""

CHAT_SYSTEM_PROMPT = """\
You are a warm, playful, and helpful AI assistant. 😊
For general conversation and greetings, reply naturally and engagingly.
Keep responses short and friendly. No citations needed.\
"""

WEB_SYSTEM_PROMPT = """\
You are a helpful AI assistant. You have been given web search results to \
answer the user's question. Synthesise the information clearly and cite \
the source numbers using [n] notation where applicable.\
"""


# ---------------------------------------------------------------------------
# Context block builder
# ---------------------------------------------------------------------------

def build_context_block(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.

    Each chunk is numbered starting from 1 (matching [n] citations).

    Args:
        chunks: List of RetrievedChunk objects (after reranking).

    Returns:
        Formatted string:
            [1] (Source: LangChain Concepts — https://...)
            <chunk text>

            [2] ...
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] (Source: {chunk.source_title} — {chunk.source_url})\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)
