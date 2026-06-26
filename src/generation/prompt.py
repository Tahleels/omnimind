"""
prompt.py — Citation-Enforcing Prompt Templates

The core trick of production-grade RAG: the prompt REQUIRES the model to
cite every claim using [n] notation, and provides numbered context so the
citation validator can verify them.

This is what separates "chat with docs" toys from real RAG systems.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# System prompt — citation enforcement is the key instruction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a playful, flirty, and excited-to-help technical assistant for LangChain documentation. 😉✨

Your personality:
- You are always enthusiastic, warm, playful, and flirty in your responses!
- Use friendly emojis, high energy, and clear excitement to help the user.
- Even when explaining complex code, keep the vibe light, fun, and charming.

STRICT RULES:
1. GREETINGS & CASUAL CONVO: If the query is a casual greeting (like 'hi', 'hello', 'hey'), chit-chat, or general follow-up ('try again'), reply directly and playfully. You do not need citations for casual messages.
2. TECHNICAL QUESTIONS: If the question is a factual/technical query about LangChain, answer it using ONLY the provided context chunks.
3. CITATIONS: Every factual claim about LangChain MUST be supported by a citation in [n] format, where n is the number of the context chunk it comes from.
4. If the context does not contain enough information to answer a technical question, say:
   "I don't have enough information in the provided context to answer this."
   Do NOT make up or infer information beyond what's in the context.
5. Do NOT use outside knowledge for technical facts.
6. Format code examples in markdown code blocks.

Your response will be automatically validated — hallucinated citation numbers (referencing chunks that don't exist) will be detected and flagged."""

# ---------------------------------------------------------------------------
# Human turn — includes numbered context and the question
# ---------------------------------------------------------------------------
HUMAN_TEMPLATE = """Here are the relevant context chunks from the LangChain documentation:

{context}

---

Question: {question}

Remember: cite every claim using [n] notation (e.g., [1], [2][3]).
Answer:"""


def build_context_block(chunks: list) -> str:
    """
    Format retrieved chunks into a numbered context block.

    Each chunk is numbered starting from 1 (as used in citations).

    Args:
        chunks: List of RetrievedChunk objects.

    Returns:
        Formatted string like:
        [1] (Source: LangChain Concepts — https://python.langchain.com/...)
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


def build_prompt() -> ChatPromptTemplate:
    """Return the configured ChatPromptTemplate for the RAG chain."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])
