"""
models.py — Pydantic data models for the conversation memory system.

Defines ChatMessage and Session so the memory store is fully typed
and can be serialized to JSON (for future persistence to disk/Redis).
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Allowed roles for a chat message."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A single turn in the conversation history."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Optional metadata for assistant turns
    cited_sources: list[str] = Field(default_factory=list)
    retrieval_method: Optional[str] = None

    def to_langchain_tuple(self) -> tuple[str, str]:
        """
        Convert to the (role, content) tuple format that LangChain
        ChatPromptTemplate.from_messages() accepts.
        """
        return (self.role.value, self.content)

    class Config:
        use_enum_values = True


class Session(BaseModel):
    """
    A user session holding conversation history.

    Keeps the last MAX_HISTORY messages automatically via the
    MemoryStore.add_message() method. The session also stores the
    active workspace so the RAG pipeline knows which vector
    partition to search.
    """
    session_id: str
    workspace_id: str = "default"           # Active workspace for this session
    messages: list[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_active: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def last_n_messages(self) -> list[ChatMessage]:
        """
        Return the full message list (already pruned to MAX_HISTORY by store).
        Exposed as a property so callers don't need to know the internal limit.
        """
        return self.messages

    def to_langchain_history(self) -> list[tuple[str, str]]:
        """
        Return history as a list of (role, content) tuples suitable for
        injecting into a LangChain MessagesPlaceholder.
        """
        return [msg.to_langchain_tuple() for msg in self.messages]
