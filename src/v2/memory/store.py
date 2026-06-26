"""
store.py — In-Memory Session Store with 10-message history per session.

Design choices:
  - Pure in-process dict (fast, zero deps).  For multi-worker deployments
    this would be replaced by a Redis-backed store — the interface stays
    identical so swapping is a one-line change.
  - Thread-safe via a threading.Lock (FastAPI runs in a single process
    by default; asyncio safety is handled by the sync lock).
  - Auto-generates UUIDs for new sessions.
  - Auto-prunes to MAX_HISTORY messages (oldest drops first).

Usage:
    store = MemoryStore()
    sid = store.create_session()
    store.add_message(sid, "user", "Hello!")
    history = store.get_history(sid)   # list[ChatMessage]
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Optional

from src.v2.memory.models import ChatMessage, MessageRole, Session

logger = logging.getLogger(__name__)

# Maximum messages retained per session (oldest evicted first)
MAX_HISTORY: int = 10


class MemoryStore:
    """
    Thread-safe in-memory store for chat session history.

    All public methods are synchronous so they can be called from both
    sync and async FastAPI route handlers without extra glue code.
    """

    def __init__(self, max_history: int = MAX_HISTORY) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.max_history = max_history
        logger.info(
            f"MemoryStore initialized (max_history={self.max_history})"
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(
        self,
        workspace_id: str = "default",
        session_id: Optional[str] = None,
    ) -> str:
        """
        Create a new session and return its ID.

        Args:
            workspace_id: The workspace this session is scoped to.
            session_id:   Optional explicit ID (useful for testing).

        Returns:
            The session_id string.
        """
        sid = session_id or str(uuid.uuid4())
        with self._lock:
            if sid not in self._sessions:
                self._sessions[sid] = Session(
                    session_id=sid,
                    workspace_id=workspace_id,
                )
                logger.debug(f"Created session {sid} (workspace={workspace_id})")
        return sid

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return the Session object or None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def session_exists(self, session_id: str) -> bool:
        """Return True if the session_id is currently active."""
        with self._lock:
            return session_id in self._sessions

    def clear_session(self, session_id: str) -> bool:
        """
        Wipe all messages from a session (but keep the session alive).

        Returns:
            True if the session was found and cleared, False otherwise.
        """
        with self._lock:
            if session_id not in self._sessions:
                return False
            self._sessions[session_id].messages.clear()
            logger.info(f"Cleared messages for session {session_id}")
            return True

    def delete_session(self, session_id: str) -> bool:
        """
        Fully remove a session and all its messages.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if session_id not in self._sessions:
                return False
            del self._sessions[session_id]
            logger.info(f"Deleted session {session_id}")
            return True

    def set_workspace(self, session_id: str, workspace_id: str) -> bool:
        """
        Update the active workspace for a session.

        Returns:
            True if updated, False if session not found.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return False
            session.workspace_id = workspace_id
            return True

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        cited_sources: Optional[list[str]] = None,
        retrieval_method: Optional[str] = None,
    ) -> ChatMessage:
        """
        Append a message to a session, auto-creating the session if needed.

        Prunes oldest messages when history exceeds max_history.

        Args:
            session_id:       Target session.
            role:             "user" | "assistant" | "system"
            content:          Message text.
            cited_sources:    Optional list of URLs cited (assistant only).
            retrieval_method: Optional pipeline method tag (e.g. "hybrid_rrf").

        Returns:
            The newly created ChatMessage.
        """
        msg = ChatMessage(
            role=MessageRole(role),
            content=content,
            cited_sources=cited_sources or [],
            retrieval_method=retrieval_method,
        )

        with self._lock:
            # Auto-create session if needed
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(session_id=session_id)
                logger.debug(f"Auto-created session {session_id} on first message")

            session = self._sessions[session_id]
            session.messages.append(msg)
            session.last_active = datetime.now(timezone.utc)

            # Prune to max_history (keep most recent)
            if len(session.messages) > self.max_history:
                excess = len(session.messages) - self.max_history
                session.messages = session.messages[excess:]
                logger.debug(
                    f"Pruned {excess} old message(s) from session {session_id}"
                )

        logger.debug(
            f"[{session_id[:8]}] {role}: {content[:60]}..."
            if len(content) > 60 else
            f"[{session_id[:8]}] {role}: {content}"
        )
        return msg

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """
        Return all stored messages for a session (up to max_history).

        Returns an empty list if the session doesn't exist.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            return list(session.messages)  # Return a copy

    def get_langchain_history(
        self, session_id: str
    ) -> list[tuple[str, str]]:
        """
        Return history as (role, content) tuples for LangChain prompts.

        This is the format expected by ChatPromptTemplate with a
        MessagesPlaceholder.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            return session.to_langchain_history()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return basic store statistics for the /metrics endpoint."""
        with self._lock:
            total_sessions = len(self._sessions)
            total_messages = sum(
                len(s.messages) for s in self._sessions.values()
            )
        return {
            "active_sessions": total_sessions,
            "total_messages": total_messages,
            "max_history_per_session": self.max_history,
        }
