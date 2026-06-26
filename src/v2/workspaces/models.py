"""
models.py — Pydantic data models for the Workspace system.

A Workspace is a named collection of document sources (PDFs, URLs,
YouTube transcripts) that share a single Qdrant payload partition
(filtered by workspace_id).

These models are used for:
  - JSON persistence (data/workspaces.json)
  - API request/response schemas (via WorkspaceManager)
  - Qdrant payload metadata attached to each vector
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
import uuid


class SourceType(str, Enum):
    """Supported document source types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    URL = "url"
    YOUTUBE = "youtube"
    LANGCHAIN_DOCS = "langchain_docs"   # Built-in default workspace


class WorkspaceSource(BaseModel):
    """
    A single ingested source document within a workspace.

    Stored in the workspace JSON for UI display and for re-ingestion
    if the workspace is ever rebuilt.
    """
    source_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    source_type: SourceType
    name: str                          # Human-readable display name
    origin: str                        # File path or URL
    chunk_count: int = 0
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    # Extra metadata depending on source type
    page_count: Optional[int] = None   # For PDFs
    word_count: Optional[int] = None

    class Config:
        use_enum_values = True


class Workspace(BaseModel):
    """
    A named document workspace containing one or more sources.

    The workspace_id is used as a Qdrant payload filter key so all
    workspace documents share a single Qdrant collection but remain
    logically isolated.
    """
    workspace_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    name: str
    description: str = ""
    sources: list[WorkspaceSource] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    is_default: bool = False           # True only for the built-in langchain_docs workspace

    @property
    def total_chunks(self) -> int:
        """Total number of indexed chunks across all sources."""
        return sum(s.chunk_count for s in self.sources)

    @property
    def source_count(self) -> int:
        """Number of distinct sources in this workspace."""
        return len(self.sources)

    def add_source(self, source: WorkspaceSource) -> None:
        """Append a source and update last_modified timestamp."""
        self.sources.append(source)
        self.last_modified = datetime.now(timezone.utc)

    def remove_source(self, source_id: str) -> bool:
        """
        Remove a source by ID.

        Returns:
            True if found and removed, False otherwise.
        """
        before = len(self.sources)
        self.sources = [s for s in self.sources if s.source_id != source_id]
        if len(self.sources) < before:
            self.last_modified = datetime.now(timezone.utc)
            return True
        return False


class WorkspaceStore(BaseModel):
    """
    Root object persisted to data/workspaces.json.

    Stores all workspaces including the default "langchain_docs" workspace
    created automatically on first run.
    """
    version: str = "1"
    workspaces: dict[str, Workspace] = Field(default_factory=dict)

    def get(self, workspace_id: str) -> Optional[Workspace]:
        return self.workspaces.get(workspace_id)

    def list_all(self) -> list[Workspace]:
        return list(self.workspaces.values())

    def upsert(self, workspace: Workspace) -> None:
        self.workspaces[workspace.workspace_id] = workspace

    def delete(self, workspace_id: str) -> bool:
        if workspace_id in self.workspaces:
            del self.workspaces[workspace_id]
            return True
        return False
