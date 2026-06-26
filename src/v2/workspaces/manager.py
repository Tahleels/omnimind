"""
manager.py — WorkspaceManager: CRUD + JSON Persistence.

Responsibilities:
  1. Load/save workspace state from data/workspaces.json
  2. Create, list, update, delete workspaces
  3. Track which sources have been ingested into each workspace
  4. Coordinate with Qdrant to purge vectors on workspace/source deletion
  5. Expose a clean interface that the API layer and ingestion pipeline use

Thread safety: All mutating operations acquire a threading.Lock before
writing to the JSON store, making it safe for FastAPI's default
single-process, multi-threaded Uvicorn mode.

For multi-process or multi-worker setups, swap the JSON store for a
SQLite or Postgres-backed store — the interface stays identical.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from src.v2.workspaces.models import (
    Workspace,
    WorkspaceSource,
    WorkspaceStore,
    SourceType,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKSPACE_ID = "default"
DEFAULT_WORKSPACE_NAME = "LangChain Docs"
WORKSPACES_FILE = Path("./data/workspaces.json")


class WorkspaceManager:
    """
    Manages workspace lifecycle: creation, source tracking, and deletion.

    Loaded once at FastAPI startup and kept alive as a singleton.

    Example usage:
        mgr = WorkspaceManager()
        ws_id = mgr.create_workspace("My Research")
        mgr.add_source(ws_id, source)
        workspaces = mgr.list_workspaces()
    """

    def __init__(
        self,
        store_path = WORKSPACES_FILE,
    ) -> None:
        self._store_path = Path(store_path)   # accept str or Path
        self._lock = threading.Lock()
        self._store: WorkspaceStore = self._load()
        self._ensure_default_workspace()
        logger.info(
            f"WorkspaceManager ready — {len(self._store.workspaces)} workspace(s) loaded"
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> WorkspaceStore:
        """Read workspaces.json from disk, returning an empty store if absent."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._store_path.exists():
            logger.info("No workspaces.json found — starting fresh")
            return WorkspaceStore()
        try:
            raw = self._store_path.read_text(encoding="utf-8")
            return WorkspaceStore.model_validate_json(raw)
        except Exception as exc:
            logger.error(f"Failed to parse workspaces.json: {exc} — starting fresh")
            return WorkspaceStore()

    def _save(self) -> None:
        """Write the current store to disk (call while holding self._lock)."""
        self._store_path.write_text(
            self._store.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _ensure_default_workspace(self) -> None:
        """Create the built-in LangChain Docs workspace if it doesn't exist."""
        with self._lock:
            if DEFAULT_WORKSPACE_ID not in self._store.workspaces:
                default = Workspace(
                    workspace_id=DEFAULT_WORKSPACE_ID,
                    name=DEFAULT_WORKSPACE_NAME,
                    description="Pre-indexed LangChain documentation (read-only).",
                    is_default=True,
                )
                self._store.upsert(default)
                self._save()
                logger.info("Created default workspace (langchain_docs)")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_workspace(
        self,
        name: str,
        description: str = "",
        workspace_id: Optional[str] = None,
    ) -> Workspace:
        """
        Create and persist a new workspace.

        Args:
            name:         Human-readable name.
            description:  Optional description for the UI.
            workspace_id: Optional explicit ID (useful for testing).

        Returns:
            The new Workspace object.
        """
        with self._lock:
            ws = Workspace(
                name=name,
                description=description,
            )
            if workspace_id:
                ws.workspace_id = workspace_id
            self._store.upsert(ws)
            self._save()
            logger.info(f"Created workspace '{name}' [{ws.workspace_id}]")
        return ws

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Return a workspace by ID or None."""
        with self._lock:
            return self._store.get(workspace_id)

    def list_workspaces(self) -> list[Workspace]:
        """Return all workspaces sorted: default first, then by creation date."""
        with self._lock:
            all_ws = self._store.list_all()
        return sorted(all_ws, key=lambda w: (not w.is_default, w.created_at))

    def delete_workspace(self, workspace_id: str) -> bool:
        """
        Delete a workspace from the JSON store.

        Callers are responsible for also purging the corresponding Qdrant
        vectors (use QdrantClient.delete with a payload filter).
        The default workspace cannot be deleted.

        Raises:
            ValueError: If attempting to delete the protected default workspace.

        Returns:
            True if deleted, False if not found.
        """
        if workspace_id == DEFAULT_WORKSPACE_ID:
            raise ValueError(
                "The default workspace cannot be deleted. "
                "Use the web UI to remove individual sources instead."
            )

        with self._lock:
            deleted = self._store.delete(workspace_id)
            if deleted:
                self._save()
                logger.info(f"Deleted workspace {workspace_id}")
        return deleted

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def add_source(
        self,
        workspace_id: str,
        source: WorkspaceSource,
    ) -> bool:
        """
        Register a newly ingested source with a workspace.

        Args:
            workspace_id: Target workspace.
            source:       WorkspaceSource metadata from the ingestion pipeline.

        Returns:
            True if the workspace was found and the source was added.
        """
        with self._lock:
            ws = self._store.get(workspace_id)
            if ws is None:
                logger.error(
                    f"Cannot add source — workspace {workspace_id} not found"
                )
                return False
            ws.add_source(source)
            self._save()
            logger.info(
                f"Added source '{source.name}' ({source.chunk_count} chunks) "
                f"to workspace [{workspace_id}]"
            )
        return True

    def remove_source(
        self,
        workspace_id: str,
        source_id: str,
    ) -> bool:
        """
        Remove a source from a workspace's metadata.

        Callers must separately delete the Qdrant vectors with a
        payload filter on source_id.

        Returns:
            True if removed, False if workspace or source not found.
        """
        with self._lock:
            ws = self._store.get(workspace_id)
            if ws is None:
                return False
            removed = ws.remove_source(source_id)
            if removed:
                self._save()
                logger.info(
                    f"Removed source {source_id} from workspace {workspace_id}"
                )
        return removed

    def get_source_ids(self, workspace_id: str) -> list[str]:
        """Return all source_ids in a workspace (for Qdrant payload filters)."""
        with self._lock:
            ws = self._store.get(workspace_id)
            if ws is None:
                return []
            return [s.source_id for s in ws.sources]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary dict suitable for the /health endpoint."""
        with self._lock:
            workspaces = self._store.list_all()
        return {
            "workspace_count": len(workspaces),
            "workspaces": [
                {
                    "id": w.workspace_id,
                    "name": w.name,
                    "sources": w.source_count,
                    "chunks": w.total_chunks,
                    "is_default": w.is_default,
                }
                for w in workspaces
            ],
        }
