"""
test_v2_units.py — Unit tests for all v2 components that don't need
the full model stack (memory, cache, retry, workspace, dispatcher helpers).

Run with:
    pytest tests/test_v2_units.py -v
"""

import pytest
import time
import threading


# ============================================================
# Memory: MemoryStore
# ============================================================

class TestMemoryStore:
    def setup_method(self):
        from src.v2.memory.store import MemoryStore
        self.store = MemoryStore(max_history=5)

    def test_create_and_exists(self):
        sid = self.store.create_session()
        assert self.store.session_exists(sid)

    def test_explicit_session_id(self):
        sid = self.store.create_session(session_id="my-fixed-id")
        assert sid == "my-fixed-id"
        assert self.store.session_exists("my-fixed-id")

    def test_add_and_get_history(self):
        sid = self.store.create_session()
        self.store.add_message(sid, "user", "Hello")
        self.store.add_message(sid, "assistant", "Hi there")
        hist = self.store.get_history(sid)
        assert len(hist) == 2
        assert hist[0].content == "Hello"
        assert hist[1].content == "Hi there"

    def test_auto_prune_to_max_history(self):
        sid = self.store.create_session()
        for i in range(10):
            self.store.add_message(sid, "user", f"msg {i}")
        hist = self.store.get_history(sid)
        assert len(hist) == 5

    def test_clear_session(self):
        sid = self.store.create_session()
        self.store.add_message(sid, "user", "test")
        self.store.clear_session(sid)
        assert self.store.get_history(sid) == []

    def test_delete_session(self):
        sid = self.store.create_session()
        self.store.delete_session(sid)
        assert not self.store.session_exists(sid)

    def test_unknown_session_returns_empty(self):
        assert self.store.get_history("nonexistent-id") == []

    def test_thread_safety(self):
        """Multiple threads writing to different sessions must not corrupt state."""
        errors = []
        def write_session(i):
            try:
                sid = self.store.create_session()
                for j in range(20):
                    self.store.add_message(sid, "user", f"thread {i} msg {j}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_session, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread safety failure: {errors}"

    def test_stats(self):
        stats = self.store.stats()
        assert "active_sessions" in stats


# ============================================================
# Core: QueryCache
# ============================================================

class TestQueryCache:
    def setup_method(self):
        from src.v2.core.cache import QueryCache
        self.cache = QueryCache(max_size=3)

    def test_miss_then_hit(self):
        result = self.cache.get("what is LCEL?", "default", "combined")
        assert result is None
        self.cache.set("what is LCEL?", "default", "combined", {"answer": "LCEL is..."})
        result = self.cache.get("what is LCEL?", "default", "combined")
        assert result is not None
        assert result["answer"] == "LCEL is..."

    def test_normalises_key(self):
        self.cache.set("  What Is LCEL?  ", "default", "combined", {"answer": "x"})
        result = self.cache.get("what is lcel?", "default", "combined")
        assert result is not None

    def test_workspace_isolation(self):
        self.cache.set("q", "ws-A", "combined", {"answer": "A answer"})
        result = self.cache.get("q", "ws-B", "combined")
        assert result is None

    def test_lru_eviction(self):
        self.cache.set("q1", "d", "c", {"a": 1})
        self.cache.set("q2", "d", "c", {"a": 2})
        self.cache.set("q3", "d", "c", {"a": 3})
        # Access q1 to make q2 the LRU
        self.cache.get("q1", "d", "c")
        # Insert q4 — should evict q2
        self.cache.set("q4", "d", "c", {"a": 4})
        assert self.cache.get("q2", "d", "c") is None
        assert self.cache.get("q1", "d", "c") is not None

    def test_clear(self):
        self.cache.set("q", "d", "c", {"a": 1})
        self.cache.clear()
        assert self.cache.get("q", "d", "c") is None

    def test_stats(self):
        self.cache.get("miss", "d", "c")
        self.cache.set("hit", "d", "c", {"x": 1})
        self.cache.get("hit", "d", "c")
        stats = self.cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


# ============================================================
# Core: with_retry decorator
# ============================================================

class TestRetry:
    def test_succeeds_first_try(self):
        from src.v2.core.retry import with_retry
        call_count = [0]

        @with_retry(max_attempts=3, base_delay=0.01)
        def fn():
            call_count[0] += 1
            return "ok"

        result = fn()
        assert result == "ok"
        assert call_count[0] == 1

    def test_retries_on_failure(self):
        from src.v2.core.retry import with_retry
        call_count = [0]

        @with_retry(max_attempts=3, base_delay=0.01, jitter=False)
        def fn():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("transient")
            return "recovered"

        result = fn()
        assert result == "recovered"
        assert call_count[0] == 3

    def test_raises_after_max_attempts(self):
        from src.v2.core.retry import with_retry

        @with_retry(max_attempts=2, base_delay=0.01, jitter=False)
        def always_fails():
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError, match="permanent"):
            always_fails()

    def test_only_catches_specified_exceptions(self):
        from src.v2.core.retry import with_retry

        @with_retry(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            raise TypeError("not retried")

        with pytest.raises(TypeError):
            raises_type_error()


# ============================================================
# Ingestion: dispatcher helpers (no network, no models)
# ============================================================

class TestDispatcherHelpers:
    def test_chunk_text_basic(self):
        from src.v2.ingestion.dispatcher import _chunk_text
        text = " ".join([f"word{i}" for i in range(200)])
        chunks = _chunk_text(text, chunk_size=50, chunk_overlap=10)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_text_empty(self):
        from src.v2.ingestion.dispatcher import _chunk_text
        assert _chunk_text("") == []
        assert _chunk_text("   ") == []

    def test_detect_pdf(self):
        from src.v2.ingestion.dispatcher import _detect_source_type
        assert _detect_source_type("report.pdf") == "pdf"

    def test_detect_docx(self):
        from src.v2.ingestion.dispatcher import _detect_source_type
        assert _detect_source_type("notes.docx") == "docx"

    def test_detect_txt(self):
        from src.v2.ingestion.dispatcher import _detect_source_type
        assert _detect_source_type("readme.txt") == "txt"
        assert _detect_source_type("readme.md") == "txt"

    def test_detect_url(self):
        from src.v2.ingestion.dispatcher import _detect_source_type
        assert _detect_source_type("https://python.langchain.com/docs") == "url"

    def test_detect_youtube(self):
        from src.v2.ingestion.dispatcher import _detect_source_type
        assert _detect_source_type("https://www.youtube.com/watch?v=abc123") == "youtube"
        assert _detect_source_type("https://youtu.be/abc123") == "youtube"


# ============================================================
# Workspace: WorkspaceManager
# ============================================================

class TestWorkspaceManager:
    def setup_method(self, tmp_path=None):
        import tempfile, os
        from pathlib import Path
        self._tmpdir = tempfile.mkdtemp()
        self._store_path = str(Path(self._tmpdir) / "workspaces.json")
        from src.v2.workspaces.manager import WorkspaceManager
        self.mgr = WorkspaceManager(store_path=self._store_path)

    def test_default_workspace_exists(self):
        ws = self.mgr.get_workspace("default")
        assert ws is not None
        assert ws.is_default

    def test_create_workspace(self):
        ws = self.mgr.create_workspace("My Notes", "test workspace")
        assert ws.name == "My Notes"
        assert not ws.is_default
        fetched = self.mgr.get_workspace(ws.workspace_id)
        assert fetched is not None

    def test_list_workspaces(self):
        self.mgr.create_workspace("WS1")
        self.mgr.create_workspace("WS2")
        result = self.mgr.list_workspaces()
        names = [w.name for w in result]
        assert "WS1" in names
        assert "WS2" in names
        assert "LangChain Docs" in names

    def test_delete_workspace(self):
        ws = self.mgr.create_workspace("Temp")
        self.mgr.delete_workspace(ws.workspace_id)
        assert self.mgr.get_workspace(ws.workspace_id) is None

    def test_cannot_delete_default(self):
        with pytest.raises(ValueError):
            self.mgr.delete_workspace("default")

    def test_unknown_workspace_returns_none(self):
        assert self.mgr.get_workspace("nonexistent-id") is None

    def test_persistence(self):
        ws = self.mgr.create_workspace("Persisted")
        # Re-open from same file
        from src.v2.workspaces.manager import WorkspaceManager
        mgr2 = WorkspaceManager(store_path=self._store_path)
        fetched = mgr2.get_workspace(ws.workspace_id)
        assert fetched is not None
        assert fetched.name == "Persisted"


# ============================================================
# Parsers: URL / YouTube detection (no network)
# ============================================================

class TestUrlParserDetection:
    def test_rejects_youtube_url(self):
        from src.v2.ingestion.parsers.url_parser import parse_url
        with pytest.raises(ValueError, match="YouTube"):
            parse_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_extract_video_id(self):
        from src.v2.ingestion.parsers.youtube_parser import _extract_video_id
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
        assert _extract_video_id("https://example.com") is None


# ============================================================
# RRF Fusion
# ============================================================

class TestRRFusion:
    def test_basic_fusion(self):
        from src.v2.retrieval.fusion import reciprocal_rank_fusion
        from src.v2.retrieval.models import RetrievedChunk

        def make_chunk(id, text):
            return RetrievedChunk(
                chunk_id=id, text=text, source_url="http://x.com",
                source_title="X", score=0.0,
            )

        list_a = [make_chunk("a", "text a"), make_chunk("b", "text b"), make_chunk("c", "text c")]
        list_b = [make_chunk("b", "text b"), make_chunk("a", "text a"), make_chunk("d", "text d")]
        fused = reciprocal_rank_fusion(list_a, list_b)
        ids = [c.chunk_id for c in fused]
        # "a" and "b" appear in both lists — must rank higher than "c" or "d"
        assert ids.index("a") < ids.index("d")
        assert ids.index("b") < ids.index("c")

    def test_empty_lists(self):
        from src.v2.retrieval.fusion import reciprocal_rank_fusion
        result = reciprocal_rank_fusion([], [])
        assert result == []

    def test_deduplication(self):
        from src.v2.retrieval.fusion import reciprocal_rank_fusion
        from src.v2.retrieval.models import RetrievedChunk

        def make_chunk(id):
            return RetrievedChunk(chunk_id=id, text="t", source_url="u", source_title="t", score=0)

        list_a = [make_chunk("x")]
        list_b = [make_chunk("x")]
        fused = reciprocal_rank_fusion(list_a, list_b)
        assert len(fused) == 1


# ============================================================
# Router: classify_query
# ============================================================

class TestRouter:
    def test_rag_classification(self):
        from src.v2.generation.nodes.router import classify_query
        assert classify_query("What is LCEL?") == "rag"
        assert classify_query("How does RecursiveCharacterTextSplitter work?") == "rag"

    def test_chat_classification(self):
        from src.v2.generation.nodes.router import classify_query
        assert classify_query("Hello!") == "chat"
        assert classify_query("Thanks") == "chat"
        assert classify_query("Hi") == "chat"

    def test_web_classification(self):
        from src.v2.generation.nodes.router import classify_query
        assert classify_query("search the web for latest LangChain release") == "web"
        assert classify_query("what is the current price of bitcoin?") == "web"
