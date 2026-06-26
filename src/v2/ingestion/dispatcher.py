"""
dispatcher.py — Ingestion Dispatcher for Dynamic Sources (v2)

The dispatcher is the single entry point for all dynamic ingestion.
It:
  1. Detects the source type (URL, PDF bytes, DOCX bytes, YouTube URL, TXT).
  2. Routes to the correct parser.
  3. Chunks the extracted text using the shared chunker logic.
  4. Embeds the chunks using the shared VectorRetriever embed helper.
  5. Upserts the vectors into Qdrant with workspace_id + source_id payloads.
  6. Returns an IngestResult that the API layer uses to build a WorkspaceSource.

This module intentionally has NO knowledge of FastAPI or HTTP — it only
knows about files, text, and the retrieval layer.  This makes it unit-
testable without starting a server.

Design:
  - All parsers accept either a file path or raw bytes so this dispatcher
    can handle both on-disk files and uploaded bytes identically.
  - Chunking is delegated to the same chunker used for static ingestion
    to guarantee consistent chunk size across the entire Qdrant collection.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from src.v2.ingestion.parsers.url_parser import parse_url, ParsedDocument
from src.v2.ingestion.parsers.pdf_parser import parse_pdf
from src.v2.ingestion.parsers.docx_parser import parse_docx
from src.v2.ingestion.parsers.youtube_parser import parse_youtube

logger = logging.getLogger(__name__)

_YOUTUBE_PATTERN = re.compile(
    r"(youtube\.com/watch|youtu\.be/)", re.IGNORECASE
)

# Default chunking parameters (kept in sync with scripts/ingest.py defaults)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


@dataclass
class IngestResult:
    """
    Result returned to the API after successful dynamic ingestion.

    The API layer uses this to build a WorkspaceSource and register it
    with the WorkspaceManager.
    """
    source_id: str
    source_type: str
    name: str
    origin: str
    chunk_count: int
    page_count: Optional[int] = None
    word_count: Optional[int] = None


def _chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Uses a simple sliding window approach.  For production use with large
    documents, consider replacing with RecursiveCharacterTextSplitter.
    """
    if not text:
        return []

    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def _detect_source_type(
    filename: str,
    content: Optional[bytes] = None,
) -> str:
    """
    Detect the source type from filename extension.

    Args:
        filename: Original filename (may be a URL string or local path).
        content:  Optional raw bytes — not currently used for detection.

    Returns:
        One of: "pdf" | "docx" | "txt" | "url" | "youtube"
    """
    lower = filename.lower().strip()

    if _YOUTUBE_PATTERN.search(lower):
        return "youtube"
    if lower.startswith("http://") or lower.startswith("https://"):
        return "url"

    suffix = Path(lower).suffix
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".docx", ".doc"):
        return "docx"
    if suffix in (".txt", ".md", ".rst"):
        return "txt"

    return "url"  # Sensible fallback for unknown URLs


def ingest_source(
    source: Union[str, bytes],
    filename: str,
    workspace_id: str,
    vector_retriever,               # VectorRetriever instance (injected)
    embedding_model,                # SentenceTransformer instance (injected)
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    source_id: Optional[str] = None,
) -> IngestResult:
    """
    Parse, chunk, embed, and index a document into a workspace.

    Args:
        source:           Raw bytes (file upload) or a URL string.
        filename:         Original filename or URL — used for type detection.
        workspace_id:     Target workspace for Qdrant payload filtering.
        vector_retriever: Shared VectorRetriever for upsert and embedding.
        embedding_model:  SentenceTransformer for batch embedding.
        chunk_size:       Words per chunk (default 512).
        chunk_overlap:    Overlap words between chunks (default 64).
        source_id:        Optional explicit source UUID.

    Returns:
        IngestResult with chunk count and metadata.

    Raises:
        ValueError:   If the source cannot be parsed (wrong format, no text).
        RuntimeError: If embedding or Qdrant upsert fails.
    """
    sid = source_id or str(uuid.uuid4())
    source_type = _detect_source_type(
        filename,
        content=source if isinstance(source, bytes) else None,
    )

    logger.info(
        f"Ingesting source [{source_type}] '{filename}' → "
        f"workspace='{workspace_id}', source_id={sid}"
    )

    # --- Step 1: Parse ---
    doc: ParsedDocument
    if source_type == "youtube":
        url = source if isinstance(source, str) else filename
        doc = parse_youtube(url)
    elif source_type == "url":
        url = source if isinstance(source, str) else filename
        doc = parse_url(url)
    elif source_type == "pdf":
        doc = parse_pdf(source, filename=filename)
    elif source_type == "docx":
        doc = parse_docx(source, filename=filename)
    elif source_type == "txt":
        text = source.decode("utf-8", errors="replace") if isinstance(source, bytes) else source
        doc = ParsedDocument(
            text=text,
            title=filename,
            origin=filename,
            source_type="txt",
        )
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

    # --- Step 2: Chunk ---
    raw_chunks = _chunk_text(doc.text, chunk_size, chunk_overlap)
    if not raw_chunks:
        raise ValueError(
            f"No chunks generated from '{filename}'. "
            "The document may be too short."
        )

    logger.info(f"  Chunked into {len(raw_chunks)} chunks")

    # --- Step 3: Embed ---
    vectors = embedding_model.encode(
        raw_chunks,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()

    # --- Step 4: Build payloads ---
    chunk_payloads = [
        {
            "chunk_id": f"{sid}_{i}",
            "text": chunk_text,
            "source_url": doc.origin,
            "source_title": doc.title,
            "workspace_id": workspace_id,
            "source_id": sid,
            "chunk_index": i,
        }
        for i, chunk_text in enumerate(raw_chunks)
    ]

    # --- Step 5: Upsert to Qdrant ---
    vector_retriever.upsert_chunks(chunk_payloads, vectors)

    logger.info(
        f"✓ Ingested '{doc.title}': {len(raw_chunks)} chunks "
        f"→ workspace '{workspace_id}'"
    )

    return IngestResult(
        source_id=sid,
        source_type=source_type,
        name=doc.title,
        origin=doc.origin,
        chunk_count=len(raw_chunks),
        page_count=doc.page_count,
        word_count=doc.word_count,
    )
