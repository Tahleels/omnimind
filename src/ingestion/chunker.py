"""
chunker.py — Semantic Text Splitting for RAG Ingestion

Splits raw documents into chunks that are small enough to embed but large
enough to contain meaningful context. Uses RecursiveCharacterTextSplitter
which respects paragraph and sentence boundaries.
"""

import logging
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import Document

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A processed text chunk ready for embedding and indexing."""
    text: str
    chunk_id: str          # Unique ID: "<url_hash>_<chunk_index>"
    source_url: str
    source_title: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Split documents into overlapping text chunks.

    Design decisions:
    - chunk_size=512 tokens (approx 400 words): Large enough for context,
      small enough to embed accurately.
    - chunk_overlap=64: Prevents information loss at chunk boundaries.
    - Separators: Tries to split on double-newlines (paragraphs) first,
      then single newlines, then sentences, then words. This respects
      natural semantic boundaries.

    Args:
        documents: List of loaded Document objects.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of Chunk objects ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: list[Chunk] = []

    for doc in documents:
        # Generate a short hash from the URL for stable chunk IDs
        url_hash = str(abs(hash(doc.url)) % 10_000_000).zfill(7)

        raw_chunks = splitter.split_text(doc.content)

        # Filter out chunks that are too short to be meaningful
        raw_chunks = [c for c in raw_chunks if len(c.strip()) > 50]

        for i, text in enumerate(raw_chunks):
            chunk = Chunk(
                text=text.strip(),
                chunk_id=f"{url_hash}_{i:04d}",
                source_url=doc.url,
                source_title=doc.title,
                chunk_index=i,
                metadata={
                    "source": doc.url,
                    "title": doc.title,
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                },
            )
            all_chunks.append(chunk)

        logger.info(
            f"  Split '{doc.title}' → {len(raw_chunks)} chunks"
        )

    logger.info(f"\nTotal chunks created: {len(all_chunks)}")
    return all_chunks
