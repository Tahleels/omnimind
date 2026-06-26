"""
pdf_parser.py — PDF Text Extraction for Dynamic Ingestion (v2)

Uses pypdf to extract text from uploaded PDF files.

Design decisions:
  - Page-by-page extraction with page number tracking.
  - Falls back to empty string for pages that are image-only (no text layer).
  - Returns a single merged ParsedDocument (the dispatcher handles chunking).
  - Does NOT use OCR — pypdf only extracts text from searchable PDFs.
    For scanned/image PDFs, the text will be empty and the user is warned.

Why pypdf over pdfminer?
  - Simpler API for pure text extraction.
  - Well-maintained, fast, and already in requirements.txt.
  - pdfminer is an optional future upgrade for complex layout preservation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from pypdf import PdfReader

from src.v2.ingestion.parsers.url_parser import ParsedDocument

logger = logging.getLogger(__name__)

# Minimum characters per page to consider it "text-bearing"
MIN_PAGE_CHARS = 30


def parse_pdf(
    source: Union[str, Path, bytes],
    filename: str = "document.pdf",
) -> ParsedDocument:
    """
    Extract text from a PDF file or raw bytes.

    Args:
        source:   File path (str or Path) OR raw bytes from an UploadFile.
        filename: Human-readable filename (used for the document title if no
                  title is found in the PDF metadata).

    Returns:
        ParsedDocument with concatenated text from all pages.

    Raises:
        ValueError: If no text could be extracted (likely a scanned PDF).
        RuntimeError: If pypdf cannot read the file.
    """
    try:
        if isinstance(source, bytes):
            import io
            reader = PdfReader(io.BytesIO(source))
        else:
            reader = PdfReader(str(source))
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF '{filename}': {exc}") from exc

    num_pages = len(reader.pages)
    logger.info(f"Parsing PDF '{filename}' ({num_pages} pages)...")

    # Try to get a title from PDF metadata
    title = filename
    if reader.metadata:
        meta_title = reader.metadata.get("/Title", "").strip()
        if meta_title:
            title = meta_title

    page_texts: list[str] = []
    empty_pages: list[int] = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
            cleaned = raw.strip()
            if len(cleaned) >= MIN_PAGE_CHARS:
                # Prefix each page block so the chunker can use page breaks
                page_texts.append(f"[Page {page_num}]\n{cleaned}")
            else:
                empty_pages.append(page_num)
        except Exception as exc:
            logger.warning(f"  Failed to extract page {page_num}: {exc}")
            empty_pages.append(page_num)

    if empty_pages:
        logger.warning(
            f"  {len(empty_pages)}/{num_pages} pages had no extractable text "
            f"(possibly image-only): pages {empty_pages[:10]}"
        )

    full_text = "\n\n".join(page_texts)

    if not full_text.strip():
        raise ValueError(
            f"No text could be extracted from '{filename}'. "
            "This may be a scanned/image-only PDF. "
            "Please use a text-based PDF or convert it first."
        )

    logger.info(
        f"✓ Parsed PDF '{title}': {len(full_text)} chars, "
        f"{num_pages - len(empty_pages)}/{num_pages} text pages"
    )

    return ParsedDocument(
        text=full_text,
        title=title,
        origin=str(source) if not isinstance(source, bytes) else filename,
        source_type="pdf",
        page_count=num_pages,
    )
