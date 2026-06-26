"""
url_parser.py — Real-Time URL Scraper for Dynamic Ingestion (v2)

Scrapes a web URL at query/upload time and extracts clean text.
Reuses the same BeautifulSoup strategy as the static loader.py but
is structured as a standalone parser that returns raw text + metadata
(no Document object — the dispatcher handles that conversion).

Handles:
  - Standard HTML documentation pages
  - News articles and blog posts
  - Simple landing pages

Does NOT handle (use youtube_parser.py for these):
  - YouTube URLs (detected and rejected with a helpful error message)
  - PDF URLs (caller should download and use pdf_parser.py)
  - Login-gated pages

Rate-limiting / politeness:
  - One attempt with a 15-second timeout
  - Retries: 2 with exponential backoff
  - Sends a descriptive User-Agent so the server knows who we are
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_YOUTUBE_PATTERN = re.compile(
    r"(youtube\.com/watch|youtu\.be/)", re.IGNORECASE
)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; AskMyDocs-RAG/2.0; "
    "+https://github.com/askmydocs)"
)


@dataclass
class ParsedDocument:
    """
    Output of any parser in this package.

    The dispatcher converts this into Chunk objects for indexing.
    """
    text: str                       # Full extracted text
    title: str                      # Document / page title
    origin: str                     # Source URL or file path
    source_type: str                # "url" | "pdf" | "docx" | "txt" | "youtube"
    page_count: Optional[int] = None
    word_count: Optional[int] = None

    def __post_init__(self):
        self.word_count = len(self.text.split())


def _clean_text(text: str) -> str:
    """Normalise whitespace."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _extract_title_and_content(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Extract the human-readable title and the main body text from a page.

    Strategy (in priority order):
      1. <h1> tag
      2. <title> tag
      3. Empty string

    Content extraction:
      1. <article> tag (news, docs, blog posts)
      2. <main> tag
      3. <div> with class containing "content", "article", "markdown", or "prose"
      4. <body> fallback
    """
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    content_el = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"content|article|markdown|prose", re.I))
        or soup.body
    )

    if content_el is None:
        return title, ""

    # Remove clutter elements
    for tag in content_el.find_all(
        ["nav", "header", "footer", "script", "style",
         "aside", "button", "svg", "img", "noscript"]
    ):
        tag.decompose()

    return title, _clean_text(content_el.get_text(separator="\n"))


def parse_url(
    url: str,
    timeout: int = 15,
    retries: int = 2,
) -> ParsedDocument:
    """
    Fetch and extract text from a web URL.

    Args:
        url:      Full URL to scrape.
        timeout:  Per-attempt request timeout in seconds.
        retries:  Number of retry attempts after the first failure.

    Returns:
        ParsedDocument with the extracted text and metadata.

    Raises:
        ValueError: If the URL is a YouTube link (use youtube_parser instead).
        RuntimeError: If all fetch attempts fail.
    """
    if _YOUTUBE_PATTERN.search(url):
        raise ValueError(
            f"YouTube URL detected — use youtube_parser.parse_youtube() instead: {url}"
        )

    headers = {"User-Agent": _USER_AGENT}
    last_exc: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            logger.info(
                f"Fetching URL (attempt {attempt + 1}/{retries + 1}): {url}"
            )
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            title, content = _extract_title_and_content(soup)

            if not content or len(content) < 100:
                raise RuntimeError(
                    f"Extracted content too short ({len(content)} chars) — "
                    "page may be JavaScript-rendered or empty."
                )

            logger.info(
                f"✓ Parsed URL: '{title}' ({len(content)} chars, {url})"
            )
            return ParsedDocument(
                text=content,
                title=title or url,
                origin=url,
                source_type="url",
            )

        except (requests.RequestException, RuntimeError) as exc:
            last_exc = exc
            logger.warning(
                f"  Attempt {attempt + 1} failed: {exc}"
            )
            if attempt < retries:
                time.sleep(2 ** attempt)  # 1s, 2s backoff

    raise RuntimeError(
        f"Failed to fetch URL after {retries + 1} attempts: {url}\n"
        f"Last error: {last_exc}"
    )
