"""
loader.py — Document Loading for RAG Ingestion

Scrapes LangChain documentation pages and returns clean text documents
ready for chunking. Uses requests + BeautifulSoup to handle web scraping.
"""

import re
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamically discover core LangChain documentation URLs
# ---------------------------------------------------------------------------
def discover_langchain_urls() -> list[str]:
    """
    Fetch the LangChain overview page and dynamically parse core documentation URLs.
    """
    overview_url = "https://docs.langchain.com/oss/python/langchain/overview"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    fallback_urls = [
        "https://docs.langchain.com/oss/python/langchain/overview",
        "https://docs.langchain.com/oss/python/langchain/agents",
        "https://docs.langchain.com/oss/python/langchain/models",
        "https://docs.langchain.com/oss/python/langchain/messages",
        "https://docs.langchain.com/oss/python/langchain/tools",
        "https://docs.langchain.com/oss/python/langchain/streaming",
        "https://docs.langchain.com/oss/python/langchain/structured-output",
        "https://docs.langchain.com/oss/python/langchain/short-term-memory",
    ]
    
    try:
        logger.info(f"Dynamically discovering doc URLs from: {overview_url}")
        response = requests.get(overview_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "lxml")
        discovered = set()
        
        # Find all links on the overview page
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Convert relative path to absolute
            if href.startswith("/"):
                href = f"https://docs.langchain.com{href}"
            
            # Match only core components (e.g. /oss/python/langchain/agents)
            # Exclude nested sections like /middleware/... or /frontend/...
            # and ignore anchors/queries
            match = re.match(r"^https://docs\.langchain\.com/oss/python/langchain/([^/#?]+)$", href)
            if match:
                discovered.add(href)
                
        if discovered:
            discovered.add(overview_url)
            sorted_urls = sorted(list(discovered))
            logger.info(f"✓ Dynamically discovered {len(sorted_urls)} documentation URLs.")
            return sorted_urls
            
    except Exception as e:
        logger.warning(f"Failed to dynamically discover URLs: {e}. Using fallback default list.")
        
    return fallback_urls

LANGCHAIN_DOC_URLS = discover_langchain_urls()


@dataclass
class Document:
    """Represents a loaded document before chunking."""
    content: str
    url: str
    title: str
    metadata: dict = field(default_factory=dict)


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize text."""
    # Collapse multiple newlines to two
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _extract_content(soup: BeautifulSoup, url: str) -> tuple[str, str]:
    """
    Extract the main content and title from a BeautifulSoup parsed page.
    Returns (title, content) tuple.
    """
    # Extract title
    title = ""
    title_tag = soup.find("h1")
    if title_tag:
        title = title_tag.get_text(strip=True)
    elif soup.title:
        title = soup.title.get_text(strip=True)

    # Try to find main article content (LangChain docs use article/main tags)
    content_el = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"content|markdown|prose"))
        or soup.body
    )

    if content_el is None:
        return title, ""

    # Remove navigation, header, footer, and code-only blocks we don't want
    for tag in content_el.find_all(
        ["nav", "header", "footer", "script", "style",
         "aside", "button", "svg", "img"]
    ):
        tag.decompose()

    content = content_el.get_text(separator="\n")
    content = _clean_text(content)
    return title, content


def load_url(url: str, timeout: int = 15, retries: int = 2) -> Optional[Document]:
    """
    Fetch and parse a single URL.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.
        retries: Number of retry attempts on failure.

    Returns:
        A Document object, or None if the page could not be fetched.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for attempt in range(retries + 1):
        try:
            logger.debug(f"Fetching: {url} (attempt {attempt + 1})")
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            title, content = _extract_content(soup, url)

            if not content or len(content) < 100:
                logger.warning(f"Skipping {url} — content too short ({len(content)} chars).")
                return None

            logger.info(f"✓ Loaded '{title}' ({len(content)} chars) from {url}")
            return Document(
                content=content,
                url=url,
                title=title,
                metadata={"source": url, "title": title},
            )

        except requests.exceptions.RequestException as e:
            logger.warning(f"  ✗ Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)  # Exponential backoff

    logger.error(f"  ✗ All attempts failed for {url}. Skipping.")
    return None


def load_documents(
    urls: Optional[list[str]] = None,
    delay_between_requests: float = 0.5,
) -> list[Document]:
    """
    Load multiple documents from a list of URLs.

    Args:
        urls: List of URLs to load. Defaults to LANGCHAIN_DOC_URLS.
        delay_between_requests: Seconds to wait between requests (be polite!).

    Returns:
        List of successfully loaded Document objects.
    """
    if urls is None:
        urls = LANGCHAIN_DOC_URLS

    documents = []
    for i, url in enumerate(urls):
        doc = load_url(url)
        if doc:
            documents.append(doc)
        if i < len(urls) - 1:
            time.sleep(delay_between_requests)

    logger.info(f"\nLoaded {len(documents)}/{len(urls)} documents successfully.")
    return documents
