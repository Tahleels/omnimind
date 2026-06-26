"""
ingest.py — CLI Ingestion Runner (v2-compatible)

Run this script once before starting the server to scrape the LangChain
documentation, chunk it, embed it, and build both the vector and BM25 indexes.

All vectors are tagged with workspace_id="default" so they're compatible
with the v2 workspace filter system from day one.

Usage:
    python scripts/ingest.py [--recreate]

Options:
    --recreate    Drop and rebuild all indexes from scratch.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.loader import load_documents, LANGCHAIN_DOC_URLS
from src.ingestion.chunker import chunk_documents
from src.ingestion.indexer import run_ingestion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest LangChain docs into RAG indexes.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop existing indexes and rebuild from scratch.",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        help="Optional list of URLs to ingest instead of the default set.",
    )
    args = parser.parse_args()

    urls = args.urls or LANGCHAIN_DOC_URLS

    logger.info(f"Starting ingestion of {len(urls)} URLs...")
    logger.info(f"Recreate indexes: {args.recreate}")
    print()

    # Step 1: Load documents
    logger.info("=== STEP 1: Loading Documents ===")
    documents = load_documents(urls=urls)
    if not documents:
        logger.error("No documents loaded. Check your internet connection.")
        sys.exit(1)

    # Step 2: Chunk
    logger.info("\n=== STEP 2: Chunking Documents ===")
    chunks = chunk_documents(documents, chunk_size=512, chunk_overlap=64)
    if not chunks:
        logger.error("No chunks created. Something went wrong during splitting.")
        sys.exit(1)

    # Step 3: Embed + Index
    # Stamp workspace_id="default" on all chunks so v2 workspace filters work.
    logger.info("\n=== STEP 3: Embedding & Indexing (workspace_id=default) ===")
    for chunk in chunks:
        if not hasattr(chunk, "metadata"):
            chunk.metadata = {}
        chunk.metadata["workspace_id"] = "default"
        chunk.metadata["source_id"] = "langchain_docs_default"

    qdrant_path = os.getenv("QDRANT_PATH", "./data/qdrant_storage")
    bm25_path = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
    collection_name = os.getenv("COLLECTION_NAME", "langchain_docs")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    qdrant_client, bm25_index = run_ingestion(
        chunks=chunks,
        embedding_model_name=embedding_model,
        qdrant_path=qdrant_path,
        bm25_path=bm25_path,
        collection_name=collection_name,
        recreate=args.recreate,
    )

    print()
    logger.info("=" * 50)
    logger.info("✅ Ingestion complete!")
    logger.info(f"   Documents loaded  : {len(documents)}")
    logger.info(f"   Chunks created    : {len(chunks)}")
    logger.info(f"   Workspace ID      : default")
    logger.info(f"   Qdrant storage    : {qdrant_path}")
    logger.info(f"   BM25 index        : {bm25_path}")
    logger.info("=" * 50)
    logger.info("\nStart the v2 server:")
    logger.info("  uvicorn src.v2.api.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
