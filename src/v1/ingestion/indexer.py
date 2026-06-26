"""
indexer.py — Embedding, Vector Storage, and BM25 Index Construction

This module:
1. Generates dense embeddings using a local SentenceTransformer model
2. Stores vectors in Qdrant (local file mode — no Docker needed)
3. Builds a BM25 sparse index from chunk texts
4. Saves the BM25 index to disk for later retrieval

Both indexes are keyed by chunk_id so the retrieval layer can join results.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (can be overridden via .env)
# ---------------------------------------------------------------------------
DEFAULT_QDRANT_PATH = "./data/qdrant_storage"
DEFAULT_BM25_PATH = "./data/bm25_index.pkl"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "langchain_docs"
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    """Load (and cache) the SentenceTransformer embedding model."""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("✓ Embedding model loaded")
    return model


def _embed_chunks(
    chunks: list[Chunk],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Batch-embed all chunk texts.

    Returns:
        numpy array of shape (num_chunks, embedding_dim)
    """
    texts = [chunk.text for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Normalize for cosine similarity
        convert_to_numpy=True,
    )
    logger.info(f"✓ Embeddings shape: {embeddings.shape}")
    return embeddings


def build_qdrant_index(
    chunks: list[Chunk],
    embeddings: np.ndarray,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    recreate: bool = False,
) -> QdrantClient:
    """
    Store chunk vectors in a local Qdrant collection.

    Uses Qdrant's local file-based storage (no server/Docker needed).
    The data persists between runs in the specified path.

    Args:
        chunks: List of Chunk objects.
        embeddings: Dense embeddings (num_chunks × embedding_dim).
        qdrant_path: Local directory for Qdrant's SQLite storage.
        collection_name: Name of the Qdrant collection.
        recreate: If True, drop and recreate the collection.

    Returns:
        Initialized QdrantClient instance.
    """
    Path(qdrant_path).mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=qdrant_path)

    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            logger.info(f"Dropping existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            logger.info(
                f"Collection '{collection_name}' already exists. "
                "Skipping Qdrant indexing. Use recreate=True to rebuild."
            )
            return client

    logger.info(f"Creating Qdrant collection: {collection_name}")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )

    # Upload in batches
    BATCH_SIZE = 128
    total = len(chunks)
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Uploading to Qdrant"):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_vectors = embeddings[i : i + BATCH_SIZE]

        points = [
            PointStruct(
                id=idx,                       # Qdrant point ID (int)
                vector=vec.tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source_url": chunk.source_url,
                    "source_title": chunk.source_title,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )
            for idx, (chunk, vec) in enumerate(
                zip(batch_chunks, batch_vectors), start=i
            )
        ]

        client.upsert(collection_name=collection_name, points=points)

    logger.info(f"✓ Indexed {total} vectors in Qdrant collection '{collection_name}'")
    return client


def build_bm25_index(
    chunks: list[Chunk],
    bm25_path: str = DEFAULT_BM25_PATH,
) -> BM25Okapi:
    """
    Build a BM25Okapi sparse index from chunk texts and save to disk.

    The index stores tokenized text for fast keyword matching.

    Args:
        chunks: List of Chunk objects.
        bm25_path: Path to save the pickled index.

    Returns:
        BM25Okapi instance.
    """
    Path(bm25_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building BM25 index over {len(chunks)} chunks...")
    tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Serialize: save both the index and chunk references needed for retrieval
    index_data = {
        "bm25": bm25,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source_url": c.source_url,
                "source_title": c.source_title,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in chunks
        ],
    }

    with open(bm25_path, "wb") as f:
        pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"✓ BM25 index saved to: {bm25_path}")
    return bm25


def run_ingestion(
    chunks: list[Chunk],
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    qdrant_path: str = DEFAULT_QDRANT_PATH,
    bm25_path: str = DEFAULT_BM25_PATH,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    recreate: bool = False,
) -> tuple[QdrantClient, BM25Okapi]:
    """
    Full ingestion pipeline: embed chunks → build Qdrant index → build BM25 index.

    Args:
        chunks: Pre-chunked documents.
        embedding_model_name: HuggingFace model name for embeddings.
        qdrant_path: Local path for Qdrant storage.
        bm25_path: Path to save the BM25 index pickle.
        collection_name: Qdrant collection name.
        recreate: If True, rebuild all indexes from scratch.

    Returns:
        Tuple of (QdrantClient, BM25Okapi)
    """
    model = _get_embedding_model(embedding_model_name)
    embeddings = _embed_chunks(chunks, model)

    qdrant_client = build_qdrant_index(
        chunks, embeddings,
        qdrant_path=qdrant_path,
        collection_name=collection_name,
        recreate=recreate,
    )
    bm25_index = build_bm25_index(chunks, bm25_path=bm25_path)

    return qdrant_client, bm25_index
