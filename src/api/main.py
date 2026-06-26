"""
main.py — FastAPI Server

Serves the RAG pipeline as a production-style REST API.
Also serves the frontend UI directly (no separate web server needed).

Endpoints:
  GET  /           — Serves the premium chat UI
  GET  /health     — Health check with index status
  POST /ask        — Main RAG Q&A endpoint
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import AskRequest, AskResponse, HealthResponse, RetrievalStats, SourceReference
from src.generation.chain import RAGPipeline

# Load environment variables from .env
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global pipeline instance (loaded once at startup)
# ---------------------------------------------------------------------------
pipeline: RAGPipeline | None = None
pipeline_ready: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline on startup, clean up on shutdown."""
    global pipeline, pipeline_ready
    logger.info("=== Starting RAG Pipeline ===")

    qdrant_path = os.getenv("QDRANT_PATH", "./data/qdrant_storage")
    bm25_path = os.getenv("BM25_INDEX_PATH", "./data/bm25_index.pkl")
    collection_name = os.getenv("COLLECTION_NAME", "langchain_docs")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Check indexes exist before loading the pipeline
    if not Path(bm25_path).exists():
        logger.warning(
            "⚠ BM25 index not found! Run: python scripts/ingest.py first."
        )
        pipeline_ready = False
    else:
        try:
            pipeline = RAGPipeline(
                qdrant_path=qdrant_path,
                bm25_path=bm25_path,
                collection_name=collection_name,
                embedding_model=embedding_model,
                reranker_model=reranker_model,
            )
            pipeline_ready = True
            logger.info("✓ RAG Pipeline initialized and ready")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            pipeline_ready = False

    yield

    logger.info("=== Shutting down RAG Pipeline ===")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Production RAG Application",
    description=(
        "A domain-specific Q&A system over LangChain documentation, "
        "featuring hybrid BM25+vector retrieval, cross-encoder reranking, "
        "and citation enforcement."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the frontend)
STATIC_DIR = Path(__file__).parent.parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the premium chat UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"message": "Frontend not found. See /docs for the API."}
    return FileResponse(str(index_path))


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint. Returns pipeline readiness status."""
    return HealthResponse(
        status="ok",
        message=(
            "RAG Pipeline is ready." if pipeline_ready
            else "⚠ Pipeline not ready — run ingestion first: python scripts/ingest.py"
        ),
        indexes_loaded=pipeline_ready,
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask(request: AskRequest):
    """
    Ask a question about LangChain documentation.

    Runs the full hybrid retrieval → reranking → generation → citation
    validation pipeline and returns a cited answer with source references.
    """
    if not pipeline_ready or pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Pipeline not ready. Please run ingestion first: "
                "python scripts/ingest.py"
            ),
        )

    try:
        result = pipeline.ask(request.question)

        sources = [SourceReference(**s) for s in result["sources"]]
        stats = RetrievalStats(**result["retrieval_stats"])

        return AskResponse(
            answer=result["answer"],
            sources=sources,
            retrieval_stats=stats,
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
