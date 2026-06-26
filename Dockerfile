# Use official Python runtime as a parent image
FROM python:3.12-slim

# Install system dependencies (needed for PDF/DOCX processing and python libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and static assets
COPY src/ /app/src/
COPY static/ /app/static/
COPY scripts/ /app/scripts/

# Create data directory for persistent mounting
RUN mkdir -p /app/data

# Expose port
EXPOSE 8000

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    WORKSPACE_STORE_PATH=/app/data/workspaces.json \
    QDRANT_PATH=/app/data/qdrant_storage \
    BM25_INDEX_PATH=/app/data/bm25_index.pkl \
    COLLECTION_NAME=langchain_docs

# Run the server
CMD ["uvicorn", "src.v2.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
