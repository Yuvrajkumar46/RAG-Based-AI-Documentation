"""
Embedding service using sentence-transformers (local, no API key).

Model: all-MiniLM-L6-v2
  - 384-dim embeddings
  - 80 MB download, fast CPU inference (~5 ms/chunk)
  - MTEB score makes it competitive with much larger models for RAG
  - 512 token limit → our 400-token chunk size fits comfortably

The model is loaded once at startup and reused (singleton pattern).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import get_settings
from utils.logger import get_logger

logger = get_logger("rag.embedding")

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        cfg = get_settings()
        logger.info("Loading embedding model: %s", cfg.embedding_model)
        _model = SentenceTransformer(cfg.embedding_model)
        logger.info("Embedding model loaded (dim=%d)", _model.get_sentence_embedding_dimension())
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts.  Returns float32 array of shape (N, dim).
    Batch-processed for efficiency.
    """
    model = get_embedding_model()
    # convert_to_numpy=True returns a contiguous float32 ndarray
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,   # FAISSStore normalises itself
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (dim,)."""
    return embed_texts([query])[0]
