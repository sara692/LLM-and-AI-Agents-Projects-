"""
embeddings.py – Embedding generation using sentence-transformers.
Isolated here so the model is loaded once and reused.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_embedder = None  # lazy singleton


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model loaded: %s", EMBEDDING_MODEL)
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e)
            _embedder = None
    return _embedder


def embed(text: str) -> Optional[list[float]]:
    """
    Generate an embedding vector for text.
    Returns None (with a log warning) if embedding fails — never crashes.
    """
    try:
        model = _get_embedder()
        if model is None:
            return None
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()
    except Exception as e:
        logger.warning("Embedding generation failed (fallback to None): %s", e)
        return None