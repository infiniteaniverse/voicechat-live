# Embedding wrapper — loads all-MiniLM-L6-v2 once and exposes embed(text) -> vector.
# Singleton pattern: model loaded once at startup, reused across all memory ops.
# Implementation: Part 3
from __future__ import annotations

import logging
import os
import time

# huggingface_hub (used internally by SentenceTransformer) makes a live HEAD
# request to check for a PEFT adapter config on every load, even when the
# model is already fully downloaded and cached locally — and raises instead
# of falling back to the cache if that request fails. This violates this
# project's "no runtime network calls" offline requirement (CLAUDE.md) and
# was caught when a real network outage crashed pipeline startup entirely
# (see docs/tradeoffs.md decision 28). setdefault so an explicit override
# (e.g. to force a fresh download) still works if ever needed.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper around SentenceTransformer. Model loaded once at init and reused."""

    def __init__(self, config: dict) -> None:
        model_name: str = config["memory"]["long_term"]["embedding_model"]

        logger.info("Loading embedding model '%s'...", model_name)
        t0 = time.perf_counter()
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded in %.2fs", time.perf_counter() - t0)

        self.dimension: int = self._model.get_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed one string. Returns a normalized float32 vector (shape: (dimension,))."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.astype(np.float32)
