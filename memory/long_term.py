# Long-term memory — FAISS vector store + SQLite for raw fact text.
# Operations: store_fact(text), retrieve_similar(query, top_k), delete_fact(id)
# Persists across sessions in data/memory/.
# Implementation: Part 3
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import faiss
import numpy as np

from memory.embeddings import Embedder

logger = logging.getLogger(__name__)


class LongTermMemory:
    """FAISS (vectors) + SQLite (fact text) fact store, persisted to disk.

    FAISS holds only vectors + integer ids (an IndexIDMap over a flat cosine-
    similarity index — exact search, fine at this scale). SQLite holds the id
    -> fact text mapping. The two are kept in sync: every store_fact()/
    delete_fact() writes to both, and the FAISS index is re-saved to disk
    after every write so a crash doesn't lose more than the last write.
    """

    # Cosine similarity above which a new fact is considered a duplicate of an
    # existing one and skipped rather than stored again.
    _DEDUP_THRESHOLD = 0.92

    def __init__(self, config: dict, embedder: Embedder) -> None:
        lt_cfg = config["memory"]["long_term"]
        self._embedder = embedder
        self._top_k: int = lt_cfg["top_k"]
        self._similarity_threshold: float = lt_cfg["similarity_threshold"]
        self._index_path = Path(lt_cfg["store_path"])
        self._db_path = Path(lt_cfg["facts_db_path"])

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._index = self._load_or_create_index()

        # check_same_thread=False: nodes.py calls into this class via asyncio.to_thread()
        # (to keep blocking FAISS/SQLite calls off the event loop), which runs each call
        # on a worker thread, not the thread that constructed this connection. Access is
        # effectively serialized in this project's usage pattern (one turn at a time).
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS facts ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "text TEXT NOT NULL, "
            "created_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        self._conn.commit()

    # ── Public API ─────────────────────────────────────────────────────────

    def store_fact(self, text: str) -> int:
        """Embed + persist one fact. Returns its id.

        Skips storing (returns the existing id instead) if a near-duplicate
        fact is already in the store — see _DEDUP_THRESHOLD.
        """
        vector = self._embedder.embed(text).reshape(1, -1)

        duplicate_id = self._find_duplicate(vector)
        if duplicate_id is not None:
            logger.info("Duplicate of fact [id=%d] — not storing: '%s'", duplicate_id, text)
            return duplicate_id

        cursor = self._conn.execute("INSERT INTO facts (text) VALUES (?)", (text,))
        self._conn.commit()
        fact_id = cursor.lastrowid

        self._index.add_with_ids(vector, np.array([fact_id], dtype=np.int64))
        self._save_index()

        logger.info("Stored fact [id=%d]: '%s'", fact_id, text)
        return fact_id

    def retrieve_similar(self, query: str, top_k: int | None = None) -> list[str]:
        """Return up to top_k fact strings above the similarity threshold, best first."""
        if self._index.ntotal == 0:
            return []

        vector = self._embedder.embed(query).reshape(1, -1)
        k = min(top_k or self._top_k, self._index.ntotal)
        scores, ids = self._index.search(vector, k)

        results: list[str] = []
        for score, fact_id in zip(scores[0], ids[0]):
            if fact_id == -1 or score < self._similarity_threshold:
                continue
            text = self._fetch_text(int(fact_id))
            if text is not None:
                results.append(text)
        return results

    def delete_fact(self, fact_id: int) -> None:
        """Remove a fact from both FAISS and SQLite."""
        self._index.remove_ids(np.array([fact_id], dtype=np.int64))
        self._save_index()
        self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        self._conn.commit()
        logger.info("Deleted fact [id=%d]", fact_id)

    # ── Internal ───────────────────────────────────────────────────────────

    def _find_duplicate(self, vector: np.ndarray) -> int | None:
        if self._index.ntotal == 0:
            return None
        scores, ids = self._index.search(vector, 1)
        if ids[0][0] != -1 and scores[0][0] >= self._DEDUP_THRESHOLD:
            return int(ids[0][0])
        return None

    def _load_or_create_index(self) -> faiss.IndexIDMap:
        if self._index_path.exists():
            logger.info("Loading FAISS index from %s", self._index_path)
            return faiss.read_index(str(self._index_path))
        logger.info("No existing FAISS index at %s — creating new one", self._index_path)
        return faiss.IndexIDMap(faiss.IndexFlatIP(self._embedder.dimension))

    def _save_index(self) -> None:
        faiss.write_index(self._index, str(self._index_path))

    def _fetch_text(self, fact_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT text FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        return row[0] if row else None
