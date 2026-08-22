"""FAISS retrieval over the prebuilt English vector space.

The index and the embedding model are loaded once at startup and held as a
singleton. Loading either per request would dominate the latency budget many
times over -- the ONNX session alone takes longer to initialize than the entire
retrieval path takes to run.

Embedding is CPU-bound and blocking, so callers run it off the event loop. FAISS
search is left inline: at ~3ms the cost of a thread hop is a meaningful fraction
of the work itself.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Sequence

import faiss
import numpy as np

from app.config import (
    CHUNK_DB_PATH,
    DEFAULT_TOP_K,
    EMBED_THREADS,
    EMBEDDING_MODEL,
    HNSW_EF_SEARCH,
    INDEX_PATH,
    METADATA_PATH,
    QUERY_INSTRUCTION,
)
from app.exceptions import IndexNotLoadedError


# Trailing sentence punctuation, in the scripts this pipeline accepts. The
# Devanagari danda is included because Sarvam ends Hindi transcriptions with it.
_TRAILING_PUNCTUATION = " \t\n.?!,;:।॥"


def canonical_query(text: str) -> str:
    """One vector for one question, however it was asked.

    Speech recognition punctuates -- saaras returns "Tell me about Obama." --
    and people typing usually do not. bge-small embeds those as different
    sentences, and the gap is large enough to matter: measured at 0.7002 bare
    against 0.6795 with a full stop, which straddles the 0.68 grounding
    threshold. The same question was answered when typed and refused when
    spoken, which looked like a bug in the voice pipeline and was really a
    difference in the string it produced.

    Only trailing punctuation is removed. Punctuation inside a query carries
    meaning -- "C++" and "C" are not the same search -- so nothing else is
    touched, and this never runs on the passage side, which was indexed as
    written.
    """
    return text.strip().rstrip(_TRAILING_PUNCTUATION) or text.strip()


@dataclass
class Hit:
    """One retrieved chunk with its cosine score."""

    chunk_id: str
    parent_id: str
    query_id: int
    score: float
    text_english: str
    parent_english: str
    translations: dict[str, str]
    is_selected: bool

    def display_text(self, language: str) -> str:
        """The passage in the requested language, falling back to English."""
        if language == "en":
            return self.parent_english
        return self.translations.get(language) or self.parent_english


class RetrievalEngine:
    """Holds the FAISS index, chunk payloads and the embedding model."""

    def __init__(self) -> None:
        self._index: faiss.Index | None = None
        # Populated only on the pickle fallback path; with a SQLite store the
        # chunks stay on disk and this stays empty, which is the entire point.
        self._chunks: list[dict[str, Any]] = []
        self._db: sqlite3.Connection | None = None
        self._meta: dict[str, Any] = {}
        self._embedder: Any = None

    def load(self) -> None:
        """Load artifacts from disk. Called once during startup."""
        has_store = CHUNK_DB_PATH.exists() or METADATA_PATH.exists()
        if not INDEX_PATH.exists() or not has_store:
            raise IndexNotLoadedError(
                "Vector index artifacts are missing.",
                detail=f"expected {INDEX_PATH.name} and one of "
                f"{CHUNK_DB_PATH.name} / {METADATA_PATH.name}; "
                "run `python -m app.indexer` to build them",
            )

        from fastembed import TextEmbedding

        self._index = faiss.read_index(str(INDEX_PATH))
        self._index.hnsw.efSearch = HNSW_EF_SEARCH

        if CHUNK_DB_PATH.exists():
            # check_same_thread=False because retrieval runs in a worker
            # thread; every access here is a read, so there is nothing to
            # serialise against.
            self._db = sqlite3.connect(str(CHUNK_DB_PATH), check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            row = self._db.execute("SELECT json FROM meta WHERE id = 1").fetchone()
            self._meta = json.loads(row["json"]) if row else {}
        else:
            # An artifact set built before the SQLite store existed. Costs the
            # full 516MB, but runs rather than refusing to start.
            with METADATA_PATH.open("rb") as handle:
                payload = pickle.load(handle)
            self._chunks = payload["chunks"]
            self._meta = payload.get("meta", {})

        self._embedder = TextEmbedding(model_name=EMBEDDING_MODEL, threads=EMBED_THREADS)
        # First inference initializes the ONNX session. Doing it now keeps that
        # one-off cost out of the first user's measured latency.
        list(self._embedder.embed(["warmup"]))

    @property
    def ready(self) -> bool:
        return self._index is not None and self._embedder is not None

    @property
    def size(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def _require_ready(self) -> None:
        if not self.ready:
            raise IndexNotLoadedError(
                "Retrieval engine is not loaded.",
                detail="the index failed to load at startup",
            )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query. Blocking; call via a worker thread.

        bge-small-en-v1.5 was trained with an instruction prefix on the query
        side only. Passages were indexed bare, so the prefix belongs here and
        nowhere else; omitting it costs real recall.
        """
        self._require_ready()
        vector = np.asarray(
            list(self._embedder.embed([QUERY_INSTRUCTION + canonical_query(text)])),
            dtype=np.float32,
        )
        faiss.normalize_L2(vector)
        return vector

    def search(self, vector: np.ndarray, top_k: int = DEFAULT_TOP_K) -> list[Hit]:
        """Search the index. Scores are cosine similarities on [-1, 1]."""
        self._require_ready()
        assert self._index is not None

        scores, indices = self._index.search(vector, top_k)
        hits: list[Hit] = []
        for score, position in zip(scores[0], indices[0]):
            if position < 0:  # FAISS pads short result sets with -1
                continue
            chunk = self._chunk_at(int(position))
            if chunk is None:
                continue
            hits.append(
                Hit(
                    chunk_id=chunk["chunk_id"],
                    parent_id=chunk["parent_id"],
                    query_id=chunk["query_id"],
                    score=float(score),
                    text_english=chunk["text_english"],
                    parent_english=chunk["parent_english"],
                    translations=chunk.get("translations") or {},
                    is_selected=bool(chunk.get("is_selected")),
                )
            )
        return hits

    def _chunk_at(self, position: int) -> dict[str, Any] | None:
        """One chunk by its FAISS position, from whichever store is loaded."""
        if self._db is None:
            return self._chunks[position] if 0 <= position < len(self._chunks) else None

        row = self._db.execute(
            "SELECT chunk_id, parent_id, query_id, is_selected, text_english, "
            "parent_english, hi, ta FROM chunk WHERE pos = ?",
            (position,),
        ).fetchone()
        if row is None:
            return None
        # Rebuilt into the shape the pickle produced, so callers cannot tell
        # which store answered.
        translations = {k: row[k] for k in ("hi", "ta") if row[k]}
        return {
            "chunk_id": row["chunk_id"],
            "parent_id": row["parent_id"],
            "query_id": row["query_id"],
            "text_english": row["text_english"],
            "parent_english": row["parent_english"],
            "translations": translations,
            "is_selected": bool(row["is_selected"]),
        }

    def build_contexts(self, hits: Sequence[Hit]) -> list[str]:
        """Context passages for the prompt, de-duplicated by parent.

        Sibling chunks share a parent, so sending both would spend prompt
        budget repeating the same passage.
        """
        contexts: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            if hit.parent_id in seen:
                continue
            seen.add(hit.parent_id)
            contexts.append(hit.parent_english)
        return contexts


# Module-level singleton, loaded during application startup.
engine = RetrievalEngine()
