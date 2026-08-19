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

import pickle
import time
from dataclasses import dataclass
from typing import Any, Sequence

import faiss
import numpy as np

from app.config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    HNSW_EF_SEARCH,
    INDEX_PATH,
    METADATA_PATH,
    QUERY_INSTRUCTION,
)
from app.exceptions import IndexNotLoadedError


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
        self._chunks: list[dict[str, Any]] = []
        self._meta: dict[str, Any] = {}
        self._embedder: Any = None

    def load(self) -> None:
        """Load artifacts from disk. Called once during startup."""
        if not INDEX_PATH.exists() or not METADATA_PATH.exists():
            raise IndexNotLoadedError(
                "Vector index artifacts are missing.",
                detail=f"expected {INDEX_PATH.name} and {METADATA_PATH.name}; "
                "run `python -m app.indexer` to build them",
            )

        from fastembed import TextEmbedding

        self._index = faiss.read_index(str(INDEX_PATH))
        self._index.hnsw.efSearch = HNSW_EF_SEARCH

        with METADATA_PATH.open("rb") as handle:
            payload = pickle.load(handle)
        self._chunks = payload["chunks"]
        self._meta = payload.get("meta", {})

        self._embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
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
            list(self._embedder.embed([QUERY_INSTRUCTION + text])), dtype=np.float32
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
            chunk = self._chunks[position]
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
