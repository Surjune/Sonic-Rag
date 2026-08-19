"""Shared configuration and tuned constants for the Sonic-RAG pipeline.

Every value that affects retrieval quality or the latency budget lives here so
the indexer, the guardrails and the API cannot drift apart. A threshold changed
in one place and not the other would silently break grounding.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- paths ------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND_DIR / "data" / "msmarco-xi" / "validation"
ARTIFACT_DIR = BACKEND_DIR / "artifacts"

INDEX_PATH = ARTIFACT_DIR / "vector_index.faiss"
METADATA_PATH = ARTIFACT_DIR / "metadata.pkl"

# Language -> local parquet file. English is carried inside every file as the
# aligned `English_passages` column, so it needs no file of its own.
LANG_FILES: dict[str, str] = {
    "hi": "hinval.parquet",
    "ta": "tamval.parquet",
}
SUPPORTED_LANGS: tuple[str, ...] = ("hi", "ta", "en")

# --- embedding --------------------------------------------------------------

# Quantized ONNX, runs locally on CPU: no network hop, ~10-15ms per query.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# bge-small-en-v1.5 was trained with an instruction prefix on the query side
# only. Passages are embedded bare; omitting this on queries costs real recall.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Small batches on purpose: the build target is a 2-core / low-RAM machine,
# where large batches raise peak memory enough to stall the pass entirely.
EMBED_BATCH_SIZE = 32

# --- FAISS HNSW -------------------------------------------------------------
# Vectors are L2-normalized, so inner product == cosine similarity.

HNSW_M = 32  # neighbours per node; 32 is the accuracy/memory knee for <1M vectors
HNSW_EF_CONSTRUCTION = 200  # build-time depth; higher = better graph, slower build
HNSW_EF_SEARCH = 64  # query-time depth; the main recall-vs-latency dial

DEFAULT_TOP_K = 5

# --- guardrails -------------------------------------------------------------

# Below this cosine score the retrieved context is treated as ungrounded and the
# request is refused without calling the LLM. Overridable per deployment.
SIMILARITY_THRESHOLD = float(os.getenv("EMBEDDING_SIMILARITY_THRESHOLD", "0.38"))

# --- latency budget ---------------------------------------------------------

LLM_TIMEOUT_MS = int(os.getenv("LLM_TIMEOUT_MS", "150"))
