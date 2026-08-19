"""Build the in-memory FAISS HNSW index from local MSMARCO-XI parquet files.

Pipeline: parquet rows -> hierarchical chunks -> deduplicate -> embed on local
ONNX CPU -> L2-normalize -> HNSW index -> artifacts on disk.

Why a single English vector space
---------------------------------
`bge-small-en-v1.5` has an English wordpiece vocabulary, so Devanagari and Tamil
tokenize almost entirely to `[UNK]` and embed meaninglessly. MSMARCO-XI is a
*parallel* corpus: every Hindi and Tamil row is row-aligned to the same English
passage by `query_id`. So we embed the English child chunks exactly once and
attach the Hindi and Tamil passages to each chunk as display payloads.

The user asks and reads in their own language; only the vector space is English.
Indexing each language separately would embed the same English text three times
over, tripling both index size and search cost for no recall benefit.

The artifacts are built ONCE and shipped. Never build at container startup: the
embedding pass takes minutes and would make every cold start a timeout.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import faiss
import numpy as np
from fastembed import TextEmbedding

from app.chunking import NATIVE_PASSAGE_KEY, chunk_row, normalize
from app.config import (
    ARTIFACT_DIR,
    DATA_DIR,
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    INDEX_PATH,
    LANG_FILES,
    METADATA_PATH,
)


@dataclass(frozen=True)
class IndexedChunk:
    """The payload stored alongside each vector and returned by retrieval."""

    chunk_id: str
    parent_id: str
    query_id: int
    passage_index: int
    text_english: str  # the child chunk that was embedded
    parent_english: str  # full English passage, for LLM context
    translations: dict[str, str] = field(default_factory=dict)  # lang -> native passage
    is_selected: bool = False


def _read_rows(path: Path, limit: int) -> Iterator[dict[str, Any]]:
    """Stream rows off a parquet file without loading the whole table."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    seen = 0
    for batch in parquet_file.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            yield row
            seen += 1
            if seen >= limit:
                return


def _native_passages(row: dict[str, Any]) -> list[str]:
    return list((row.get("passages") or {}).get(NATIVE_PASSAGE_KEY) or [])


def load_tamil_passages(rows: int) -> dict[int, list[str]]:
    """Map query_id -> Tamil passages, for attaching to the English chunks."""
    path = DATA_DIR / LANG_FILES["ta"]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run test_dataset_connection.py first")
    return {int(row["query_id"]): _native_passages(row) for row in _read_rows(path, rows)}


def collect_chunks(rows: int) -> tuple[list[str], list[IndexedChunk]]:
    """Chunk English passages once and attach the aligned Indic translations.

    Duplicate passages recur across MSMARCO queries, so deduplicating on the
    embedded English text is a large, free saving in build time and index size.
    """
    hindi_path = DATA_DIR / LANG_FILES["hi"]
    if not hindi_path.exists():
        raise FileNotFoundError(f"missing {hindi_path}; run test_dataset_connection.py first")

    tamil = load_tamil_passages(rows)
    print(f"  loaded Tamil passages for {len(tamil):,} queries")

    texts: list[str] = []
    payloads: list[IndexedChunk] = []
    seen: set[str] = set()
    duplicates = 0
    missing_tamil = 0

    for row in _read_rows(hindi_path, rows):
        query_id = int(row["query_id"])
        hindi = _native_passages(row)
        tamil_passages = tamil.get(query_id)
        if tamil_passages is None:
            missing_tamil += 1
            tamil_passages = []

        # The English side is identical in every language file, so chunking the
        # Hindi row with lang="en" yields the shared English passages.
        parents, children = chunk_row(row, "en")
        parent_text = {parent.parent_id: parent.text for parent in parents}

        for chunk in children:
            embed_text = chunk.embed_text
            if not embed_text.strip():
                continue

            fingerprint = hashlib.blake2b(embed_text.encode(), digest_size=16).hexdigest()
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)

            index = chunk.passage_index
            translations: dict[str, str] = {}
            if index < len(hindi):
                translations["hi"] = normalize(hindi[index])
            if index < len(tamil_passages):
                translations["ta"] = normalize(tamil_passages[index])

            texts.append(embed_text)
            payloads.append(
                IndexedChunk(
                    chunk_id=chunk.chunk_id,
                    parent_id=chunk.parent_id,
                    query_id=query_id,
                    passage_index=index,
                    text_english=chunk.text,
                    parent_english=parent_text.get(chunk.parent_id, chunk.text),
                    translations=translations,
                    is_selected=chunk.is_selected,
                )
            )

    print(f"  {len(payloads):,} unique English chunks (deduplicated {duplicates:,} repeats)")
    if missing_tamil:
        print(f"  note: {missing_tamil:,} queries had no aligned Tamil row")
    return texts, payloads


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Embed on local ONNX CPU and L2-normalize for cosine-via-inner-product.

    Vectors are written into one preallocated array in slices rather than
    accumulated in a Python list. On a small-RAM machine the list-of-arrays form
    fragments the heap badly enough to stall the whole pass, and going dark for
    minutes with no output makes a stall indistinguishable from slow progress.
    """
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    total = len(texts)
    vectors = np.empty((total, EMBEDDING_DIM), dtype=np.float32)

    started = time.perf_counter()
    filled = 0
    for offset in range(0, total, EMBED_BATCH_SIZE):
        batch = list(texts[offset : offset + EMBED_BATCH_SIZE])
        for vector in model.embed(batch, batch_size=EMBED_BATCH_SIZE):
            vectors[filled] = vector
            filled += 1
        elapsed_so_far = time.perf_counter() - started
        rate_so_far = filled / elapsed_so_far if elapsed_so_far else 0.0
        remaining = (total - filled) / rate_so_far if rate_so_far else 0.0
        print(
            f"    {filled:,}/{total:,} ({100 * filled / total:.0f}%) "
            f"{rate_so_far:.0f}/s  eta {remaining / 60:.1f} min",
            flush=True,
        )
    elapsed = time.perf_counter() - started

    if filled != total:
        raise ValueError(f"embedded {filled} vectors for {total} texts")
    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"expected dim {EMBEDDING_DIM}, model returned {vectors.shape[1]}")

    # Normalizing turns FAISS inner product into exact cosine similarity, which
    # is what the grounding threshold is calibrated against.
    faiss.normalize_L2(vectors)

    rate = len(texts) / elapsed if elapsed else 0.0
    print(f"  embedded {len(texts):,} chunks in {elapsed:.1f}s ({rate:.0f}/s)")
    return vectors


def build_index(vectors: np.ndarray) -> faiss.IndexHNSWFlat:
    """Build an HNSW graph over normalized vectors (O(log N) search)."""
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    started = time.perf_counter()
    index.add(vectors)
    print(f"  built HNSW over {index.ntotal:,} vectors in {time.perf_counter() - started:.1f}s")
    return index


def save_artifacts(
    index: faiss.Index, payloads: Sequence[IndexedChunk], meta: dict[str, Any]
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with METADATA_PATH.open("wb") as handle:
        pickle.dump(
            {"chunks": [asdict(p) for p in payloads], "meta": meta},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    index_mb = INDEX_PATH.stat().st_size / 1e6
    meta_mb = METADATA_PATH.stat().st_size / 1e6
    print(f"  wrote {INDEX_PATH.name} ({index_mb:.1f} MB) + {METADATA_PATH.name} ({meta_mb:.1f} MB)")


def _smoke_test(index: faiss.Index, payloads: Sequence[IndexedChunk]) -> None:
    """Search the finished index once to prove it is queryable and timed."""
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    probe = "what is a corporation?"
    list(model.embed(["warmup"]))  # exclude ONNX session init from the timing

    started = time.perf_counter()
    query_vector = np.asarray(list(model.embed([probe])), dtype=np.float32)
    faiss.normalize_L2(query_vector)
    embed_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    scores, ids = index.search(query_vector, 3)
    search_ms = (time.perf_counter() - started) * 1000

    print(f"\n  probe: {probe!r}")
    print(f"  embed {embed_ms:.1f}ms | faiss {search_ms:.2f}ms")
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        hit = payloads[idx]
        print(f"    {score:.4f}  en: {hit.text_english[:70]}")
        if "hi" in hit.translations:
            print(f"            hi: {hit.translations['hi'][:60]}")
        if "ta" in hit.translations:
            print(f"            ta: {hit.translations['ta'][:60]}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Build the Sonic-RAG FAISS index.")
    parser.add_argument("--rows", type=int, default=250, help="rows to ingest")
    args = parser.parse_args()

    print(f"model : {EMBEDDING_MODEL}")
    print(f"rows  : {args.rows:,}  (English vector space, hi/ta attached)\n")

    print("[1/4] chunking")
    texts, payloads = collect_chunks(args.rows)
    if not texts:
        print("[FAIL] no chunks produced")
        return 1

    print("\n[2/4] embedding")
    vectors = embed_texts(texts)

    print("\n[3/4] indexing")
    index = build_index(vectors)

    print("\n[4/4] saving")
    save_artifacts(
        index,
        payloads,
        {
            "model": EMBEDDING_MODEL,
            "dim": EMBEDDING_DIM,
            "vector_space": "en",
            "display_langs": ["en", "hi", "ta"],
            "rows": args.rows,
            "count": len(payloads),
            "hnsw_m": HNSW_M,
            "ef_search": HNSW_EF_SEARCH,
        },
    )

    _smoke_test(index, payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
