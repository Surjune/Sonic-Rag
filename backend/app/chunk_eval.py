"""Retrieval-quality evaluation across chunking strategies.

Ground truth comes from the corpus itself: MSMARCO-XI marks the passage that
answers each query with `is_selected`, so a retrieval is correct when the chunk
it returns belongs to that passage. No labelling by hand, and no scoring the
system against its own opinion.

Reported per strategy:

    Recall@1/@3/@5   did a gold passage appear in the top K
    MRR@5            1/rank of the first gold passage, averaged
    chunks           index size, since overlap buys recall by duplicating text
    bytes            vector memory, which is what a free host actually limits
    search_ms        median FAISS latency at that index size

Recall and index size are reported together on purpose. A strategy that wins on
recall by tripling the index has not won for free, and on a $0 host the memory
is the binding constraint.

Run offline: embedding several thousand chunks four times over is minutes of
CPU, far too slow to serve from a request. Results are cached to
artifacts/chunk_comparison.json and the API serves that.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Sequence

import faiss
import numpy as np

from app.chunkers import STRATEGY_ORDER, BaseChunker, build_chunker, get_chunker
from app.chunking import MAX_CHILD_CHARS
from app.config import (
    ARTIFACT_DIR,
    DATA_DIR,
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_M,
    LANG_FILES,
    QUERY_INSTRUCTION,
)

COMPARISON_PATH = ARTIFACT_DIR / "chunk_comparison.json"

RECALL_K = (1, 3, 5)
MRR_K = 5

# Bytes per stored vector, before HNSW graph overhead.
BYTES_PER_VECTOR = EMBEDDING_DIM * 4


@dataclass
class StrategyResult:
    name: str
    description: str
    chunks: int
    vector_bytes: int
    mean_chunk_chars: float
    build_ms: float
    search_ms_p50: float
    recall: dict[str, float] = field(default_factory=dict)
    mrr5: float = 0.0


@dataclass
class EvalCase:
    """One query with the passages that should be retrieved for it."""

    query: str
    passages: list[str]
    gold_indices: set[int]


def load_cases(limit: int) -> list[EvalCase]:
    """Load queries that have at least one gold passage.

    Queries with no `is_selected` passage are skipped: they cannot distinguish
    a good strategy from a bad one, so including them would only dilute every
    score equally.
    """
    import pyarrow.parquet as pq

    path = DATA_DIR / LANG_FILES["hi"]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run test_dataset_connection.py first")

    cases: list[EvalCase] = []
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            passages = (row.get("passages") or {}).get("English_passages") or []
            selected = (row.get("passages") or {}).get("is_selected") or []
            gold = {index for index, flag in enumerate(selected) if flag}
            query = (row.get("Eng_Query") or "").strip()
            if not query or not passages or not gold:
                continue
            cases.append(EvalCase(query=query, passages=list(passages), gold_indices=gold))
            if len(cases) >= limit:
                return cases
    return cases


def _embed(model: Any, texts: Sequence[str]) -> np.ndarray:
    vectors = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
    filled = 0
    for offset in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = list(texts[offset : offset + EMBED_BATCH_SIZE])
        for vector in model.embed(batch, batch_size=EMBED_BATCH_SIZE):
            vectors[filled] = vector
            filled += 1
    faiss.normalize_L2(vectors)
    return vectors


def evaluate_strategy(
    chunker: BaseChunker, cases: Sequence[EvalCase], model: Any, progress: bool = True
) -> StrategyResult:
    """Build an index with this strategy and score retrieval against the gold labels."""
    texts: list[str] = []
    # Which (case, passage) each vector came from, so a hit can be checked.
    origins: list[tuple[int, int]] = []
    lengths: list[int] = []

    for case_index, case in enumerate(cases):
        for passage_index, passage in enumerate(case.passages):
            for chunk in chunker.chunk(passage, lang="en"):
                texts.append(chunk.embed_text)
                origins.append((case_index, passage_index))
                lengths.append(chunk.length)

    if not texts:
        raise ValueError(f"strategy {chunker.name} produced no chunks")

    if progress:
        print(f"  {chunker.name}: embedding {len(texts):,} chunks…", flush=True)

    started = time.perf_counter()
    vectors = _embed(model, texts)
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    index.add(vectors)
    build_ms = (time.perf_counter() - started) * 1000

    query_vectors = _embed(model, [QUERY_INSTRUCTION + case.query for case in cases])

    hits_at: dict[int, int] = {k: 0 for k in RECALL_K}
    reciprocal_ranks: list[float] = []
    search_times: list[float] = []
    top_k = max(max(RECALL_K), MRR_K)

    for case_index, case in enumerate(cases):
        began = time.perf_counter()
        _, positions = index.search(query_vectors[case_index : case_index + 1], top_k)
        search_times.append((time.perf_counter() - began) * 1000)

        # Collapse chunks back to passages, keeping best rank per passage: two
        # chunks of the same passage are one retrieval from the user's view.
        ranked_passages: list[int] = []
        for position in positions[0]:
            if position < 0:
                continue
            origin_case, origin_passage = origins[position]
            if origin_case != case_index:
                # Another query's passage; counts as a miss, not a match.
                ranked_passages.append(-1)
                continue
            if origin_passage not in ranked_passages:
                ranked_passages.append(origin_passage)

        for k in RECALL_K:
            if any(passage in case.gold_indices for passage in ranked_passages[:k]):
                hits_at[k] += 1

        rank = next(
            (
                position + 1
                for position, passage in enumerate(ranked_passages[:MRR_K])
                if passage in case.gold_indices
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    total = len(cases)
    return StrategyResult(
        name=chunker.name,
        description=chunker.description,
        chunks=len(texts),
        vector_bytes=len(texts) * BYTES_PER_VECTOR,
        mean_chunk_chars=round(statistics.fmean(lengths), 1),
        build_ms=round(build_ms, 1),
        search_ms_p50=round(statistics.median(search_times), 4),
        recall={f"@{k}": round(hits_at[k] / total, 4) for k in RECALL_K},
        mrr5=round(statistics.fmean(reciprocal_ranks), 4),
    )


def run_comparison(
    queries: int,
    strategies: Sequence[str] = STRATEGY_ORDER,
    size: int | None = None,
) -> dict[str, Any]:
    from fastembed import TextEmbedding

    cases = load_cases(queries)
    if not cases:
        raise ValueError("no evaluation cases with gold passages were found")
    print(f"loaded {len(cases)} queries with gold labels")

    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    list(model.embed(["warmup"]))

    chunkers = [
        build_chunker(name, size) if size else get_chunker(name) for name in strategies
    ]
    results = [evaluate_strategy(chunker, cases, model) for chunker in chunkers]

    passage_lengths = [len(passage) for case in cases for passage in case.passages]
    baseline = next((r for r in results if r.name == "fixed"), results[0])
    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": EMBEDDING_MODEL,
        "queries": len(cases),
        "passages": len(passage_lengths),
        "chunk_size": size or MAX_CHILD_CHARS,
        # Recorded because window size relative to passage length decides
        # whether the fixed strategies split at all.
        "passage_chars_median": int(statistics.median(passage_lengths)),
        "baseline": baseline.name,
        "strategies": [asdict(result) for result in results],
    }


def load_cached() -> dict[str, Any] | None:
    if not COMPARISON_PATH.exists():
        return None
    try:
        return json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def format_table(report: dict[str, Any]) -> str:
    lines = [
        f"queries={report['queries']}  passages={report['passages']}  "
        f"size={report.get('chunk_size')}  median passage={report.get('passage_chars_median')} chars",
        "",
        f"| {'strategy':16} | {'chunks':>7} | {'MB':>6} | {'avg chars':>9} | "
        f"{'R@1':>6} | {'R@3':>6} | {'R@5':>6} | {'MRR@5':>6} | {'search':>7} |",
        f"| {'-' * 16} | {'-' * 7} | {'-' * 6} | {'-' * 9} | {'-' * 6} | {'-' * 6} | "
        f"{'-' * 6} | {'-' * 6} | {'-' * 7} |",
    ]
    for entry in report["strategies"]:
        lines.append(
            f"| {entry['name']:16} | {entry['chunks']:7,} | "
            f"{entry['vector_bytes'] / 1e6:6.2f} | {entry['mean_chunk_chars']:9.1f} | "
            f"{entry['recall']['@1']:6.3f} | {entry['recall']['@3']:6.3f} | "
            f"{entry['recall']['@5']:6.3f} | {entry['mrr5']:6.3f} | "
            f"{entry['search_ms_p50']:6.3f}ms |"
        )
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Compare chunking strategies on retrieval quality.")
    parser.add_argument("--queries", type=int, default=40)
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGY_ORDER))
    parser.add_argument("--size", type=int, default=None,
                        help="chunk window in chars; defaults to the production setting")
    args = parser.parse_args()

    report = run_comparison(args.queries, args.strategies, args.size)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + format_table(report))
    print(f"\nwrote {COMPARISON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
