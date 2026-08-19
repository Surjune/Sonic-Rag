"""Phase 0 handshake: verify access to ai4bharat/MSMARCO-XI for hi / ta / en.

Why this does not use `load_dataset(..., streaming=True)`
--------------------------------------------------------
The upstream repo ships one parquet file per language, and each `train/` file is a
SINGLE parquet row group (e.g. hintrain.parquet = 778,638 rows, 9.7 GB uncompressed).
A row group is the smallest readable unit in parquet, so reading even one row forces a
full ~3.7 GB download plus a ~9.7 GB in-memory materialisation. That reliably dies with
`realloc of size 3221225472 failed` on a normal machine.

This is an upstream defect, not a local one: Hugging Face's own datasets-server fails on
this dataset with
    TooBigRowGroupsError: first row group has 9711021317, exceeds the limit of 300000000
which is why the dataset has no working viewer / rows API.

The `validation/` files carry the same schema at a workable size (97,941 rows,
~460 MB on disk, ~1.2 GB uncompressed), so they are the ingestion source. Files are
fetched once via `hf_hub_download` (resumable, cached) and then read locally, which also
removes network flakiness from the hot path.

Language scope is restricted to Hindi, Tamil and English. English is not a separate file;
it travels inside every row as `Eng_Query` / `Eng_Answer` / `passages.English_passages`.
"""

from __future__ import annotations

import sys
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# The Windows console defaults to cp1252, which cannot encode Devanagari or Tamil.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATASET_NAME = "ai4bharat/MSMARCO-XI"
LOCAL_DIR = "data/msmarco-xi"

# The repo names files by 3-letter code; we expose the 2-letter codes used by the app.
LANG_FILES: dict[str, str] = {
    "hi": "validation/hinval.parquet",
    "ta": "validation/tamval.parquet",
}

# English is derived from the Eng_* columns of the files above, so it has no entry here.
DERIVED_LANGS = ("en",)

# Only the columns the RAG pipeline needs; skips `meta`, `source_lang`, `target_lang`.
PROJECTED_COLUMNS = [
    "query_id",
    "query_type",
    "query",
    "Answer",
    "Eng_Query",
    "Eng_Answer",
    "passages",
]

PASSAGE_SUBFIELDS = ("English_passages", "Translated_passages", "is_selected")

SAMPLE_COUNT = 3


def fetch(repo_file: str) -> str:
    """Download one parquet file (resumable, cached) and return its local path."""
    return hf_hub_download(
        DATASET_NAME,
        repo_file,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
    )


def preview(text: str | None, limit: int = 90) -> str:
    if not text:
        return "<empty>"
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def check_schema(row: dict[str, Any]) -> list[str]:
    """Return a list of schema problems; empty means the row matches the spec."""
    problems = [col for col in PROJECTED_COLUMNS if col not in row]

    passages = row.get("passages")
    if not isinstance(passages, dict):
        problems.append("passages is not a struct")
        return problems

    problems.extend(
        f"passages.{sub}" for sub in PASSAGE_SUBFIELDS if sub not in passages
    )

    english = passages.get("English_passages") or []
    translated = passages.get("Translated_passages") or []
    if english and translated and len(english) != len(translated):
        problems.append(
            f"passage count mismatch: {len(english)} English vs {len(translated)} translated"
        )
    return problems


def handshake(lang: str, repo_file: str) -> bool:
    print(f"\n--- {lang}  ({repo_file}) ---")

    try:
        path = fetch(repo_file)
    except Exception as exc:
        print(f"[FAIL] download failed: {type(exc).__name__}: {exc}")
        return False

    try:
        pf = pq.ParquetFile(path)
        total_rows = pf.metadata.num_rows
        batch = next(pf.iter_batches(batch_size=SAMPLE_COUNT, columns=PROJECTED_COLUMNS))
        rows = batch.to_pylist()
    except Exception as exc:
        print(f"[FAIL] parquet read failed: {type(exc).__name__}: {exc}")
        return False

    if not rows:
        print("[FAIL] file yielded zero rows")
        return False

    problems = check_schema(rows[0])
    if problems:
        print(f"[FAIL] schema mismatch: {problems}")
        return False

    print(f"[OK] {total_rows:,} rows available, schema verified")

    row = rows[0]
    passages = row["passages"]
    print(f"     query_id      : {row['query_id']}  ({row['query_type']})")
    print(f"     query    [{lang}] : {preview(row['query'])}")
    print(f"     query    [en] : {preview(row['Eng_Query'])}")
    print(f"     answer   [{lang}] : {preview(row['Answer'])}")
    print(f"     answer   [en] : {preview(row['Eng_Answer'])}")
    print(f"     passages      : {len(passages['English_passages'])} per row")
    print(f"     passage  [{lang}] : {preview(passages['Translated_passages'][0])}")
    print(f"     passage  [en] : {preview(passages['English_passages'][0])}")
    print(f"     is_selected   : {passages['is_selected']}")
    return True


def main() -> None:
    print(f"dataset : {DATASET_NAME}")
    print(f"scope   : {', '.join([*LANG_FILES, *DERIVED_LANGS])}")
    print(f"source  : validation split (train row groups are 9.7GB and unreadable)")

    results = {lang: handshake(lang, f) for lang, f in LANG_FILES.items()}

    print("\n=== summary ===")
    for lang, ok in results.items():
        print(f"{'OK  ' if ok else 'FAIL'} {lang}")
    if all(results.values()):
        print("OK   en  (derived from Eng_Query / Eng_Answer / English_passages)")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
