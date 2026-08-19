"""Boundary-aware hierarchical chunking for Indic and English passages.

Two-level hierarchy:

    ParentPassage   one full passage from an MSMARCO-XI row (display + context)
      └─ ChildChunk   a semantic unit inside it (what actually gets embedded)

Retrieval matches on small child chunks for precision, then the parent is
available for display and for giving the LLM wider context.

Sentence boundaries are script-aware. Devanagari terminates sentences with the
danda (U+0964) and double danda (U+0965); Tamil and English use ASCII `.`/`?`/`!`.
A bare `.` is ambiguous, so it is guarded against decimals ("3.14"), initials
("A. Sharma") and common abbreviations ("Dr.", "etc.").

This module runs at INDEX time only. It is deliberately not on the query path,
so it costs nothing against the sub-200ms request budget.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# --- script constants -------------------------------------------------------

DEVANAGARI_DANDA = "।"
DEVANAGARI_DOUBLE_DANDA = "॥"

# Characters that can end a sentence in any script we index.
TERMINATOR_CHARS = f".!?{DEVANAGARI_DANDA}{DEVANAGARI_DOUBLE_DANDA}"

# Trailing characters allowed to ride along after a terminator, e.g. `word."` .
CLOSING_CHARS = "\"')]}»”’"

# --- chunk sizing -----------------------------------------------------------
# Tuned for bge-small-en-v1.5 (512-token limit). Chars, not tokens: a cheap
# proxy that avoids loading a tokenizer during indexing.

MIN_CHILD_CHARS = 80  # below this a chunk is too sparse to retrieve well; merge it
MAX_CHILD_CHARS = 480  # keeps a child comfortably inside the embedder's window
CONTEXT_PREFIX_CHARS = 96  # parent lead-in prepended to non-leading children

# Abbreviations whose trailing period is not a sentence boundary.
ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc",
        "inc", "ltd", "co", "corp", "no", "fig", "approx", "dept", "est",
        "min", "max", "avg", "al", "eg", "ie", "cf", "vol", "pp", "ed",
    }
)

_TRAILING_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)$")
_WHITESPACE = re.compile(r"\s+")

# MSMARCO-XI stores the native-language passage under a different key per split.
NATIVE_PASSAGE_KEY = "Translated_passages"
ENGLISH_PASSAGE_KEY = "English_passages"
SELECTED_KEY = "is_selected"


@dataclass(frozen=True)
class ParentPassage:
    """One full passage: the unit shown to the user and fed to the LLM."""

    parent_id: str
    query_id: int
    lang: str
    passage_index: int
    text: str
    text_english: str
    is_selected: bool


@dataclass(frozen=True)
class ChildChunk:
    """One semantic unit inside a parent: the unit that gets embedded."""

    chunk_id: str
    parent_id: str
    query_id: int
    lang: str
    passage_index: int
    child_index: int
    text: str
    text_english: str
    embed_text: str
    is_selected: bool


def normalize(text: str) -> str:
    """NFC-normalize and collapse whitespace.

    NFC matters for Indic scripts: the same visual grapheme can arrive as either
    a precomposed codepoint or a base plus combining mark, and the two would
    otherwise embed and deduplicate as different strings.
    """
    if not text:
        return ""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def _is_sentence_boundary(text: str, index: int) -> bool:
    """Decide whether the terminator at `index` really ends a sentence."""
    char = text[index]

    # Danda, double danda, `!` and `?` are unambiguous.
    if char != ".":
        return True

    # "3.14" — a decimal point, not a full stop.
    if 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False

    match = _TRAILING_WORD.search(text[:index])
    if match:
        word = match.group(1).lower().rstrip(".")
        if word in ABBREVIATIONS:
            return False
        # A single letter before the period reads as an initial: "A. Sharma".
        if len(word) == 1:
            return False

    return True


def split_sentences(text: str) -> list[str]:
    """Split into sentences using script-aware, guarded boundaries."""
    cleaned = normalize(text)
    if not cleaned:
        return []

    sentences: list[str] = []
    start = 0
    index = 0
    length = len(cleaned)

    while index < length:
        if cleaned[index] in TERMINATOR_CHARS and _is_sentence_boundary(cleaned, index):
            end = index + 1
            # Absorb runs like "?!" and any closing quote or bracket.
            while end < length and cleaned[end] in TERMINATOR_CHARS + CLOSING_CHARS:
                end += 1
            # A real boundary is followed by whitespace or the end of the text.
            if end >= length or cleaned[end].isspace():
                piece = cleaned[start:end].strip()
                if piece:
                    sentences.append(piece)
                start = end
                index = end
                continue
        index += 1

    tail = cleaned[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_oversized(sentence: str, max_chars: int) -> list[str]:
    """Break a sentence that exceeds `max_chars`, preferring word boundaries."""
    if len(sentence) <= max_chars:
        return [sentence]

    parts: list[str] = []
    current = ""
    for word in sentence.split(" "):
        # A single token longer than the limit (URL, long compound) is sliced.
        if len(word) > max_chars:
            if current:
                parts.append(current)
                current = ""
            for offset in range(0, len(word), max_chars):
                parts.append(word[offset : offset + max_chars])
            continue
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return parts


def pack_sentences(
    sentences: Sequence[str],
    min_chars: int = MIN_CHILD_CHARS,
    max_chars: int = MAX_CHILD_CHARS,
) -> list[str]:
    """Group sentences into chunks between `min_chars` and `max_chars`."""
    units: list[str] = []
    for sentence in sentences:
        units.extend(_split_oversized(sentence, max_chars))

    packed: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
        elif len(current) + 1 + len(unit) <= max_chars:
            current = f"{current} {unit}"
        else:
            packed.append(current)
            current = unit
        # Emit as soon as the chunk is substantial enough to stand alone.
        if len(current) >= min_chars:
            packed.append(current)
            current = ""

    if current:
        # A short trailing remainder rides along with the previous chunk rather
        # than becoming a sparse, poorly-retrievable fragment of its own — but
        # only when the merge still fits, otherwise it would overflow the
        # embedder's window and be silently truncated.
        merged_fits = packed and len(packed[-1]) + 1 + len(current) <= max_chars
        if packed and len(current) < min_chars and merged_fits:
            packed[-1] = f"{packed[-1]} {current}"
        else:
            packed.append(current)
    return packed


def build_embed_text(child_text: str, parent_text: str, child_index: int, lang: str) -> str:
    """Compose the string that is actually embedded.

    Two additions to the raw child text:

    * a language tag, which is discriminative in a mixed hi/ta/en index; and
    * for any child after the first, a short lead-in from its parent, so a chunk
      lifted out of the middle of a passage still carries what it refers to.

    The originating query is deliberately NOT injected: the query is what we
    search with, so embedding it into the passage would leak the answer and make
    retrieval scores meaningless.
    """
    tag = f"[{lang}]"
    if child_index == 0:
        return f"{tag} {child_text}"

    prefix = parent_text[:CONTEXT_PREFIX_CHARS].strip()
    if not prefix or parent_text.startswith(child_text):
        return f"{tag} {child_text}"
    return f"{tag} {prefix}… {child_text}"


def chunk_passage(
    passage_text: str,
    english_text: str,
    *,
    query_id: int,
    lang: str,
    passage_index: int,
    is_selected: bool,
) -> tuple[ParentPassage | None, list[ChildChunk]]:
    """Chunk a single passage into its parent record and child chunks."""
    parent_text = normalize(passage_text)
    if not parent_text:
        return None, []

    parent_id = f"{query_id}:{lang}:{passage_index}"
    parent = ParentPassage(
        parent_id=parent_id,
        query_id=query_id,
        lang=lang,
        passage_index=passage_index,
        text=parent_text,
        text_english=normalize(english_text),
        is_selected=is_selected,
    )

    children: list[ChildChunk] = []
    for child_index, child_text in enumerate(pack_sentences(split_sentences(parent_text))):
        children.append(
            ChildChunk(
                chunk_id=f"{parent_id}:{child_index}",
                parent_id=parent_id,
                query_id=query_id,
                lang=lang,
                passage_index=passage_index,
                child_index=child_index,
                text=child_text,
                text_english=parent.text_english,
                embed_text=build_embed_text(child_text, parent_text, child_index, lang),
                is_selected=is_selected,
            )
        )
    return parent, children


def chunk_row(
    row: Mapping[str, Any], lang: str
) -> tuple[list[ParentPassage], list[ChildChunk]]:
    """Chunk every passage in one MSMARCO-XI row.

    For `en` the English passage list is the native text; for `hi`/`ta` the
    translated list is native and the aligned English is carried alongside.
    """
    passages = row.get("passages") or {}
    english: Sequence[str] = passages.get(ENGLISH_PASSAGE_KEY) or []
    translated: Sequence[str] = passages.get(NATIVE_PASSAGE_KEY) or []
    selected: Sequence[int] = passages.get(SELECTED_KEY) or []

    native = english if lang == "en" else translated
    query_id = int(row["query_id"])

    parents: list[ParentPassage] = []
    children: list[ChildChunk] = []
    for passage_index, passage_text in enumerate(native):
        aligned_english = english[passage_index] if passage_index < len(english) else ""
        is_selected = bool(selected[passage_index]) if passage_index < len(selected) else False
        parent, child_chunks = chunk_passage(
            passage_text,
            aligned_english,
            query_id=query_id,
            lang=lang,
            passage_index=passage_index,
            is_selected=is_selected,
        )
        if parent is None:
            continue
        parents.append(parent)
        children.extend(child_chunks)
    return parents, children


def iter_rows(parquet_path: Path, limit: int) -> Iterator[dict[str, Any]]:
    """Stream the first `limit` rows off a local parquet file."""
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(parquet_path)
    seen = 0
    for batch in parquet_file.iter_batches(batch_size=min(limit, 64)):
        for row in batch.to_pylist():
            yield row
            seen += 1
            if seen >= limit:
                return


def _demo(lang: str, rows: int) -> int:
    """Print the chunk hierarchy for a few real rows (verification aid)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    filenames = {
        "hi": "hinval.parquet",
        "ta": "tamval.parquet",
        "en": "hinval.parquet",  # English rides along inside the Hindi file
    }
    data_dir = Path(__file__).resolve().parents[1] / "data" / "msmarco-xi" / "validation"
    path = data_dir / filenames[lang]
    if not path.exists():
        print(f"[FAIL] missing {path}; run test_dataset_connection.py first")
        return 1

    total_parents = 0
    total_children = 0
    for row in iter_rows(path, rows):
        parents, children = chunk_row(row, lang)
        total_parents += len(parents)
        total_children += len(children)
        print(f"\n=== query_id {row['query_id']} [{lang}] ===")
        print(f"    query: {row['query'][:90]}")
        for parent in parents[:2]:
            kids = [c for c in children if c.parent_id == parent.parent_id]
            flag = " *selected" if parent.is_selected else ""
            print(f"  PARENT {parent.parent_id}{flag}  ({len(parent.text)} chars, {len(kids)} children)")
            for child in kids:
                print(f"    CHILD {child.child_index}  ({len(child.text)} chars)  {child.text[:70]}")

    print(f"\n{total_parents} parents -> {total_children} children")
    if total_parents:
        print(f"avg {total_children / total_parents:.2f} children per parent")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Sonic-RAG chunking on real rows.")
    parser.add_argument("--lang", choices=["hi", "ta", "en"], default="hi")
    parser.add_argument("--rows", type=int, default=2)
    args = parser.parse_args()
    raise SystemExit(_demo(args.lang, args.rows))


if __name__ == "__main__":
    main()
