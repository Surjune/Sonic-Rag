"""Four chunking strategies behind one interface, so they can be compared.

    FixedSizeChunker         hard character windows, no regard for meaning
    FixedOverlapChunker      the same, with a sliding overlap
    SemanticChunker          splits on script-aware sentence boundaries
    HierarchicalChunker      semantic children carrying parent context

The point of keeping the naive strategies is measurement. "Semantic splitting is
better" is an assumption until a fixed-size baseline is scored against it on the
same corpus, and chunk_eval.py does exactly that.

Every strategy returns the same Chunk shape, so the indexer, the evaluator and
the preview endpoint are all strategy-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Sequence

from app.chunking import (
    MAX_CHILD_CHARS,
    MIN_CHILD_CHARS,
    build_embed_text,
    normalize,
    pack_sentences,
    split_sentences,
)

# Overlap as a share of window size. 20% is the usual starting point: enough to
# carry a sentence across a boundary, small enough that the index does not
# balloon (overlap inflates vector count by roughly 1/(1-ratio)).
DEFAULT_OVERLAP_RATIO = 0.20


@dataclass
class Chunk:
    """One unit produced by any strategy."""

    text: str  # the chunk itself
    embed_text: str  # what gets embedded, may carry injected context
    parent_text: str  # the passage it came from, for display and LLM context
    strategy: str
    index: int  # position within the parent
    char_start: int
    char_end: int
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.text)


class BaseChunker(ABC):
    """Common interface. `name` identifies the strategy in results and configs."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def split(self, text: str) -> list[tuple[str, int, int]]:
        """Return (chunk_text, char_start, char_end) triples."""

    def make_embed_text(self, chunk_text: str, parent_text: str, index: int, lang: str) -> str:
        """Default: embed the chunk as-is, tagged with its language."""
        return f"[{lang}] {chunk_text}"

    def chunk(self, text: str, *, lang: str = "en", metadata: dict[str, str] | None = None) -> list[Chunk]:
        parent = normalize(text)
        if not parent:
            return []
        chunks: list[Chunk] = []
        for index, (piece, start, end) in enumerate(self.split(parent)):
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    text=piece,
                    embed_text=self.make_embed_text(piece, parent, index, lang),
                    parent_text=parent,
                    strategy=self.name,
                    index=index,
                    char_start=start,
                    char_end=end,
                    metadata=dict(metadata or {}),
                )
            )
        return chunks


class FixedSizeChunker(BaseChunker):
    """Hard character windows. The naive baseline every other strategy is scored against.

    Cuts mid-word and mid-sentence by design: that is the failure mode being
    measured, not an oversight.
    """

    name = "fixed"
    description = "Fixed character windows with no overlap and no boundary awareness."

    def __init__(self, size: int = MAX_CHILD_CHARS) -> None:
        if size < 1:
            raise ValueError("size must be positive")
        self.size = size

    def split(self, text: str) -> list[tuple[str, int, int]]:
        return [
            (text[start : start + self.size], start, min(start + self.size, len(text)))
            for start in range(0, len(text), self.size)
        ]


class FixedOverlapChunker(BaseChunker):
    """Sliding windows that overlap.

    Overlap exists so a fact split across a boundary still appears whole in one
    chunk. The cost is duplication: at ratio r the index grows by about
    1/(1-r), which is why the evaluation reports index size alongside recall.
    """

    name = "fixed_overlap"
    description = "Fixed windows with a sliding overlap so boundary-straddling facts survive."

    def __init__(self, size: int = MAX_CHILD_CHARS, overlap: int | None = None) -> None:
        if size < 1:
            raise ValueError("size must be positive")
        self.size = size
        self.overlap = int(size * DEFAULT_OVERLAP_RATIO) if overlap is None else overlap
        if self.overlap >= size:
            raise ValueError("overlap must be smaller than size")

    def split(self, text: str) -> list[tuple[str, int, int]]:
        step = self.size - self.overlap
        pieces: list[tuple[str, int, int]] = []
        start = 0
        while start < len(text):
            end = min(start + self.size, len(text))
            pieces.append((text[start:end], start, end))
            if end >= len(text):
                break
            start += step
        return pieces


class SemanticChunker(BaseChunker):
    """Script-aware sentence boundaries, packed to a target size.

    Uses the Devanagari danda, Tamil and English terminators with guards for
    decimals, initials and abbreviations, then merges short sentences so no
    chunk is too sparse to retrieve.
    """

    name = "semantic"
    description = "Boundary-aware Indic and English sentence splitting, packed to size."

    def __init__(self, min_chars: int = MIN_CHILD_CHARS, max_chars: int = MAX_CHILD_CHARS) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

    def split(self, text: str) -> list[tuple[str, int, int]]:
        packed = pack_sentences(split_sentences(text), self.min_chars, self.max_chars)
        return _locate(text, packed)


class HierarchicalChunker(SemanticChunker):
    """Semantic children that carry a lead-in from their parent.

    A chunk lifted from the middle of a passage loses what it refers to. Adding
    the parent's opening restores the referent, at the cost of some repeated
    text in the embedding.
    """

    name = "hierarchical"
    description = "Semantic children with parent context injected into the embedded text."

    def make_embed_text(self, chunk_text: str, parent_text: str, index: int, lang: str) -> str:
        return build_embed_text(chunk_text, parent_text, index, lang)


def _locate(text: str, pieces: Sequence[str]) -> list[tuple[str, int, int]]:
    """Find each piece's offsets in the parent, scanning forward.

    Packing rejoins sentences with single spaces, so a piece is usually but not
    always a literal substring. When it is not, the offsets fall back to a
    running cursor rather than dropping the chunk.
    """
    located: list[tuple[str, int, int]] = []
    cursor = 0
    for piece in pieces:
        found = text.find(piece, cursor)
        if found < 0:
            located.append((piece, cursor, cursor + len(piece)))
            cursor += len(piece)
        else:
            located.append((piece, found, found + len(piece)))
            cursor = found + len(piece)
    return located


# Registry, keyed by the name used in configs, results and API payloads.
STRATEGIES: dict[str, BaseChunker] = {
    chunker.name: chunker
    for chunker in (
        FixedSizeChunker(),
        FixedOverlapChunker(),
        SemanticChunker(),
        HierarchicalChunker(),
    )
}

STRATEGY_ORDER: tuple[str, ...] = ("fixed", "fixed_overlap", "semantic", "hierarchical")


def get_chunker(name: str) -> BaseChunker:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; expected one of {sorted(STRATEGIES)}") from None


def build_chunker(name: str, size: int) -> BaseChunker:
    """Construct a strategy at a specific window size.

    Window size dominates any comparison between strategies. Measured on this
    corpus, passages run p50 294 and p90 478 characters, so at the default
    480-char window roughly 90% of them fit in a single chunk and overlap never
    engages -- the two fixed strategies then produce identical output. Being
    able to re-run at a smaller window is what makes the comparison informative
    rather than an artefact of one setting.
    """
    if name == "fixed":
        return FixedSizeChunker(size=size)
    if name == "fixed_overlap":
        return FixedOverlapChunker(size=size)
    if name == "semantic":
        return SemanticChunker(max_chars=size)
    if name == "hierarchical":
        return HierarchicalChunker(max_chars=size)
    raise ValueError(f"unknown strategy {name!r}; expected one of {sorted(STRATEGIES)}")
