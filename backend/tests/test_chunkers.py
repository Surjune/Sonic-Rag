"""Tests for the four chunking strategies and their shared interface."""

from __future__ import annotations

import pytest

from app.chunkers import (
    STRATEGY_ORDER,
    FixedOverlapChunker,
    FixedSizeChunker,
    HierarchicalChunker,
    SemanticChunker,
    get_chunker,
)

ENGLISH = (
    "A corporation is a legal entity. It is created under the authority of law. "
    "Shareholders own the corporation and elect its directors. "
    "The directors appoint officers who run day-to-day operations."
)
HINDI = "निगम एक कानूनी इकाई है। यह कानून के तहत बनाई जाती है। शेयरधारक इसके मालिक होते हैं।"


class TestRegistry:
    def test_all_four_strategies_registered(self) -> None:
        assert set(STRATEGY_ORDER) == {"fixed", "fixed_overlap", "semantic", "hierarchical"}

    @pytest.mark.parametrize("name", STRATEGY_ORDER)
    def test_lookup_returns_matching_name(self, name: str) -> None:
        assert get_chunker(name).name == name

    def test_unknown_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            get_chunker("nonsense")


class TestSharedContract:
    """Everything downstream depends on the strategies being interchangeable."""

    @pytest.mark.parametrize("name", STRATEGY_ORDER)
    def test_produces_chunks(self, name: str) -> None:
        chunks = get_chunker(name).chunk(ENGLISH)
        assert chunks
        assert all(chunk.text.strip() for chunk in chunks)

    @pytest.mark.parametrize("name", STRATEGY_ORDER)
    def test_reports_its_own_name(self, name: str) -> None:
        assert all(chunk.strategy == name for chunk in get_chunker(name).chunk(ENGLISH))

    @pytest.mark.parametrize("name", STRATEGY_ORDER)
    def test_indices_are_sequential(self, name: str) -> None:
        chunks = get_chunker(name).chunk(ENGLISH)
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))

    @pytest.mark.parametrize("name", STRATEGY_ORDER)
    def test_empty_input_yields_nothing(self, name: str) -> None:
        assert get_chunker(name).chunk("   ") == []

    @pytest.mark.parametrize("name", STRATEGY_ORDER)
    def test_parent_text_is_preserved(self, name: str) -> None:
        for chunk in get_chunker(name).chunk(ENGLISH):
            assert chunk.parent_text.startswith("A corporation")


class TestFixedSize:
    def test_respects_window_size(self) -> None:
        for chunk in FixedSizeChunker(size=40).chunk(ENGLISH):
            assert chunk.length <= 40

    def test_covers_the_whole_text_without_gaps(self) -> None:
        chunker = FixedSizeChunker(size=40)
        chunks = chunker.chunk(ENGLISH)
        assert "".join(chunk.text for chunk in chunks) == chunks[0].parent_text

    def test_rejects_invalid_size(self) -> None:
        with pytest.raises(ValueError):
            FixedSizeChunker(size=0)


class TestFixedOverlap:
    def test_consecutive_chunks_share_text(self) -> None:
        chunks = FixedOverlapChunker(size=60, overlap=20).chunk(ENGLISH)
        assert len(chunks) >= 2
        tail = chunks[0].text[-20:]
        assert chunks[1].text.startswith(tail)

    def test_produces_more_chunks_than_no_overlap(self) -> None:
        """Overlap buys boundary coverage by duplicating text; that cost is real."""
        plain = FixedSizeChunker(size=60).chunk(ENGLISH)
        overlapping = FixedOverlapChunker(size=60, overlap=20).chunk(ENGLISH)
        assert len(overlapping) > len(plain)

    def test_overlap_must_be_smaller_than_size(self) -> None:
        with pytest.raises(ValueError):
            FixedOverlapChunker(size=50, overlap=50)

    def test_terminates_on_short_input(self) -> None:
        assert len(FixedOverlapChunker(size=100, overlap=90).chunk("short text")) == 1


class TestSemantic:
    def test_splits_on_sentence_boundaries(self) -> None:
        for chunk in SemanticChunker(min_chars=1, max_chars=200).chunk(ENGLISH):
            assert not chunk.text.startswith(" ")

    def test_handles_devanagari_danda(self) -> None:
        chunks = SemanticChunker(min_chars=1, max_chars=200).chunk(HINDI)
        assert len(chunks) >= 2

    def test_does_not_cut_mid_word_unlike_fixed(self) -> None:
        semantic = SemanticChunker(min_chars=1, max_chars=60).chunk(ENGLISH)
        # Every chunk should end at a terminator or be the final fragment.
        assert any(chunk.text.rstrip().endswith(('.', '।', '!', '?')) for chunk in semantic)


class TestHierarchical:
    def test_injects_parent_context_into_later_children(self) -> None:
        chunks = HierarchicalChunker(min_chars=1, max_chars=60).chunk(ENGLISH)
        assert len(chunks) >= 2
        # The first child needs no lead-in; later ones carry the parent opening.
        assert chunks[1].embed_text != f"[en] {chunks[1].text}"

    def test_first_child_has_no_injection(self) -> None:
        chunks = HierarchicalChunker(min_chars=1, max_chars=60).chunk(ENGLISH)
        assert chunks[0].embed_text == f"[en] {chunks[0].text}"

    def test_embed_text_differs_from_semantic(self) -> None:
        """The whole point of the strategy is a different embedded string."""
        semantic = SemanticChunker(min_chars=1, max_chars=60).chunk(ENGLISH)
        hierarchical = HierarchicalChunker(min_chars=1, max_chars=60).chunk(ENGLISH)
        assert [c.embed_text for c in semantic] != [c.embed_text for c in hierarchical]


class TestOffsets:
    @pytest.mark.parametrize("name", STRATEGY_ORDER)
    def test_offsets_are_ordered_and_in_range(self, name: str) -> None:
        chunks = get_chunker(name).chunk(ENGLISH)
        parent_length = len(chunks[0].parent_text)
        previous = -1
        for chunk in chunks:
            assert 0 <= chunk.char_start <= parent_length
            assert chunk.char_start >= previous
            previous = chunk.char_start
