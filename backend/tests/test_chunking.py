"""Tests for boundary-aware Indic/English chunking."""

from __future__ import annotations

from app.chunking import (
    MAX_CHILD_CHARS,
    MIN_CHILD_CHARS,
    ChildChunk,
    build_embed_text,
    chunk_passage,
    chunk_row,
    normalize,
    pack_sentences,
    split_sentences,
)


class TestNormalize:
    def test_collapses_whitespace(self) -> None:
        assert normalize("a  \n\t b ") == "a b"

    def test_empty_input(self) -> None:
        assert normalize("") == ""

    def test_nfc_composes_devanagari(self) -> None:
        # U+0928 U+093C (na + nukta) composes to U+0929.
        assert normalize("ऩ") == "ऩ"


class TestSplitSentences:
    def test_devanagari_danda(self) -> None:
        text = "यह पहला वाक्य है। यह दूसरा वाक्य है।"
        assert split_sentences(text) == ["यह पहला वाक्य है।", "यह दूसरा वाक्य है।"]

    def test_double_danda(self) -> None:
        assert len(split_sentences("पहला॥ दूसरा॥")) == 2

    def test_tamil_uses_ascii_period(self) -> None:
        text = "இது முதல் வாக்கியம். இது இரண்டாவது வாக்கியம்."
        assert len(split_sentences(text)) == 2

    def test_english_basic(self) -> None:
        assert len(split_sentences("One. Two! Three?")) == 3

    def test_decimal_is_not_a_boundary(self) -> None:
        assert split_sentences("Pi is 3.14 exactly.") == ["Pi is 3.14 exactly."]

    def test_abbreviation_is_not_a_boundary(self) -> None:
        assert split_sentences("Dr. Rao arrived.") == ["Dr. Rao arrived."]

    def test_initial_is_not_a_boundary(self) -> None:
        assert split_sentences("A. Sharma wrote it.") == ["A. Sharma wrote it."]

    def test_closing_quote_rides_along(self) -> None:
        assert split_sentences('He said "go." Then left.') == ['He said "go."', "Then left."]

    def test_unterminated_tail_is_kept(self) -> None:
        assert split_sentences("Complete. Incomplete") == ["Complete.", "Incomplete"]

    def test_empty_input(self) -> None:
        assert split_sentences("") == []


class TestPackSentences:
    def test_merges_short_sentences(self) -> None:
        packed = pack_sentences(["Tiny.", "Also tiny.", "Still small."], min_chars=40)
        assert len(packed) < 3

    def test_respects_max_chars(self) -> None:
        long_sentence = " ".join(["word"] * 400)
        for chunk in pack_sentences([long_sentence]):
            assert len(chunk) <= MAX_CHILD_CHARS

    def test_slices_token_longer_than_limit(self) -> None:
        for chunk in pack_sentences(["x" * (MAX_CHILD_CHARS * 2)]):
            assert len(chunk) <= MAX_CHILD_CHARS

    def test_short_remainder_is_absorbed(self) -> None:
        packed = pack_sentences(["A" * MIN_CHILD_CHARS + ".", "tail"])
        assert not any(len(chunk) < 10 for chunk in packed)

    def test_empty_input(self) -> None:
        assert pack_sentences([]) == []


class TestEmbedText:
    def test_first_child_has_no_parent_prefix(self) -> None:
        result = build_embed_text("First part.", "First part. Second part.", 0, "hi")
        assert result == "[hi] First part."

    def test_later_child_gets_parent_context(self) -> None:
        parent = "Corporations are legal entities. They can own property."
        result = build_embed_text("They can own property.", parent, 1, "en")
        assert result.startswith("[en] Corporations")
        assert "They can own property." in result

    def test_query_is_never_injected(self) -> None:
        # Guards the retrieval-integrity decision documented in build_embed_text.
        parent = "Some passage text here."
        result = build_embed_text("Some passage text here.", parent, 0, "en")
        assert "?" not in result


class TestChunkPassage:
    def test_builds_parent_and_children(self) -> None:
        parent, children = chunk_passage(
            "यह पहला वाक्य है। यह दूसरा वाक्य है। यह तीसरा वाक्य है।",
            "This is one. This is two.",
            query_id=42,
            lang="hi",
            passage_index=3,
            is_selected=True,
        )
        assert parent is not None
        assert parent.parent_id == "42:hi:3"
        assert parent.is_selected is True
        assert children
        assert all(isinstance(c, ChildChunk) for c in children)
        assert all(c.parent_id == "42:hi:3" for c in children)
        assert [c.child_index for c in children] == list(range(len(children)))

    def test_empty_passage_yields_nothing(self) -> None:
        parent, children = chunk_passage(
            "   ", "", query_id=1, lang="hi", passage_index=0, is_selected=False
        )
        assert parent is None
        assert children == []


class TestChunkRow:
    def _row(self) -> dict[str, object]:
        return {
            "query_id": 7,
            "passages": {
                "is_selected": [0, 1],
                "English_passages": ["English one. English two.", "English three."],
                "Translated_passages": ["हिंदी एक। हिंदी दो।", "हिंदी तीन।"],
            },
        }

    def test_hindi_uses_translated_passages(self) -> None:
        parents, _ = chunk_row(self._row(), "hi")
        assert len(parents) == 2
        assert "हिंदी" in parents[0].text
        assert parents[0].text_english.startswith("English one")

    def test_english_uses_english_passages(self) -> None:
        parents, _ = chunk_row(self._row(), "en")
        assert parents[0].text.startswith("English one")

    def test_is_selected_is_carried_through(self) -> None:
        parents, _ = chunk_row(self._row(), "hi")
        assert parents[0].is_selected is False
        assert parents[1].is_selected is True

    def test_missing_passages_key_is_safe(self) -> None:
        parents, children = chunk_row({"query_id": 1}, "hi")
        assert parents == []
        assert children == []
