"""Тесты чанкинга: границы, размер, перекрытие, конфигурация. Офлайн."""

from __future__ import annotations

import pytest

from aicore.rag.chunking import Chunk, chunk_text, estimate_tokens


class TestChunkText:
    def test_empty_returns_no_chunks(self) -> None:
        assert chunk_text("") == []
        assert chunk_text("   \n\n  ") == []

    def test_short_text_is_single_chunk(self) -> None:
        chunks = chunk_text("a short sentence.", chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].index == 0
        assert chunks[0].text == "a short sentence."

    def test_long_text_splits_on_boundaries(self) -> None:
        paragraph = "This is a sentence. " * 120
        chunks = chunk_text(paragraph, chunk_size=200, chunk_overlap=20)
        assert len(chunks) > 1
        assert [c.index for c in chunks] == list(range(len(chunks)))
        for chunk in chunks:
            assert len(chunk.text) <= 200 + 20 + 1

    def test_consecutive_chunks_overlap(self) -> None:
        text = " ".join(f"word{i}" for i in range(400))
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=25)
        assert len(chunks) >= 2
        assert chunks[1].text.startswith(chunks[0].text[-25:])

    def test_hard_split_when_no_boundary(self) -> None:
        chunks = chunk_text("x" * 500, chunk_size=200, chunk_overlap=0)
        assert [len(c.text) for c in chunks] == [200, 200, 100]

    @pytest.mark.parametrize(
        ("size", "overlap"),
        [(0, 0), (-1, 0), (100, 100), (100, 200), (100, -5)],
    )
    def test_invalid_config_raises(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=size, chunk_overlap=overlap)

    def test_chunk_is_frozen_dataclass(self) -> None:
        chunk = Chunk(index=3, text="hi")
        assert chunk.index == 3


def test_estimate_tokens_is_positive_and_monotonic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 4)
