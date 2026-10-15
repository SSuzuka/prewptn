"""Тесты сборки контекста: нумерация, дедуп, бюджет, промпт."""

from __future__ import annotations

import pytest

from aicore.rag.context import build_context, build_rag_prompt
from aicore.rag.store import SearchHit


def _hit(text: str, doc: str = "d", index: int = 0, score: float = 0.9) -> SearchHit:
    return SearchHit(
        id=f"{doc}:{index}",
        score=score,
        text=text,
        document_id=doc,
        chunk_index=index,
        title="Doc",
        source="src",
    )


class TestBuildContext:
    def test_numbers_and_labels_chunks(self) -> None:
        bundle = build_context([_hit("first"), _hit("second", index=1)])
        assert "[1] (Doc) first" in bundle.text
        assert "[2] (Doc) second" in bundle.text
        assert len(bundle.citations) == 2
        assert bundle.citations[0].marker == 1

    def test_deduplicates_identical_text(self) -> None:
        bundle = build_context([_hit("same text"), _hit("same text", index=1)])
        assert bundle.used_chunks == 1

    def test_respects_budget_and_flags_truncation(self) -> None:
        hits = [_hit("x" * 100 + str(i), index=i) for i in range(5)]
        bundle = build_context(hits, max_chars=250)
        assert bundle.truncated is True
        assert bundle.used_chunks < 5

    def test_empty_hits_bundle_is_falsy(self) -> None:
        bundle = build_context([])
        assert not bundle
        assert bundle.text == ""

    def test_invalid_budget_raises(self) -> None:
        with pytest.raises(ValueError):
            build_context([], max_chars=0)


class TestBuildRagPrompt:
    def test_empty_context_returns_question(self) -> None:
        assert build_rag_prompt("what?", build_context([])) == "what?"

    def test_prompt_contains_context_and_question(self) -> None:
        bundle = build_context([_hit("the answer is 42")])
        prompt = build_rag_prompt("what is the answer?", bundle)
        assert "the answer is 42" in prompt
        assert "what is the answer?" in prompt
        assert "[n]" in prompt
