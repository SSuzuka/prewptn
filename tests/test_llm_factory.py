"""Tests for the LLM client factory and process-wide singleton."""

from __future__ import annotations

import pytest

from aicore.config import Settings
from aicore.llm import (
    build_llm_client,
    get_llm_client,
    reset_llm_client,
    set_llm_client,
)
from aicore.llm.openai_compat import OpenAICompatibleLLM
from aicore.llm.stub import StubLLM


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_llm_client()
    yield
    reset_llm_client()


def test_stub_backend_builds_stub() -> None:
    assert isinstance(build_llm_client(Settings(llm_backend="stub")), StubLLM)


def test_openai_backend_builds_openai_client() -> None:
    cfg = Settings(llm_backend="openai", openai_api_key="sk-x")
    assert isinstance(build_llm_client(cfg), OpenAICompatibleLLM)


def test_singleton_is_cached_and_overridable() -> None:
    first = get_llm_client()
    assert get_llm_client() is first

    replacement = StubLLM(settings=Settings())
    set_llm_client(replacement)
    assert get_llm_client() is replacement
