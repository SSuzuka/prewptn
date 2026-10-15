"""Тесты детерминированной агентной модели (офлайн, без ключей)."""

from __future__ import annotations

import pytest
from langchain_core.messages import ToolMessage

from aicore.agents.chat_model import StubChatModel, build_chat_model
from aicore.config import Settings


class TestStubChatModel:
    @pytest.mark.asyncio
    async def test_first_turn_calls_the_first_bound_tool(self) -> None:
        model = StubChatModel()
        bound = model.bind_tools([{"name": "alpha"}])
        message = await bound.ainvoke([{"role": "user", "content": "find x"}])
        assert message.tool_calls[0]["name"] == "alpha"
        assert message.tool_calls[0]["args"] == {"query": "find x"}

    @pytest.mark.asyncio
    async def test_answers_after_a_tool_message(self) -> None:
        model = StubChatModel()
        bound = model.bind_tools([{"name": "alpha"}])
        message = await bound.ainvoke(
            [
                {"role": "user", "content": "find x"},
                ToolMessage(content="RESULT", tool_call_id="stub-call-1"),
            ]
        )
        assert "RESULT" in message.content

    @pytest.mark.asyncio
    async def test_shared_model_does_not_clobber_bindings(self) -> None:
        # Один инстанс безопасно делить супервизору и подагенту: инструменты
        # читаются из kwargs вызова, а не хранятся на модели.
        model = StubChatModel()
        first = await model.bind_tools([{"name": "a"}]).ainvoke([{"role": "user", "content": "q"}])
        second = await model.bind_tools([{"name": "b"}]).ainvoke([{"role": "user", "content": "q"}])
        assert first.tool_calls[0]["name"] == "a"
        assert second.tool_calls[0]["name"] == "b"

    @pytest.mark.asyncio
    async def test_reports_usage_metadata(self) -> None:
        model = StubChatModel()
        message = await model.bind_tools([{"name": "a"}]).ainvoke(
            [{"role": "user", "content": "hello world"}]
        )
        assert message.usage_metadata["total_tokens"] > 0


class TestBuildChatModel:
    def test_defaults_to_stub(self) -> None:
        assert isinstance(build_chat_model(Settings()), StubChatModel)

    def test_openai_without_key_raises(self) -> None:
        with pytest.raises(RuntimeError):
            build_chat_model(Settings(agent_backend="openai", openai_api_key=None))
