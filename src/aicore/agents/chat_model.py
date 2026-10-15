"""LangChain chat-model factory for the agent layer.

``create_agent`` (LangChain 1.x) needs a LangChain ``BaseChatModel``. In
production that is ``ChatOpenAI`` against any OpenAI-compatible endpoint (extra
``openai``); for offline dev and tests we ship :class:`StubChatModel`, a
deterministic tool-calling model, so the whole agent graph runs without a
network or an API key — the same "stub by default" rule as the LLM and RAG
layers.

Bound tools are read from the invocation kwargs (``bind(tools=...)``) rather
than stored on the instance, so a single model object can safely be shared by a
supervisor and its subagents without their tool lists clobbering each other.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aicore.config import Settings, get_settings


def _tool_name(tool: Any) -> str | None:
    if isinstance(tool, dict):
        function = tool.get("function") or {}
        return tool.get("name") or function.get("name")
    return getattr(tool, "name", None)


def _last_human_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            if isinstance(message.content, str):
                return message.content
            return str(message.content)
    return ""


def _usage(text: str) -> dict[str, int]:
    """Approximate token counts so cost metrics are exercised offline.

    Stub-only: input and output are both ``len(text) // 4``, so the split is
    not realistic (it roughly doubles a ``len // 4`` total). That is fine here
    — the goal is to keep the cost counters moving, not to be token-accurate.
    """
    tokens = max(1, len(text) // 4)
    return {"input_tokens": tokens, "output_tokens": tokens, "total_tokens": tokens * 2}


class StubChatModel(BaseChatModel):
    """Deterministic tool-calling chat model: no network, no API keys.

    Emulates the minimal loop ``create_agent`` needs: on the first turn it calls
    the first bound tool with the user's question; once a ``ToolMessage`` is
    present it returns a grounded answer echoing the tool result. This exercises
    the real model → tools → model loop offline.
    """

    model_name: str = "stub-agent"

    @property
    def _llm_type(self) -> str:
        return "aicore-stub-agent"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[override]
        return self.bind(tools=list(tools), **kwargs)

    def _respond(self, messages: Sequence[BaseMessage], tools: Any) -> ChatResult:
        last = messages[-1] if messages else None
        if isinstance(last, ToolMessage):
            result = last.content if isinstance(last.content, str) else str(last.content)
            message = AIMessage(content=f"[stub-agent] {result}", usage_metadata=_usage(result))
            return ChatResult(generations=[ChatGeneration(message=message)])

        names = [name for name in (_tool_name(tool) for tool in tools or []) if name]
        if names:
            query = _last_human_text(messages)
            call = {
                "name": names[0],
                "args": {"query": query},
                "id": "stub-call-1",
                "type": "tool_call",
            }
            message = AIMessage(content="", tool_calls=[call], usage_metadata=_usage(query))
            return ChatResult(generations=[ChatGeneration(message=message)])

        message = AIMessage(content="[stub-agent] no tools bound", usage_metadata=_usage(""))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        return self._respond(messages, kwargs.get("tools"))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        return self._respond(messages, kwargs.get("tools"))


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Build the LangChain chat model used by agents.

    The OpenAI backend imports ``langchain-openai`` lazily, so a base install
    that only uses the stub never needs the optional ``openai`` extra.
    """
    cfg = settings or get_settings()
    if cfg.agent_backend == "openai":
        if not cfg.openai_api_key:
            raise RuntimeError("agent_backend='openai' requires OPENAI_API_KEY to be set")

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.agent_model,
            base_url=cfg.openai_base_url,
            api_key=cfg.openai_api_key,
            temperature=cfg.agent_temperature,
            timeout=cfg.llm_timeout_seconds,
            max_retries=cfg.llm_max_retries,
        )

    return StubChatModel(model_name=f"stub:{cfg.agent_model}")
