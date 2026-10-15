"""Agent run accounting: usage/cost totals and final-answer extraction.

One place turns a run's model/tool activity into totals. Both the agent task
(`tasks/agent_run.py`) and the eval harness (`evals/runner.py`) read the same
:class:`AgentUsage`, so the saved `TaskRun` metrics, Prometheus and the eval
report can never disagree.

Why a callback handler rather than middleware: the supervisor calls the
knowledge subagent as a *tool*, and the subagent runs its own model loop. A
callback passed in the invocation config propagates into that nested run, so
:class:`AgentUsageCallbackHandler` sees every model and tool call in the whole
graph — including the subagent's — which a supervisor-only middleware cannot.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

from aicore.config import Settings, get_settings
from aicore.llm import LLMUsage, compute_cost_usd
from aicore.metrics import record_llm_call


@dataclass(slots=True)
class AgentUsage:
    """Accumulated model/tool activity of one agent run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0

    def add_model(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.model_calls += 1
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)

    def add_tool(self) -> None:
        self.tool_calls += 1

    def cost_usd(self, settings: Settings | None = None) -> float:
        cfg = settings or get_settings()
        return compute_cost_usd(
            LLMUsage(prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens),
            prompt_per_1k=cfg.llm_price_prompt_per_1k,
            completion_per_1k=cfg.llm_price_completion_per_1k,
        )


def usage_from_metadata(metadata: Any) -> tuple[int, int]:
    """(prompt_tokens, completion_tokens) from a ``usage_metadata`` mapping."""
    if not metadata:
        return 0, 0
    return (
        int(metadata.get("input_tokens", 0) or 0),
        int(metadata.get("output_tokens", 0) or 0),
    )


def _response_usage(response: Any) -> tuple[int, int]:
    """Extract token usage from an LLMResult (chat message or llm_output)."""
    try:
        generation = response.generations[0][0]
        message = getattr(generation, "message", None)
        prompt, completion = usage_from_metadata(getattr(message, "usage_metadata", None))
        if prompt or completion:
            return prompt, completion
    except (IndexError, AttributeError, TypeError):
        pass
    llm_output = getattr(response, "llm_output", None) or {}
    token_usage = llm_output.get("token_usage") or {}
    return (
        int(token_usage.get("prompt_tokens", 0) or 0),
        int(token_usage.get("completion_tokens", 0) or 0),
    )


def final_answer(messages: Sequence[Any]) -> str:
    """Last non-empty assistant answer in the run (the user-facing result)."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


class AgentUsageCallbackHandler(BaseCallbackHandler):
    """Accumulate run usage and record the same Prometheus metrics as the LLM layer.

    Attached through the invocation config, so it also observes the nested
    subagent model loop and every tool call (``research_knowledge_base`` and the
    subagent's ``search_knowledge_base``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.usage = AgentUsage()
        self._starts: dict[Any, float] = {}

    def _mark_start(self, run_id: Any) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_llm_start(self, *args: Any, run_id: Any = None, **kwargs: Any) -> None:
        self._mark_start(run_id)

    def on_chat_model_start(self, *args: Any, run_id: Any = None, **kwargs: Any) -> None:
        self._mark_start(run_id)

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        cfg = get_settings()
        prompt, completion = _response_usage(response)
        self.usage.add_model(prompt, completion)

        started = self._starts.pop(run_id, None)
        latency_ms = int((time.perf_counter() - started) * 1000) if started is not None else 0
        cost = compute_cost_usd(
            LLMUsage(prompt_tokens=prompt, completion_tokens=completion),
            prompt_per_1k=cfg.llm_price_prompt_per_1k,
            completion_per_1k=cfg.llm_price_completion_per_1k,
        )
        record_llm_call(
            backend=cfg.agent_backend,
            model=cfg.agent_model,
            prompt_tokens=prompt,
            completion_tokens=completion,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

    def on_llm_error(self, *args: Any, run_id: Any = None, **kwargs: Any) -> None:
        # A failed call never reaches on_llm_end, so without this the start
        # timestamp would linger in `_starts` for the rest of the run.
        self._starts.pop(run_id, None)

    def on_chain_error(self, *args: Any, run_id: Any = None, **kwargs: Any) -> None:
        self._starts.pop(run_id, None)

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
        self.usage.add_tool()
