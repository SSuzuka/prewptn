"""Supervisor agent: a ``create_agent`` graph that delegates to subagents.

Structure (the 2026 "subagents as tools" pattern):

    supervisor  --research_knowledge_base-->  knowledge subagent
                                                  └── search_knowledge_base (RAG)

The supervisor owns the conversation and decides what to delegate; the subagent
owns the retrieval loop. Guardrails (model/tool call limits, optional history
summarization) are attached as middleware, not hard-coded into the loop.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver

from aicore.agents.chat_model import build_chat_model
from aicore.agents.subagents import build_knowledge_subagent, build_research_tool
from aicore.config import Settings, get_settings

SUPERVISOR_SYSTEM_PROMPT = (
    "You are a supervisor agent with access to a knowledge-base research "
    "subagent. For factual questions that need grounding, delegate to the "
    "research_knowledge_base tool and answer from its findings, citing the "
    "documents it used. Keep answers concise; never invent facts the subagent "
    "did not return."
)


class BoundedInMemorySaver(InMemorySaver):
    """``InMemorySaver`` that keeps at most ``max_threads`` conversations.

    The checkpointer is process-wide and keyed by a client-supplied
    ``thread_id``, so an unbounded store grows with every distinct id. Oldest
    threads are evicted once the cap is reached, keeping worker memory bounded.

    DEV/TEST ONLY — replace before production. ``InMemorySaver`` lives in one
    process: under ``RedisStreamBroker`` with several worker processes/replicas
    a client's ``thread_id`` lands on a different process per request, so the
    "conversation memory" silently vanishes. A real deployment must inject a
    shared saver (Postgres/Redis) via the ``checkpointer`` argument; the
    process-wide singleton here is a dev convenience, not a prod guarantee.
    """

    def __init__(self, max_threads: int = 256) -> None:
        super().__init__()
        self._max_threads = max(1, int(max_threads))
        self._thread_order: deque[str] = deque()

    def _record_thread(self, thread_id: str) -> None:
        try:
            self._thread_order.remove(thread_id)
        except ValueError:
            pass
        self._thread_order.append(thread_id)
        while len(self._thread_order) > self._max_threads:
            self.delete_thread(self._thread_order.popleft())

    def put(self, config, checkpoint, metadata, new_versions):  # type: ignore[override]
        result = super().put(config, checkpoint, metadata, new_versions)
        self._record_thread(config["configurable"]["thread_id"])
        return result

    async def aput(self, config, checkpoint, metadata, new_versions):  # type: ignore[override]
        result = await super().aput(config, checkpoint, metadata, new_versions)
        self._record_thread(config["configurable"]["thread_id"])
        return result


def build_agent(
    settings: Settings | None = None,
    *,
    model: BaseChatModel | None = None,
    checkpointer: Any = None,
) -> Runnable:
    """Build the supervisor agent graph from settings."""
    cfg = settings or get_settings()
    chat_model = model or build_chat_model(cfg)
    checkpoint = (
        checkpointer if checkpointer is not None else BoundedInMemorySaver(cfg.agent_max_threads)
    )

    # The subagent is one-shot (invoked as a tool), so it gets no shared
    # checkpointer: retaining a thread per call would leak memory forever.
    subagent = build_knowledge_subagent(chat_model)
    tools = [build_research_tool(subagent)]

    middleware: list[Any] = [
        ModelCallLimitMiddleware(run_limit=cfg.agent_max_model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=cfg.agent_max_tool_calls),
    ]
    if cfg.agent_summarize_trigger_tokens > 0:
        middleware.append(
            SummarizationMiddleware(
                chat_model,
                trigger=("tokens", cfg.agent_summarize_trigger_tokens),
            )
        )

    return create_agent(
        chat_model,
        tools=tools,
        system_prompt=SUPERVISOR_SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=checkpoint,
        name="supervisor",
    )


_agent: Runnable | None = None


def get_agent() -> Runnable:
    """Process-wide agent graph, built on first use."""
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def set_agent(agent: Runnable) -> None:
    global _agent
    _agent = agent


def reset_agent() -> None:
    global _agent
    _agent = None
