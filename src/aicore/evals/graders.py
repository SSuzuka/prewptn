"""Deterministic graders for agent runs.

These are code graders, not LLM judges: they check the *trajectory* (which tools
ran, with what arguments) and the *outcome* (does the final answer contain the
expected grounding). Deterministic grading is the recommended first layer in
2026 agent-eval guidance — it is fast, cheap, reproducible in CI, and an LLM
judge is only worth adding on top for subjective quality.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage

from aicore.agents.usage import final_answer

__all__ = [
    "answer_contains",
    "args_carry_query",
    "called_tool",
    "called_tools",
    "final_answer",
    "tool_args",
    "tool_calls",
]


def tool_calls(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """All tool calls requested by the assistant across the run."""
    calls: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            calls.extend(message.tool_calls or [])
    return calls


def called_tools(messages: Sequence[Any]) -> list[str]:
    return [call.get("name", "") for call in tool_calls(messages)]


def called_tool(messages: Sequence[Any], name: str) -> bool:
    return name in called_tools(messages)


def tool_args(messages: Sequence[Any], name: str) -> list[dict[str, Any]]:
    return [call.get("args", {}) for call in tool_calls(messages) if call.get("name") == name]


def answer_contains(answer: str, keywords: Sequence[str]) -> bool:
    lowered = answer.lower()
    return all(keyword.lower() in lowered for keyword in keywords)


def args_carry_query(messages: Sequence[Any], name: str, query: str) -> bool:
    """The tool must be called with the user's question as its query argument."""
    expected = query.strip().lower()
    return any(
        str(args.get("query", "")).strip().lower() == expected for args in tool_args(messages, name)
    )
