"""Agent layer: supervisor + subagents built on LangChain 1.x ``create_agent``.

Provider choice is a config switch, exactly like the LLM and RAG layers:
``agent_backend=stub`` is the deterministic offline model for dev and tests,
``openai`` is ``ChatOpenAI`` on an OpenAI-compatible endpoint. Callers depend on
the compiled graph from :func:`build_agent` / :func:`get_agent`, never on a
concrete model.
"""

from __future__ import annotations

from aicore.agents.agent import (
    SUPERVISOR_SYSTEM_PROMPT,
    build_agent,
    get_agent,
    reset_agent,
    set_agent,
)
from aicore.agents.chat_model import StubChatModel, build_chat_model
from aicore.agents.subagents import build_knowledge_subagent, build_research_tool
from aicore.agents.tools import search_knowledge_base
from aicore.agents.usage import AgentUsage, AgentUsageCallbackHandler, final_answer

__all__ = [
    "SUPERVISOR_SYSTEM_PROMPT",
    "AgentUsage",
    "AgentUsageCallbackHandler",
    "StubChatModel",
    "build_agent",
    "build_chat_model",
    "build_knowledge_subagent",
    "build_research_tool",
    "final_answer",
    "get_agent",
    "reset_agent",
    "search_knowledge_base",
    "set_agent",
]
