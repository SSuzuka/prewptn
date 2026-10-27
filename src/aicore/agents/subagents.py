"""Knowledge-base research subagent, exposed to the supervisor as a tool.

The 2026-correct pattern for multi-agent systems in LangChain 1.x is **subagents
as tools**: a supervisor ``create_agent`` delegates to a specialized subagent
through an ordinary tool call. The older ``langgraph-supervisor`` /
``create_supervisor`` package is no longer maintained and is intentionally not
used here.

Keeping the subagent async and wrapping it via :func:`build_research_tool` means
the supervisor sees a normal tool, while the retrieval loop stays encapsulated in
its own graph with its own thread.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.base import BaseCheckpointSaver

from aicore.agents.tools import search_knowledge_base

KNOWLEDGE_SYSTEM_PROMPT = (
    "You are a knowledge-base research subagent. Answer the question using ONLY "
    "the passages returned by the search_knowledge_base tool, and cite the "
    "document ids you used. If the tool finds nothing relevant, say so plainly "
    "instead of guessing."
)


def build_knowledge_subagent(
    model: BaseChatModel, *, checkpointer: BaseCheckpointSaver | None = None
) -> Runnable:
    """Build the RAG research subagent (model + retrieval tool).

    The subagent is invoked once per delegation, so it carries no checkpointer
    of its own (a shared one would retain a thread per call forever).
    """
    return create_agent(
        model,
        tools=[search_knowledge_base],
        system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        name="knowledge_subagent",
    )


def build_research_tool(subagent: Runnable) -> BaseTool:
    """Wrap a subagent graph as a tool the supervisor can call."""

    @tool
    async def research_knowledge_base(query: str, config: RunnableConfig) -> str:
        """Delegate a factual question to the knowledge-base research subagent.

        The subagent runs its own retrieval loop and returns a grounded answer
        grounded in the indexed documents.
        """
        # Forward callbacks (usage/metrics handlers) into the nested run; the
        # subagent keeps its own fresh config, not the supervisor's thread, but
        # it does inherit the caller's step budget so a raised recursion_limit
        # is honored by the nested graph too.
        sub_config: RunnableConfig = {"callbacks": (config or {}).get("callbacks")}
        recursion_limit = (config or {}).get("recursion_limit")
        if recursion_limit is not None:
            sub_config["recursion_limit"] = recursion_limit
        result = await subagent.ainvoke(
            {"messages": [{"role": "user", "content": query}]},
            config=sub_config,
        )
        messages = result.get("messages") or []
        if not messages:
            return "The research subagent returned no answer."
        last: BaseMessage = messages[-1]
        content = last.content
        return content if isinstance(content, str) else str(content)

    return research_knowledge_base
