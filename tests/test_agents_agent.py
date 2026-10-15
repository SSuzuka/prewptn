"""Тесты supervisor-графа: subagent-as-tool, grounding, память — офлайн."""

from __future__ import annotations

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from aicore.agents import build_agent, reset_agent, set_agent
from aicore.config import Settings
from aicore.rag import chunk_text, reset_retriever, set_retriever
from aicore.rag.retriever import Retriever
from aicore.rag.store import VectorStore
from aicore.rag.stub import StubEmbeddings

DOC = "The redis stream broker supports acknowledge and redelivery if a worker dies."


@pytest_asyncio.fixture
async def seeded_retriever():
    cfg = Settings(vector_backend="memory", embedding_backend="stub", embedding_dim=384)
    retriever = Retriever(
        embeddings=StubEmbeddings(settings=cfg),
        store=VectorStore(
            client=AsyncQdrantClient(":memory:"), collection="agent_seeded", dimension=384
        ),
        settings=cfg,
    )
    set_retriever(retriever)
    await retriever.index(
        document_id="broker",
        chunks=chunk_text(DOC, chunk_size=200, chunk_overlap=20),
        title="broker",
    )
    yield retriever
    await retriever.aclose()
    reset_retriever()
    reset_agent()


def _tool_names(state: dict) -> list[str]:
    return [
        call["name"]
        for message in state["messages"]
        if getattr(message, "tool_calls", None)
        for call in message.tool_calls
    ]


class TestSupervisorAgent:
    @pytest.mark.asyncio
    async def test_delegates_to_subagent_and_grounds(self, seeded_retriever: Retriever) -> None:
        agent = build_agent(Settings(agent_backend="stub", rag_retrieve_limit=1))
        state = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "how does the broker handle redelivery"}]},
            config={"configurable": {"thread_id": "agent-1"}},
        )
        assert "research_knowledge_base" in _tool_names(state)
        answer = state["messages"][-1].content.lower()
        assert "redelivery" in answer
        assert "broker" in answer

    @pytest.mark.asyncio
    async def test_shared_thread_keeps_history(self, seeded_retriever: Retriever) -> None:
        agent = build_agent(Settings(agent_backend="stub", rag_retrieve_limit=1))
        set_agent(agent)
        config = {"configurable": {"thread_id": "agent-thread"}}
        first = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "redis redelivery worker"}]}, config=config
        )
        second = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "redis redelivery worker"}]}, config=config
        )
        # С чекпойнтером второй прогон видит историю первого.
        assert len(second["messages"]) > len(first["messages"])
