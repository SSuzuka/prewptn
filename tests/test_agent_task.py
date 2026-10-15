"""Тесты агентной задачи (run_agent) на БД — офлайн."""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

from aicore.agents import reset_agent
from aicore.config import Settings
from aicore.db.models import TaskStatus
from aicore.db.session import Database
from aicore.rag import chunk_text, reset_retriever, set_retriever
from aicore.rag.retriever import Retriever
from aicore.rag.store import VectorStore
from aicore.rag.stub import StubEmbeddings
from aicore.tasks.agent_run import run_agent
from aicore.tasks.service import TaskService

DOC = "The redis stream broker supports acknowledge and redelivery if a worker dies."


async def _use_retriever() -> Retriever:
    cfg = Settings(vector_backend="memory", embedding_backend="stub", embedding_dim=384)
    retriever = Retriever(
        embeddings=StubEmbeddings(settings=cfg),
        store=VectorStore(
            client=AsyncQdrantClient(":memory:"), collection="agent_task", dimension=384
        ),
        settings=cfg,
    )
    set_retriever(retriever)
    await retriever.index(
        document_id="broker",
        chunks=chunk_text(DOC, chunk_size=200, chunk_overlap=20),
        title="broker",
    )
    return retriever


class TestRunAgentTask:
    @pytest.mark.asyncio
    async def test_succeeds_with_metrics(self, db: Database, use_db: Database) -> None:
        retriever = await _use_retriever()
        reset_agent()
        try:
            async with db.session() as session:
                task, _ = await TaskService(session).create_task(
                    task_type="run_agent",
                    payload={"question": "how does the broker handle redelivery"},
                )

            result = await run_agent(str(task.id))
            assert result["answer"]
            assert result["model_calls"] >= 1
            assert result["tool_calls"] >= 1
            assert result["prompt_tokens"] > 0

            async with db.session() as session:
                reloaded = await TaskService(session).get_task(task.id)
            assert reloaded is not None
            assert reloaded.status is TaskStatus.SUCCEEDED
            assert reloaded.cost_usd is not None
        finally:
            await retriever.aclose()
            reset_retriever()
            reset_agent()

    @pytest.mark.asyncio
    async def test_malformed_task_id_reports_failed(self, db: Database, use_db: Database) -> None:
        result = await run_agent("not-a-uuid")
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_missing_task_reports_failed(self, db: Database, use_db: Database) -> None:
        result = await run_agent("00000000-0000-0000-0000-000000000000")
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_empty_question_goes_to_dead_letter(self, db: Database, use_db: Database) -> None:
        async with db.session() as session:
            task, _ = await TaskService(session).create_task(
                task_type="run_agent",
                payload={"question": "   "},
            )

        result = await run_agent(str(task.id))
        assert result["status"] == "dead_letter"

        async with db.session() as session:
            reloaded = await TaskService(session).get_task(task.id)
        assert reloaded is not None
        assert reloaded.status is TaskStatus.DEAD_LETTER
