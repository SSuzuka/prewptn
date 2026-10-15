"""Тесты задачи индексации: статусы документа, retry vs dead-letter. Офлайн."""

from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

from aicore.config import Settings
from aicore.db.models import DocumentStatus, TaskStatus
from aicore.db.session import Database
from aicore.rag import EmbeddingPermanentError, EmbeddingTransientError
from aicore.rag.base import EmbeddingRequest
from aicore.rag.documents import DocumentService
from aicore.rag.retriever import Retriever
from aicore.rag.store import VectorStore
from aicore.rag.stub import StubEmbeddings
from aicore.tasks.rag_index import index_document
from aicore.tasks.service import TaskService
from aicore.telemetry import reset_telemetry


class _PermanentEmbeddings:
    model = "boom"
    dimension = 384

    async def embed(self, request: EmbeddingRequest) -> None:
        raise EmbeddingPermanentError("no such model")

    async def aclose(self) -> None:
        return None


class _TransientEmbeddings:
    model = "flaky"
    dimension = 384

    async def embed(self, request: EmbeddingRequest) -> None:
        raise EmbeddingTransientError("connection reset")

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch):
    from aicore.rag import reset_retriever

    reset_retriever()
    reset_telemetry()
    yield
    reset_retriever()
    reset_telemetry()


def _memory_retriever(embeddings=None) -> Retriever:
    cfg = Settings(vector_backend="memory", embedding_backend="stub", embedding_dim=384)
    return Retriever(
        embeddings=embeddings or StubEmbeddings(settings=cfg),
        store=VectorStore(client=AsyncQdrantClient(":memory:"), collection="t", dimension=384),
        settings=cfg,
    )


def _set_retriever(retriever: Retriever) -> None:
    from aicore.rag import set_retriever

    set_retriever(retriever)


async def _make(db: Database, content: str, *, document_id: str | None = None):
    async with db.session() as session:
        if document_id is None:
            document, _ = await DocumentService(session).create_document(
                title="doc", content=content
            )
            document_id = str(document.id)
        task, _ = await TaskService(session).create_task(
            task_type="index_document",
            payload={"document_id": document_id, "title": "doc", "content": content},
        )
    return document_id, task.id


async def _load(db: Database, document_id: str, task_id):
    async with db.session() as session:
        document = await DocumentService(session).get_document(document_id)
        task = await TaskService(session).get_task(task_id)
    return document, task


class TestIndexDocument:
    @pytest.mark.asyncio
    async def test_success_indexes_and_updates_status(self, db: Database, use_db: Database) -> None:
        retriever = _memory_retriever()
        _set_retriever(retriever)
        document_id, task_id = await _make(db, "some indexable content here")

        result = await index_document(str(task_id))
        assert result["chunks"] >= 1

        document, task = await _load(db, document_id, task_id)
        assert document is not None and document.status is DocumentStatus.INDEXED
        assert document.chunk_count >= 1
        assert task is not None and task.status is TaskStatus.SUCCEEDED
        assert await retriever.store.count() >= 1
        await retriever.aclose()

    @pytest.mark.asyncio
    async def test_permanent_error_dead_letters_document(
        self, db: Database, use_db: Database
    ) -> None:
        _set_retriever(_memory_retriever(embeddings=_PermanentEmbeddings()))
        document_id, task_id = await _make(db, "content")

        result = await index_document(str(task_id))
        assert result["status"] == "dead_letter"

        document, task = await _load(db, document_id, task_id)
        assert document is not None and document.status is DocumentStatus.FAILED
        assert task is not None and task.status is TaskStatus.DEAD_LETTER

    @pytest.mark.asyncio
    async def test_transient_error_returns_to_pending(self, db: Database, use_db: Database) -> None:
        _set_retriever(_memory_retriever(embeddings=_TransientEmbeddings()))
        document_id, task_id = await _make(db, "content")

        with pytest.raises(EmbeddingTransientError):
            await index_document(str(task_id))

        _document, task = await _load(db, document_id, task_id)
        assert task is not None and task.status is TaskStatus.PENDING
        assert task.can_retry is True

    @pytest.mark.asyncio
    async def test_missing_document_dead_letters(self, db: Database, use_db: Database) -> None:
        _set_retriever(_memory_retriever())
        missing = "00000000-0000-0000-0000-000000000000"
        _document_id, task_id = await _make(db, "x", document_id=missing)

        result = await index_document(str(task_id))
        assert result["status"] == "dead_letter"

    @pytest.mark.asyncio
    async def test_malformed_document_id_dead_letters(self, db: Database, use_db: Database) -> None:
        # Битый (не-UUID) document_id в payload не должен уводить задачу в
        # transient-retry: get_document возвращает None, задача — dead_letter.
        _set_retriever(_memory_retriever())
        _document_id, task_id = await _make(db, "x", document_id="not-a-uuid")

        result = await index_document(str(task_id))
        assert result["status"] == "dead_letter"

        _document, task = await _load(db, "not-a-uuid", task_id)
        assert task is not None and task.status is TaskStatus.DEAD_LETTER

    @pytest.mark.asyncio
    async def test_missing_task_is_noop(self, db: Database, use_db: Database) -> None:
        _set_retriever(_memory_retriever())
        result = await index_document("00000000-0000-0000-0000-000000000000")
        assert result["status"] == "failed"
