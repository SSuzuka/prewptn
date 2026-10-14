"""Тесты идемпотентности, ретраев и dead-letter очереди."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from aicore.db.models import IdempotencyKey, TaskStatus
from aicore.db.session import Database
from aicore.tasks.ping import PermanentTaskError, TransientTaskError, process_task
from aicore.tasks.service import TaskService


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_same_key_returns_same_task(self, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            first, created = await svc.create_task(
                task_type="ping", payload={}, idempotency_key="k1"
            )
            second, replayed = await svc.create_task(
                task_type="ping", payload={}, idempotency_key="k1"
            )
        assert created is True
        assert replayed is False
        assert first.id == second.id

    @pytest.mark.asyncio
    async def test_no_duplicate_rows(self, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            for _ in range(3):
                await svc.create_task(task_type="ping", payload={}, idempotency_key="same")

            tasks = await svc.list_tasks()
            keys = await session.execute(select(func.count()).select_from(IdempotencyKey))
        assert len(tasks) == 1
        assert keys.scalar_one() == 1

    @pytest.mark.asyncio
    async def test_different_keys_create_separate_tasks(self, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            _, a = await svc.create_task(task_type="ping", payload={}, idempotency_key="a")
            _, b = await svc.create_task(task_type="ping", payload={}, idempotency_key="b")
        assert a is True and b is True

    @pytest.mark.asyncio
    async def test_cascade_delete_frees_key(self, db: Database) -> None:
        """ON DELETE CASCADE: удаление задачи удаляет ключ (иначе ключ утечёт)."""
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="ping", payload={}, idempotency_key="cascade")
            await session.flush()

            # Ключ обязан ссылаться на задачу, иначе CASCADE не на что вешать.
            linked = (
                await session.execute(
                    select(IdempotencyKey.task_id).where(IdempotencyKey.key == "cascade")
                )
            ).scalar_one_or_none()
            assert linked == task.id

            await session.delete(task)
            await session.flush()

            count = await session.execute(select(func.count()).select_from(IdempotencyKey))
            assert count.scalar_one() == 0

    @pytest.mark.asyncio
    async def test_foreign_keys_pragma_enabled(self, db: Database) -> None:
        """Регресс: SQLite игнорирует FK без PRAGMA, и каскады молча ломаются."""
        async with db.session() as session:
            result = await session.execute(text("PRAGMA foreign_keys"))
        assert result.scalar_one() == 1


class TestRetriesAndDeadLetter:
    @pytest.mark.asyncio
    async def test_failure_returns_to_pending_while_attempts_left(self, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="ping", payload={}, max_attempts=3)
            await svc.mark_running(task)
            await svc.mark_failed(task, "transient")
            assert task.status is TaskStatus.PENDING
            assert task.can_retry is True

    @pytest.mark.asyncio
    async def test_exhausted_goes_to_dead_letter(self, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="ping", payload={}, max_attempts=2)

            for _ in range(2):
                await svc.mark_running(task)
                await svc.mark_failed(task, "boom")

            assert task.status is TaskStatus.DEAD_LETTER
            assert [t.id for t in await svc.list_dead_letters()] == [task.id]

    @pytest.mark.asyncio
    async def test_reset_for_retry(self, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            task, _ = await svc.create_task(task_type="ping", payload={}, max_attempts=1)
            await svc.mark_running(task)
            await svc.mark_failed(task, "boom")
            assert task.status is TaskStatus.DEAD_LETTER

            await svc.reset_for_retry(task)
            assert task.status is TaskStatus.PENDING
            assert task.attempts == 0
            assert task.error is None


class TestProcessTask:
    @pytest.mark.asyncio
    async def test_success_path(self, db: Database, use_db: Database) -> None:
        async with db.session() as session:
            task, _ = await TaskService(session).create_task(
                task_type="ping", payload={"echo": "hi"}
            )

        await process_task(str(task.id))

        async with db.session() as session:
            reloaded = await TaskService(session).get_task(task.id)
        assert reloaded is not None
        assert reloaded.status is TaskStatus.SUCCEEDED
        assert reloaded.result is not None

    @pytest.mark.asyncio
    async def test_attempts_survive_rollback(self, db: Database, use_db: Database) -> None:
        """Регресс: счётчик попыток должен переживать бросок исключения в taskiq."""
        async with db.session() as session:
            task, _ = await TaskService(session).create_task(
                task_type="ping", payload={}, max_attempts=3
            )

        for _ in range(3):
            try:
                await process_task(str(task.id), fail_times=2)
                break
            except TransientTaskError:
                continue

        async with db.session() as session:
            reloaded = await TaskService(session).get_task(task.id)
        assert reloaded is not None
        assert reloaded.status is TaskStatus.SUCCEEDED
        assert reloaded.attempts == 3

    @pytest.mark.asyncio
    async def test_exhausted_transient_becomes_dead_letter(
        self, db: Database, use_db: Database
    ) -> None:
        async with db.session() as session:
            task, _ = await TaskService(session).create_task(
                task_type="ping", payload={}, max_attempts=1
            )

        await process_task(str(task.id), fail_times=99)

        async with db.session() as session:
            reloaded = await TaskService(session).get_task(task.id)
        assert reloaded is not None
        assert reloaded.status is TaskStatus.DEAD_LETTER

    @pytest.mark.asyncio
    async def test_missing_task_raises_permanent(self, db: Database, use_db: Database) -> None:
        with pytest.raises(PermanentTaskError):
            await process_task(str(uuid4()))

    @pytest.mark.asyncio
    async def test_invalid_uuid_rejected(self, db: Database, use_db: Database) -> None:
        with pytest.raises(PermanentTaskError):
            await process_task("not-a-uuid")

    @pytest.mark.asyncio
    async def test_timeout_marks_task_failed(
        self, db: Database, use_db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """task_timeout_seconds должен реально ограничивать выполнение."""
        # Крошечный таймаут вместо ожидания дефолтных 120с.
        monkeypatch.setenv("TASK_TIMEOUT_SECONDS", "0.05")
        from aicore.config import reset_settings_cache

        reset_settings_cache()

        async with db.session() as session:
            task, _ = await TaskService(session).create_task(task_type="ping", payload={})

        with pytest.raises(TimeoutError):
            await process_task(str(task.id), delay_seconds=5.0)

        reset_settings_cache()

        async with db.session() as session:
            reloaded = await TaskService(session).get_task(task.id)
        assert reloaded is not None
        assert "TimeoutError" in (reloaded.error or "")

    @pytest.mark.asyncio
    async def test_timeout_uses_configured_value(self, db: Database, use_db: Database) -> None:
        """Мельче нуля таймаут не бывает — но значение из конфига применяется."""
        from aicore.config import Settings, get_settings

        assert Settings().task_timeout_seconds > 0
        assert get_settings().task_timeout_seconds > 0


class TestDocumentModel:
    @pytest.mark.asyncio
    async def test_document_defaults(self, db: Database) -> None:
        from aicore.db.models import Document, DocumentStatus

        async with db.session() as session:
            session.add(Document(title="T", content_hash="h" * 64))
            await session.flush()
            stored = (await session.execute(select(Document))).scalar_one()

        assert stored.status is DocumentStatus.PENDING
        assert stored.chunk_count == 0

    @pytest.mark.asyncio
    async def test_content_hash_is_unique(self, db: Database) -> None:
        from aicore.db.models import Document

        async with db.session() as session:
            session.add(Document(title="A", content_hash="x" * 64))
            await session.flush()
            session.add(Document(title="B", content_hash="x" * 64))
            with pytest.raises(IntegrityError):
                await session.flush()
            # После ошибки flush сессия в состоянии PendingRollback —
            # нужен явный rollback, иначе упадёт commit() в контекстном менеджере.
            await session.rollback()
