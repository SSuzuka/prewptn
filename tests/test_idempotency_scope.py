"""Регресс-тесты на изоляцию идемпотентности по scope.

Баг, который здесь ловится: `get_task_by_idempotency_key` принимал `scope`,
но не использовал его в запросе, а `TaskRun.idempotency_key` был уникален
глобально. В результате два разных тенанта с одним ключом схлопывались в
одну задачу — ключ одного клиента отдавал чужую задачу.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from aicore.db.models import IdempotencyKey
from aicore.db.session import Database
from aicore.tasks.service import TaskService


class TestScopeIsolation:
    @pytest.mark.asyncio
    async def test_same_key_different_scope_creates_separate_tasks(self, db: Database) -> None:
        """Разные тененты с одним ключом НЕ должны делить задачу."""
        async with db.session() as session:
            svc = TaskService(session)
            task_a, created_a = await svc.create_task(
                task_type="ping", payload={"tenant": "a"}, idempotency_key="shared-key"
            )
            task_b, created_b = await svc.create_task(
                task_type="ping",
                payload={"tenant": "b"},
                idempotency_key="shared-key",
                idempotency_scope="tenant-b",
            )

        assert created_a is True
        assert created_b is True, "второй тенант должен получить СВОЮ задачу"
        assert task_a.id != task_b.id

    @pytest.mark.asyncio
    async def test_same_key_same_scope_replays(self, db: Database) -> None:
        """Внутри одного скоупа повтор возвращает ту же задачу."""
        async with db.session() as session:
            svc = TaskService(session)
            first, created = await svc.create_task(
                task_type="ping", payload={}, idempotency_key="k", idempotency_scope="tenant-a"
            )
            second, replayed = await svc.create_task(
                task_type="ping", payload={}, idempotency_key="k", idempotency_scope="tenant-a"
            )

        assert created is True
        assert replayed is False
        assert first.id == second.id

    @pytest.mark.asyncio
    async def test_scope_is_actually_used_in_lookup(self, db: Database) -> None:
        """Поиск по ключу в чужом скоупе должен возвращать None."""
        async with db.session() as session:
            svc = TaskService(session)
            await svc.create_task(
                task_type="ping", payload={}, idempotency_key="k", idempotency_scope="tenant-a"
            )

            found_in_other = await svc.get_task_by_idempotency_key("k", scope="tenant-b")
            found_in_own = await svc.get_task_by_idempotency_key("k", scope="tenant-a")

        assert found_in_other is None
        assert found_in_own is not None

    @pytest.mark.asyncio
    async def test_no_duplicate_rows_across_scopes(self, db: Database) -> None:
        async with db.session() as session:
            svc = TaskService(session)
            await svc.create_task(
                task_type="ping", payload={}, idempotency_key="k", idempotency_scope="a"
            )
            await svc.create_task(
                task_type="ping", payload={}, idempotency_key="k", idempotency_scope="b"
            )

            tasks = len(await svc.list_tasks())
            keys = await session.execute(select(func.count()).select_from(IdempotencyKey))

        assert tasks == 2
        assert keys.scalar_one() == 2
