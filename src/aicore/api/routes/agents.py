"""Роут агентов: запуск supervisor-графа (subagents + tools) как задачи.

Тонкая обёртка над пайплайном задач: создаёт `run_agent`-задачу (идемпотентно по
`Idempotency-Key`) и ставит её в брокер. Статус, метрики стоимости и SSE-поток
читаются через общие `/v1/tasks/{id}` и `/v1/tasks/{id}/stream`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from aicore.api.deps import rate_limit
from aicore.api.schemas import AgentRunRequest, TaskView
from aicore.db.session import get_session
from aicore.logging import get_logger
from aicore.tasks.agent_run import enqueue_run_agent
from aicore.tasks.service import IdempotencyConflictError, TaskService

logger = get_logger(__name__)
router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key", max_length=255)]


@router.post(
    "/agents",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit)],
    summary="Запустить агента (supervisor + subagents)",
)
async def run_agent(
    body: AgentRunRequest,
    session: SessionDep,
    idempotency_key: IdempotencyKey = None,
) -> TaskView:
    svc = TaskService(session)
    payload: dict[str, object] = {"question": body.question}
    if body.thread_id:
        payload["thread_id"] = body.thread_id

    try:
        task, created = await svc.create_task(
            task_type="run_agent",
            payload=payload,
            idempotency_key=idempotency_key,
            # Отдельный скоуп на эндпоинт: иначе один и тот же `Idempotency-Key`,
            # отправленный на /v1/tasks и /v1/agents, вернул бы задачу чужого типа.
            idempotency_scope="agents",
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if created:
        # Коммитим ДО постановки в очередь: задача читает себя отдельной сессией
        # и должна быть уже видна.
        await session.commit()
        await enqueue_run_agent(str(task.id))
        await session.refresh(task)
    else:
        logger.info("task.idempotent_replay", task_id=str(task.id))
    return TaskView.from_task(task)
