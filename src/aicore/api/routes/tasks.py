"""Роуты задач: создание (идемпотентно), чтение, ручной retry и SSE-поток."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from aicore.api.deps import rate_limit
from aicore.api.schemas import TaskCreateRequest, TaskView
from aicore.db.session import get_session
from aicore.logging import get_logger
from aicore.streaming.events import stream_task_events
from aicore.tasks.llm import enqueue_answer_question
from aicore.tasks.rag_index import enqueue_index_document
from aicore.tasks.service import TaskService

logger = get_logger(__name__)
router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key", max_length=255)]

# Диспетчеризация ручного retry по типу задачи: каждый тип перезапускается
# своим хендлером, иначе index_document ушёл бы в LLM-хендлер (и наоборот).
_ENQUEUERS: dict[str, Callable[[str], Awaitable[None]]] = {
    "answer_question": enqueue_answer_question,
    "index_document": enqueue_index_document,
}


def _require_uuid(task_id: str) -> None:
    try:
        UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="task_id must be a valid UUID") from None


@router.post(
    "/tasks",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit)],
    summary="Поставить AI-задачу (идемпотентно по Idempotency-Key)",
)
async def create_task(
    body: TaskCreateRequest,
    session: SessionDep,
    idempotency_key: IdempotencyKey = None,
) -> TaskView:
    svc = TaskService(session)
    task, created = await svc.create_task(
        task_type="answer_question",
        payload=body.model_dump(exclude_none=True),
        idempotency_key=idempotency_key,
    )
    if created:
        # Коммитим ДО постановки в очередь: задача читает себя отдельной сессией
        # и должна быть уже видна.
        await session.commit()
        await enqueue_answer_question(str(task.id))
        await session.refresh(task)
    else:
        logger.info("task.idempotent_replay", task_id=str(task.id))
    return TaskView.from_task(task)


@router.get("/tasks/{task_id}", summary="Статус задачи с метриками")
async def get_task(task_id: str, session: SessionDep) -> TaskView:
    _require_uuid(task_id)
    task = await TaskService(session).get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskView.from_task(task)


@router.post("/tasks/{task_id}/retry", summary="Ручной перезапуск упавшей задачи")
async def retry_task(task_id: str, session: SessionDep) -> TaskView:
    _require_uuid(task_id)
    svc = TaskService(session)
    task = await svc.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    enqueue = _ENQUEUERS.get(task.task_type)
    if enqueue is None:
        raise HTTPException(
            status_code=409, detail=f"task type {task.task_type!r} cannot be retried"
        )
    try:
        await svc.reset_for_retry(task)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    await session.commit()
    await enqueue(str(task.id))
    await session.refresh(task)
    return TaskView.from_task(task)


@router.get("/tasks/{task_id}/stream", summary="SSE-поток событий выполнения")
async def stream_task(task_id: str) -> EventSourceResponse:
    _require_uuid(task_id)
    return EventSourceResponse(stream_task_events(task_id))
