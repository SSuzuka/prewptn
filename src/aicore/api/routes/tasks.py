"""Роуты задач: создание (идемпотентно), чтение, ручной retry и SSE-поток."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask

from aicore.api.deps import rate_limit
from aicore.api.schemas import TaskCreateRequest, TaskView
from aicore.config import get_settings
from aicore.db.session import get_session
from aicore.logging import get_logger
from aicore.streaming.events import stream_task_events
from aicore.tasks.agent_run import enqueue_run_agent
from aicore.tasks.llm import enqueue_answer_question
from aicore.tasks.rag_index import enqueue_index_document
from aicore.tasks.service import IdempotencyConflictError, TaskService

logger = get_logger(__name__)
router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]
IdempotencyKey = Annotated[str | None, Header(alias="Idempotency-Key", max_length=255)]

# Ограничение одновременных SSE-потоков: каждый поток раз в 0.25 с открывает
# сессию и делает SELECT. Без лимита один клиент может выесть пул БД.
# Счётчик меняется без await, поэтому в одном event loop операции атомарны;
# отдельный lock не нужен и только привязал бы модуль к конкретному loop.
_active_streams = 0


async def _acquire_stream_slot() -> bool:
    global _active_streams
    if _active_streams >= get_settings().sse_max_concurrent_streams:
        return False
    _active_streams += 1
    return True


async def _release_stream_slot() -> None:
    global _active_streams
    if _active_streams > 0:
        _active_streams -= 1


def reset_stream_slots() -> None:
    """Сбросить счётчик (тесты)."""
    global _active_streams
    _active_streams = 0


class _StreamSlot:
    """Idempotent release: generator finally and response background may both run."""

    def __init__(self) -> None:
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await _release_stream_slot()


async def _limited_events(task_id: str, slot: _StreamSlot) -> AsyncIterator[dict[str, str]]:
    """Прокинуть события, гарантированно освободив слот при закрытии."""
    try:
        async for event in stream_task_events(task_id):
            yield event
    finally:
        await slot.release()


# Диспетчеризация ручного retry по типу задачи: каждый тип перезапускается
# своим хендлером, иначе index_document ушёл бы в LLM-хендлер (и наоборот).
_ENQUEUERS: dict[str, Callable[[str], Awaitable[bool]]] = {
    "answer_question": enqueue_answer_question,
    "index_document": enqueue_index_document,
    "run_agent": enqueue_run_agent,
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
    try:
        task, created = await svc.create_task(
            task_type="answer_question",
            payload=body.model_dump(exclude_none=True),
            idempotency_key=idempotency_key,
            # Отдельный скоуп на эндпоинт: иначе один и тот же `Idempotency-Key`,
            # отправленный на /v1/tasks и /v1/agents, вернул бы задачу чужого типа.
            idempotency_scope="tasks",
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if created:
        # Коммитим ДО постановки в очередь: задача читает себя отдельной сессией
        # и должна быть уже видна.
        await session.commit()
        await enqueue_answer_question(str(task.id))
        await session.refresh(task)
    else:
        logger.info("task.idempotent_replay", task_id=str(task.id))
    return TaskView.from_task(task)


@router.get(
    "/tasks/{task_id}",
    dependencies=[Depends(rate_limit)],
    summary="Статус задачи с метриками",
)
async def get_task(task_id: str, session: SessionDep) -> TaskView:
    _require_uuid(task_id)
    task = await TaskService(session).get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskView.from_task(task)


@router.post(
    "/tasks/{task_id}/retry",
    dependencies=[Depends(rate_limit)],
    summary="Ручной перезапуск упавшей задачи",
)
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


@router.get(
    "/tasks/{task_id}/stream",
    dependencies=[Depends(rate_limit)],
    summary="SSE-поток событий выполнения",
)
async def stream_task(task_id: str) -> EventSourceResponse:
    _require_uuid(task_id)
    slot = _StreamSlot()
    if not await _acquire_stream_slot():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="too many concurrent streams",
        )
    # BackgroundTask runs even when the client disconnects before the generator's
    # first __anext__ (an unstarted async generator's aclose() does not run
    # finally), so the slot cannot leak.
    return EventSourceResponse(
        _limited_events(task_id, slot),
        background=BackgroundTask(slot.release),
    )
