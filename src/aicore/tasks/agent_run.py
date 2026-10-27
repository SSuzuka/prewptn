"""Агентная задача: прогоняет вопрос через граф агента (supervisor + subagents).

Симметрично `tasks/llm.py`: фаза БД → фаза графа (без удержания соединения) →
фаза БД. Токены/стоимость суммируются по всем шагам модели — и супервизора, и
подагента — через `AgentUsageCallbackHandler`, который прокидывается в граф
(включая вложенный прогон подагента) и пишется в `TaskRun`, поэтому `/metrics`
и API показывают реальную цену агентного прогона.

Граф исполняется через `ainvoke`: инструменты асинхронные, а sync-путь LangGraph
на них падает (`StructuredTool does not support sync invocation`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from langgraph.errors import GraphRecursionError

from aicore.agents import get_agent
from aicore.agents.usage import AgentUsage, AgentUsageCallbackHandler, final_answer
from aicore.broker import broker, ensure_broker_started
from aicore.config import get_settings
from aicore.db.session import get_database
from aicore.logging import get_logger
from aicore.metrics import record_enqueue_failed
from aicore.tasks._common import load_task, mark_task_enqueue_failed, mark_task_failed
from aicore.tasks.service import TaskService

logger = get_logger(__name__)


@broker.task(retry_on_error=True)
async def run_agent(task_id: str) -> dict[str, Any]:
    """Выполнить агентный прогон для задачи и сохранить результат + метрики."""
    settings = get_settings()
    try:
        async with asyncio.timeout(settings.task_timeout_seconds):
            return await _run(task_id)
    except TimeoutError:
        logger.error("task.timeout", task_id=task_id, timeout=settings.task_timeout_seconds)
        await mark_task_failed(task_id, "TimeoutError: task exceeded deadline", retryable=True)
        raise


async def _run(task_id: str) -> dict[str, Any]:
    cfg = get_settings()
    db = get_database()
    await db.ensure_ready()

    # Фаза 1: загрузить задачу, атомарно заклеймить её, отпустить соединение.
    async with db.session() as session:
        svc = TaskService(session)
        task = await load_task(svc, task_id)
        if task is None:
            return {"status": "failed", "task_id": task_id, "error": "task not found"}
        if not await svc.mark_running(task):
            # Задачу уже выполняет другой воркер или она завершена: не перезаписываем
            # терминальный статус и не платим за граф повторно.
            return {"status": "skipped", "task_id": task_id, "error": "task is not pending"}
        payload = dict(task.payload or {})
        attempts = task.attempts

    question = str(payload.get("question", "")).strip()
    thread_id = str(payload.get("thread_id") or task_id)
    if not question:
        # API-слой отсекает пустой вопрос (min_length=1), но прямой вызов
        # create_task(payload={"question": ""}) дошёл бы сюда: пустой ввод
        # неисправим ретраем, поэтому сразу dead-letter, а не прогон графа.
        # Проверка идёт ПОСЛЕ claim (см. фазу 1), поэтому неисправимый ввод
        # стоит одной попытки (attempts=1, PENDING -> RUNNING -> DEAD_LETTER).
        # Это осознанный размен: терминальный статус пишется только по уже
        # заклеймлённой задаче, а не поверх параллельного воркера. Так же в
        # tasks/llm.py — ради консистентности хендлеров.
        await mark_task_failed(task_id, "ValueError: question must not be empty", retryable=False)
        return {"status": "dead_letter", "task_id": task_id, "error": "empty question"}

    # Фаза 2: прогон графа агента без удержания БД-соединения. Хендлер в
    # callbacks видит все вызовы модели/инструментов, включая подагента.
    graph = get_agent()

    # Ретрай того же task_id иначе реплеит историю чекпойнта упавшей попытки:
    # удвоенный контекст и stale tool-calls, за которые платишь повторно.
    # Сбрасываем только собственный (default) тред задачи — клиентский
    # thread_id это настоящая беседа, и она обязана пережить ретрай.
    if attempts > 1 and "thread_id" not in payload:
        checkpointer = getattr(graph, "checkpointer", None)
        if checkpointer is not None:
            await checkpointer.adelete_thread(thread_id)

    handler = AgentUsageCallbackHandler()
    try:
        state = await graph.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": cfg.agent_recursion_limit,
                "callbacks": [handler],
            },
        )
    except GraphRecursionError as exc:
        error = f"{type(exc).__name__}: {exc}"
        await mark_task_failed(task_id, error, retryable=False)
        return {"status": "dead_letter", "task_id": task_id, "error": error}
    except Exception as exc:  # noqa: BLE001 — transient: пусть брокер ретраит
        await mark_task_failed(task_id, f"{type(exc).__name__}: {exc}", retryable=True)
        raise

    messages = state.get("messages") or []
    result = _summarize(messages, usage=handler.usage, model=cfg.agent_model, thread_id=thread_id)

    if not result["answer"].strip():
        # Супервизор завершился без текстового ответа (например, только
        # tool-calls при исчерпанном лимите). Пустой ответ бесполезен клиенту,
        # а причина часто транзиентна — отдаём брокеру на ретрай, а не пишем
        # SUCCEEDED с `answer=""`.
        error = "ValueError: empty final answer"
        await mark_task_failed(task_id, error, retryable=True)
        raise ValueError(error)

    # Фаза 3: сохранить ответ и суммарные метрики.
    async with db.session() as session:
        svc = TaskService(session)
        task = await load_task(svc, task_id)
        if task is None:
            return {"status": "failed", "task_id": task_id, "error": "task not found"}
        await svc.mark_succeeded(
            task,
            result=result,
            prompt_tokens=result["prompt_tokens"],
            completion_tokens=result["completion_tokens"],
            cost_usd=result["cost_usd"],
        )
    return result


def _summarize(
    messages: Sequence[Any], *, usage: AgentUsage, model: str, thread_id: str
) -> dict[str, Any]:
    """Финальный ответ + суммарные токены/стоимость/вызовы (супервизор + подагент)."""
    return {
        "answer": final_answer(messages),
        "model": model,
        "thread_id": thread_id,
        "model_calls": usage.model_calls,
        "tool_calls": usage.tool_calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": usage.cost_usd(),
    }


async def enqueue_run_agent(task_id: str, *, mark_failed_on_error: bool = True) -> bool:
    """Поставить агентную задачу в брокер (dev — inline, prod — в очередь).

    `ensure_broker_started` внутри try: недоступный брокер не должен оставлять
    задачу в PENDING навсегда. Returns ``False`` on transport failure so the
    sweeper does not count a failed re-enqueue as reclaimed.
    """
    try:
        await ensure_broker_started()
        await run_agent.kiq(task_id)
    except Exception as exc:  # noqa: BLE001 — фиксируем в БД, а не только в логе
        logger.exception("task.enqueue_failed", task_id=task_id)
        record_enqueue_failed("run_agent")
        if mark_failed_on_error:
            await mark_task_enqueue_failed(task_id, f"{type(exc).__name__}: {exc}")
        return False
    return True
