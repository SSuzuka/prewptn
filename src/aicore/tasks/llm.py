"""LLM-задача: прогоняет промпт через сконфигурированный LLM-клиент.

Связывает LLM-слой (этап 3) с пайплайном задач: один вызов сохраняет токены,
latency и стоимость в `TaskRun`, поэтому API и `/metrics` показывают реальные
числа. Временные ошибки LLM ретраит брокер; постоянные (4xx) сразу уходят в
dead-letter.

БД-сессия не удерживается во время сетевого вызова LLM: фазы «отметить running»
и «сохранить результат» открывают короткие сессии, а сам вызов идёт между ними.
"""

from __future__ import annotations

import asyncio
from typing import Any

from aicore.broker import broker, ensure_broker_started
from aicore.config import get_settings
from aicore.db.session import get_database
from aicore.llm import LLMPermanentError, LLMRequest, get_llm_client
from aicore.logging import get_logger
from aicore.metrics import record_enqueue_failed, record_rag_fallback
from aicore.tasks._common import load_task, mark_task_enqueue_failed, mark_task_failed
from aicore.tasks.service import TaskService

logger = get_logger(__name__)


@broker.task(retry_on_error=True)
async def answer_question(task_id: str) -> dict[str, Any]:
    """Ответить на вопрос задачи через LLM и сохранить метрики."""
    settings = get_settings()
    try:
        async with asyncio.timeout(settings.task_timeout_seconds):
            return await _run(task_id)
    except TimeoutError:
        logger.error("task.timeout", task_id=task_id, timeout=settings.task_timeout_seconds)
        await mark_task_failed(task_id, "TimeoutError: task exceeded deadline", retryable=True)
        raise


async def _run(task_id: str) -> dict[str, Any]:
    db = get_database()
    await db.ensure_ready()

    # Фаза 1: загрузить задачу и атомарно заклеймить её, затем отпустить соединение.
    async with db.session() as session:
        svc = TaskService(session)
        task = await load_task(svc, task_id)
        if task is None:
            return {"status": "failed", "task_id": task_id, "error": "task not found"}
        if not await svc.mark_running(task):
            # Задачу уже обрабатывает другой воркер или она завершена: не выполняем
            # повторно (иначе перезапишем терминальный статус и оплатим LLM дважды).
            return {"status": "skipped", "task_id": task_id, "error": "task is not pending"}
        payload = dict(task.payload or {})

    question = str(payload.get("question", "")).strip()
    if not question:
        # API-слой отсекает пустой вопрос (min_length=1), но пробельная строка или
        # прямой create_task(payload={"question": ""}) дошли бы сюда и сожгли
        # LLM-вызов на пустом промпте. Пустой ввод неисправим ретраем.
        # Проверка идёт ПОСЛЕ claim, поэтому неисправимый ввод стоит одной
        # попытки (attempts=1); так же в tasks/agent_run.py — ради консистентности.
        await mark_task_failed(task_id, "ValueError: question must not be empty", retryable=False)
        return {"status": "dead_letter", "task_id": task_id, "error": "empty question"}
    prompt = await _with_rag(question) if payload.get("use_rag") else question
    request = LLMRequest(
        prompt=prompt,
        system=payload.get("system"),
        max_tokens=payload.get("max_tokens"),
    )

    # Фаза 2: вызов LLM без удержания БД-соединения.
    client = get_llm_client()
    try:
        response = await client.complete(request)
    except LLMPermanentError as exc:
        error = f"{type(exc).__name__}: {exc}"
        await mark_task_failed(task_id, error, retryable=False)
        return {"status": "dead_letter", "task_id": task_id, "error": error}
    except Exception as exc:  # noqa: BLE001 — transient: пусть брокер ретраит
        await mark_task_failed(task_id, f"{type(exc).__name__}: {exc}", retryable=True)
        raise

    # Фаза 3: сохранить ответ и его метрики.
    result = {"answer": response.text, "model": response.model, "latency_ms": response.latency_ms}
    async with db.session() as session:
        svc = TaskService(session)
        task = await load_task(svc, task_id)
        if task is None:
            return {"status": "failed", "task_id": task_id, "error": "task not found"}
        await svc.mark_succeeded(
            task,
            result=result,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost_usd=response.cost_usd,
        )
    return result


async def _with_rag(question: str) -> str:
    """Prepend retrieved context to the question (opt-in via payload.use_rag).

    Retrieval failures degrade gracefully: a broken vector store must not fail
    the whole answer task, it just answers without context.
    """
    from aicore.rag import build_context, build_rag_prompt, get_retriever

    try:
        hits = await get_retriever().retrieve(question)
    except Exception:  # noqa: BLE001 — деградируем до ответа без контекста
        # Не логируем сам вопрос: он может содержать PII. Длины достаточно,
        # чтобы отличить «пустой ввод» от реального сбоя ретрива.
        logger.exception("rag.retrieve_failed", question_chars=len(question))
        record_rag_fallback(reason="retrieve_error")
        return question

    bundle = build_context(hits, max_chars=get_settings().rag_context_max_chars)
    if not bundle:
        return question
    return build_rag_prompt(question, bundle)


async def enqueue_answer_question(task_id: str, *, mark_failed_on_error: bool = True) -> bool:
    """Поставить задачу в брокер.

    На InMemoryBroker (dev) задача исполняется прямо здесь; на
    RedisStreamBroker (prod) вызов только кладёт её в очередь и возвращается.

    `ensure_broker_started` внутри try: если Redis/сеть недоступны, startup
    падает так же, как и `kiq`, и задача должна быть помечена FAILED, а не
    остаться в PENDING навсегда.

    Returns ``False`` when the broker call failed (the row is marked FAILED),
    so the sweeper does not count a failed re-enqueue as reclaimed.
    """
    try:
        await ensure_broker_started()
        await answer_question.kiq(task_id)
    except Exception as exc:  # noqa: BLE001 — фиксируем в БД, а не только в логе
        logger.exception("task.enqueue_failed", task_id=task_id)
        record_enqueue_failed("answer_question")
        if mark_failed_on_error:
            await mark_task_enqueue_failed(task_id, f"{type(exc).__name__}: {exc}")
        return False
    return True
