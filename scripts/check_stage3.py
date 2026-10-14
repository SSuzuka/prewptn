"""Проверка этапа 3: LLM-слой — timeout, retries, rate limit, токены и стоимость.

Всё офлайн: StubLLM без ключей и сети, OpenAI-клиент на httpx.MockTransport.

Запуск:
    uv run python scripts/check_stage3.py
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx  # noqa: E402

from aicore.config import Settings, reset_settings_cache  # noqa: E402
from aicore.db.models import TaskStatus  # noqa: E402
from aicore.db.session import Database, reset_database  # noqa: E402
from aicore.llm import (  # noqa: E402
    LLMPermanentError,
    LLMRequest,
    LLMTransientError,
    LLMUsage,
    RetryPolicy,
    build_llm_client,
    compute_cost_usd,
    run_with_retry,
)
from aicore.llm.openai_compat import OpenAICompatibleLLM  # noqa: E402
from aicore.logging import configure_logging  # noqa: E402
from aicore.tasks.service import TaskService  # noqa: E402
from aicore.telemetry import configure_telemetry, record_generation, reset_telemetry  # noqa: E402

ok = True


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        ok = False


async def _no_sleep(_delay: float) -> None:
    return None


async def main() -> int:
    reset_settings_cache()
    reset_database()
    reset_telemetry()
    configure_logging(Settings(log_level="WARNING", log_json=False))

    print("== config ==")
    cfg = Settings()
    check("default llm backend is stub", cfg.llm_backend == "stub", cfg.llm_backend)
    check("timeout configured", cfg.llm_timeout_seconds > 0, f"{cfg.llm_timeout_seconds}s")
    check("concurrency limit set", cfg.llm_max_concurrency >= 1, str(cfg.llm_max_concurrency))
    check("pricing configured", cfg.llm_price_completion_per_1k >= 0)

    print("\n== cost accounting ==")
    cost = compute_cost_usd(
        LLMUsage(prompt_tokens=1000, completion_tokens=1000),
        prompt_per_1k=cfg.llm_price_prompt_per_1k,
        completion_per_1k=cfg.llm_price_completion_per_1k,
    )
    check("cost computed from tokens", cost > 0, f"${cost:.8f}")

    print("\n== stub client ==")
    client = build_llm_client(Settings())
    resp = await client.complete(LLMRequest(prompt="hello stage three", system="be brief"))
    check("deterministic answer", resp.text.startswith("[stub:"), resp.text[:40])
    check("tokens counted", resp.usage.total_tokens > 0, str(resp.usage))
    check("cost > 0", resp.cost_usd > 0, f"${resp.cost_usd:.8f}")
    check("latency recorded", resp.latency_ms >= 0, f"{resp.latency_ms}ms")
    await client.aclose()

    print("\n== retry policy ==")
    policy = RetryPolicy(max_retries=2, base_delay=1.0, jitter=False)
    delays = [policy.delay_for(i) for i in range(2)]
    check("exponential backoff", delays == [1.0, 2.0], str(delays))

    attempts = {"n": 0}

    async def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise LLMTransientError("blip")
        return "recovered"

    recovered = await run_with_retry(flaky, policy, sleep=_no_sleep)
    check("transient retried to success", recovered == "recovered", f"{attempts['n']} attempts")

    permanent_calls = {"n": 0}

    async def permanent() -> str:
        permanent_calls["n"] += 1
        raise LLMPermanentError("400 bad request")

    try:
        await run_with_retry(permanent, policy, sleep=_no_sleep)
        check("permanent error not retried", False)
    except LLMPermanentError:
        check(
            "permanent error not retried", permanent_calls["n"] == 1, f"{permanent_calls['n']} call"
        )

    print("\n== openai-compatible client (offline mock) ==")
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            json={
                "model": "gpt-x",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            },
        )

    oai = OpenAICompatibleLLM(
        settings=Settings(
            llm_backend="openai",
            openai_api_key="sk-check",
            llm_max_retries=2,
            llm_retry_base_delay=0.001,
        ),
        transport=httpx.MockTransport(handler),
    )
    oai_resp = await oai.complete(LLMRequest(prompt="hi"))
    check("429 retried then succeeded", oai_resp.text == "ok" and calls["n"] == 2)
    check("openai cost computed", oai_resp.cost_usd > 0, f"${oai_resp.cost_usd:.8f}")
    await oai.aclose()

    try:
        OpenAICompatibleLLM(settings=Settings(llm_backend="openai", openai_api_key=None))
        check("missing api key rejected", False)
    except LLMPermanentError:
        check("missing api key rejected", True)

    print("\n== metrics persisted on a task ==")
    db_path = ROOT / "stage3_check.sqlite3"
    if db_path.exists():
        db_path.unlink()

    db = Database(Settings(db_url=f"sqlite+aiosqlite:///{db_path.as_posix()}"))
    await db.connect()
    await db.create_all()

    async with db.session() as session:
        svc = TaskService(session)
        task, _ = await svc.create_task(task_type="answer_question", payload={"question": "hello"})
        await svc.mark_running(task)
        await svc.mark_succeeded(
            task,
            result={"answer": resp.text},
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            cost_usd=resp.cost_usd,
        )

    async with db.session() as session:
        reloaded = await TaskService(session).get_task(task.id)

    check("task reloaded", reloaded is not None)
    if reloaded is not None:
        check("status succeeded", reloaded.status is TaskStatus.SUCCEEDED)
        check("tokens stored", reloaded.prompt_tokens == resp.usage.prompt_tokens)
        check(
            "cost stored",
            reloaded.cost_usd is not None and abs(reloaded.cost_usd - resp.cost_usd) < 1e-9,
        )

    await db.disconnect()
    reset_database()
    if db_path.exists():
        db_path.unlink()

    print("\n== telemetry ==")
    tel = configure_telemetry(Settings())
    check("langfuse off without keys", tel.langfuse is None)
    record_generation(
        name="llm.complete",
        model=resp.model,
        input={"prompt": "x"},
        output=resp.text,
        prompt_tokens=1,
        completion_tokens=1,
        cost_usd=0.0,
        latency_ms=1,
    )
    check("record_generation is no-op offline", True)

    print()
    print("STAGE 3 " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
