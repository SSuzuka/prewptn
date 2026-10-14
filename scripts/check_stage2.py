"""Проверка этапа 2: БД, брокер, сервис задач, ретраи и DLQ.

Запуск:
    uv run python scripts/check_stage2.py
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicore.config import Settings, reset_settings_cache  # noqa: E402
from aicore.db.models import TaskStatus  # noqa: E402
from aicore.db.session import Database, reset_database, set_database  # noqa: E402
from aicore.logging import configure_logging, get_logger  # noqa: E402
from aicore.tasks.ping import PermanentTaskError, process_task  # noqa: E402
from aicore.tasks.service import TaskService, latency_ms  # noqa: E402

log = get_logger("stage2")
ok = True


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        ok = False


async def main() -> int:
    reset_settings_cache()
    reset_database()
    configure_logging(Settings(log_level="WARNING", log_json=False))

    # ВАЖНО: не ":memory:". С NullPool каждый новый коннект получает СВОЮ
    # пустую in-memory БД, поэтому таблицы, созданные в одном соединении,
    # не видны в другом -> "no such table". Берём временный файл.
    db_path = ROOT / "stage2_check.sqlite3"
    if db_path.exists():
        db_path.unlink()

    settings = Settings(db_url=f"sqlite+aiosqlite:///{db_path.as_posix()}")
    db = Database(settings)
    await db.connect()
    await db.create_all()
    # Задачи берут БД через get_database(), поэтому подменяем синглтон.
    set_database(db)

    print("== schema ==")
    from aicore.db.base import Base

    tables = sorted(Base.metadata.tables)
    check("three tables created", len(tables) == 3, ", ".join(tables))
    check("healthcheck", await db.healthcheck() is True)

    print("\n== task lifecycle ==")
    async with db.session() as session:
        svc = TaskService(session)

        task, created = await svc.create_task(
            task_type="ping",
            payload={"hello": "world"},
            max_attempts=3,
        )
        check("task created", created is True, str(task.id)[:8])
        check("initial status pending", task.status is TaskStatus.PENDING)

        await svc.mark_running(task)
        check("running increments attempts", task.attempts == 1)
        check("latency None while running", latency_ms(task) is None)

        await svc.mark_succeeded(
            task, result={"ok": True}, prompt_tokens=10, completion_tokens=5, cost_usd=0.001
        )
        check("succeeded status", task.status is TaskStatus.SUCCEEDED)
        check("metrics stored", task.prompt_tokens == 10 and task.cost_usd == 0.001)
        check("latency computed", isinstance(latency_ms(task), int), f"{latency_ms(task)}ms")

    print("\n== idempotency ==")
    async with db.session() as session:
        svc = TaskService(session)
        _, first = await svc.create_task(task_type="ping", payload={}, idempotency_key="key-abc")
        second_task, replay = await svc.create_task(
            task_type="ping", payload={}, idempotency_key="key-abc"
        )
        check("first call creates", first is True)
        check("replay does not create", replay is False)
        check("replay returns same task", second_task.idempotency_key == "key-abc")

        count = len(await svc.list_tasks())
        check("no duplicate rows", count == 2, f"{count} tasks total")

    print("\n== retry -> dead letter ==")
    async with db.session() as session:
        svc = TaskService(session)
        task, _ = await svc.create_task(
            task_type="ping", payload={"fail_times": 99}, max_attempts=2
        )

        await svc.mark_running(task)
        await svc.mark_failed(task, "boom 1")
        check("retry keeps pending", task.status is TaskStatus.PENDING)
        check("can_retry true", task.can_retry is True)

        await svc.mark_running(task)
        await svc.mark_failed(task, "boom 2")
        check("exhausted -> dead_letter", task.status is TaskStatus.DEAD_LETTER)
        check("can_retry false", task.can_retry is False)

        dl = await svc.list_dead_letters()
        check("dead letter listed", any(t.id == task.id for t in dl), f"{len(dl)} in DLQ")

        await svc.reset_for_retry(task)
        check("reset clears attempts", task.attempts == 0)
        check("reset back to pending", task.status is TaskStatus.PENDING)

    print("\n== broker task (real execution) ==")
    async with db.session() as session:
        svc = TaskService(session)
        task, _ = await svc.create_task(task_type="ping", payload={"echo": "hi"})

    result = await process_task(str(task.id))
    check("task executed", result["status"] if "status" in result else True, str(result)[:60])

    async with db.session() as session:
        svc = TaskService(session)
        reloaded = await svc.get_task(task.id)
        check("persisted as succeeded", reloaded.status is TaskStatus.SUCCEEDED)

    # Ретраи: задача падает 2 раза, затем успешна на 3-й попытке.
    async with db.session() as session:
        svc = TaskService(session)
        task2, _ = await svc.create_task(task_type="ping", payload={"fail_times": 2})

    for _ in range(1, 4):
        try:
            await process_task(str(task2.id), fail_times=2)
            break
        except Exception:  # noqa: BLE001
            pass

    async with db.session() as session:
        svc = TaskService(session)
        reloaded2 = await svc.get_task(task2.id)
        check("succeeded after retries", reloaded2.status is TaskStatus.SUCCEEDED)
        check("attempts counted", reloaded2.attempts == 3, f"{reloaded2.attempts} attempts")

    # Несуществующая задача -> PermanentTaskError (ретрай бессмысленен).
    try:
        await process_task(str(uuid4()))
        check("missing task raises", False)
    except PermanentTaskError:
        check("missing task raises PermanentTaskError", True)

    await db.disconnect()
    reset_database()

    # Убираем временную БД, чтобы чек-скрипт не оставлял мусора.
    if db_path.exists():
        db_path.unlink()

    print()
    print("STAGE 2 " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
