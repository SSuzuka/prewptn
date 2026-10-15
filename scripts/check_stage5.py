"""Проверка этапа 5: агенты — supervisor + subagents, tools, evals, задача.

Всё офлайн: детерминированная stub-модель (tool calling), stub-эмбеддер,
векторное хранилище `AsyncQdrantClient(":memory:")`. Ни Docker, ни ключей, ни
сети — как в check_stage4.

Запуск:
    uv run python scripts/check_stage5.py
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aicore.agents import (  # noqa: E402
    StubChatModel,
    build_agent,
    build_chat_model,
    reset_agent,
    search_knowledge_base,
    set_agent,
)
from aicore.config import Settings, reset_settings_cache  # noqa: E402
from aicore.db.models import TaskStatus  # noqa: E402
from aicore.db.session import Database, reset_database, set_database  # noqa: E402
from aicore.evals import run_evals  # noqa: E402
from aicore.logging import configure_logging  # noqa: E402
from aicore.rag import (  # noqa: E402
    build_retriever,
    chunk_text,
    get_retriever,
    reset_retriever,
    set_retriever,
)
from aicore.tasks.agent_run import run_agent  # noqa: E402
from aicore.tasks.service import TaskService  # noqa: E402
from aicore.telemetry import reset_telemetry  # noqa: E402

ok = True

DOCS = {
    "idempotency": "Idempotency keys stop duplicate task creation and duplicate LLM spend.",
    "broker": "The redis stream broker supports acknowledge and redelivery if a worker dies.",
    "pricing": "Cloudflare Containers bill provisioned memory but only active CPU time.",
}


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        ok = False


async def _index_all() -> None:
    retriever = get_retriever()
    for document_id, text in DOCS.items():
        await retriever.index(
            document_id=document_id,
            chunks=chunk_text(text, chunk_size=200, chunk_overlap=20),
            title=document_id,
        )


async def main() -> int:
    reset_settings_cache()
    reset_database()
    reset_telemetry()
    reset_retriever()
    reset_agent()
    configure_logging(Settings(log_level="WARNING", log_json=False))

    cfg = Settings()
    print("== config ==")
    check("default agent backend is stub", cfg.agent_backend == "stub", cfg.agent_backend)
    check("model call limit set", cfg.agent_max_model_calls >= 1, str(cfg.agent_max_model_calls))
    check("tool call limit set", cfg.agent_max_tool_calls >= 1, str(cfg.agent_max_tool_calls))
    check("recursion limit set", cfg.agent_recursion_limit >= 1, str(cfg.agent_recursion_limit))

    print("\n== chat model (stub) ==")
    model = build_chat_model(cfg)
    check("stub model by default", isinstance(model, StubChatModel), type(model).__name__)
    bound = model.bind_tools([{"name": "search_knowledge_base"}])
    first = await bound.ainvoke([{"role": "user", "content": "hello"}])
    check(
        "first turn emits a tool call",
        bool(first.tool_calls) and first.tool_calls[0]["name"] == "search_knowledge_base",
    )
    check("tool call carries the query", first.tool_calls[0]["args"] == {"query": "hello"})

    print("\n== tools + retrieval ==")
    set_retriever(build_retriever(cfg))
    await _index_all()
    tool_result = await search_knowledge_base.ainvoke({"query": "redis redelivery worker"})
    check("rag tool returns grounded passage", "redelivery" in tool_result, tool_result[:60])

    print("\n== agent loop (supervisor -> subagent -> tools) ==")
    agent = build_agent(cfg)
    set_agent(agent)
    state = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "how does the broker handle redelivery"}]},
        config={
            "configurable": {"thread_id": "check-1"},
            "recursion_limit": cfg.agent_recursion_limit,
        },
    )
    messages = state["messages"]
    tool_names = [
        call["name"]
        for message in messages
        if getattr(message, "tool_calls", None)
        for call in message.tool_calls
    ]
    check("supervisor called the research subagent", "research_knowledge_base" in tool_names)
    answer = messages[-1].content or ""
    check("final answer is grounded", "redelivery" in answer.lower(), answer[:60])

    print("\n== evals ==")
    report = await run_evals(settings=Settings(rag_retrieve_limit=1))
    check("all eval cases pass", report.passed == report.total, f"{report.passed}/{report.total}")
    check("tool routing 100%", report.tool_routing_rate == 1.0, str(report.tool_routing_rate))
    check("answer grounding 100%", report.grounded_rate == 1.0, str(report.grounded_rate))
    check("cost tracked", report.total_cost_usd > 0, str(report.total_cost_usd))

    print("\n== agent task (DB) ==")
    reset_retriever()
    reset_agent()
    db_path = ROOT / "stage5_check.sqlite3"
    if db_path.exists():
        db_path.unlink()
    db = Database(Settings(db_url=f"sqlite+aiosqlite:///{db_path.as_posix()}"))
    await db.connect()
    await db.create_all()
    set_database(db)
    set_retriever(build_retriever(cfg))
    await _index_all()

    async with db.session() as session:
        task, _ = await TaskService(session).create_task(
            task_type="run_agent",
            payload={"question": "how does the broker handle redelivery"},
        )
    result = await run_agent(str(task.id))
    answer = result.get("answer", "")
    check("agent task produced an answer", bool(answer), answer[:50])
    check("model calls recorded", result.get("model_calls", 0) >= 1, str(result.get("model_calls")))
    check("tokens recorded", result.get("prompt_tokens", 0) > 0, str(result.get("prompt_tokens")))

    async with db.session() as session:
        reloaded = await TaskService(session).get_task(task.id)
    check("task succeeded", reloaded is not None and reloaded.status is TaskStatus.SUCCEEDED)
    check("cost stored", reloaded is not None and reloaded.cost_usd is not None)

    await db.disconnect()
    reset_database()
    reset_retriever()
    reset_agent()
    if db_path.exists():
        db_path.unlink()

    print()
    print("STAGE 5 " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
