"""Проверка этапа 4: RAG — chunking, embeddings, Qdrant, retrieval, контекст.

Всё офлайн: stub-эмбеддер (детерминированный), векторное хранилище
`AsyncQdrantClient(":memory:")`, StubLLM. Ни Docker, ни ключей, ни сети.

Запуск:
    uv run python scripts/check_stage4.py
"""

import asyncio
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import AsyncQdrantClient  # noqa: E402

from aicore.config import Settings, reset_settings_cache  # noqa: E402
from aicore.db.models import DocumentStatus, TaskStatus  # noqa: E402
from aicore.db.session import Database, reset_database, set_database  # noqa: E402
from aicore.llm import reset_llm_client, set_llm_client  # noqa: E402
from aicore.llm.stub import StubLLM  # noqa: E402
from aicore.logging import configure_logging  # noqa: E402
from aicore.rag import (  # noqa: E402
    build_context,
    build_rag_prompt,
    build_retriever,
    chunk_text,
    reset_retriever,
    set_retriever,
)
from aicore.rag.base import EmbeddingRequest  # noqa: E402
from aicore.rag.documents import DocumentService  # noqa: E402
from aicore.rag.store import VectorStore  # noqa: E402
from aicore.rag.stub import StubEmbeddings  # noqa: E402
from aicore.tasks.llm import answer_question  # noqa: E402
from aicore.tasks.rag_index import index_document  # noqa: E402
from aicore.tasks.service import TaskService  # noqa: E402
from aicore.telemetry import configure_telemetry, reset_telemetry  # noqa: E402

ok = True

DOCS = {
    "idempotency": "Idempotency keys stop duplicate task creation and duplicate LLM spend.",
    "broker": "The redis stream broker supports acknowledge and redelivery when a worker dies.",
    "pricing": "Cloudflare Containers bill provisioned memory but only active CPU time.",
}


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok
    print(f"  [{'OK  ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        ok = False


def _memory_retriever(cfg: Settings):
    return build_retriever(cfg)


async def main() -> int:
    reset_settings_cache()
    reset_database()
    reset_telemetry()
    reset_retriever()
    reset_llm_client()
    configure_logging(Settings(log_level="WARNING", log_json=False))

    cfg = Settings()
    print("== config ==")
    check(
        "default embedding backend is stub", cfg.embedding_backend == "stub", cfg.embedding_backend
    )
    check("vector backend is memory", cfg.vector_backend == "memory", cfg.vector_backend)
    check("embedding dim set", cfg.embedding_dim > 0, str(cfg.embedding_dim))
    check("chunk size > overlap", cfg.rag_chunk_size > cfg.rag_chunk_overlap)

    print("\n== chunking ==")
    chunks = chunk_text("Sentence one. " * 80, chunk_size=200, chunk_overlap=20)
    check("long text splits", len(chunks) > 1, f"{len(chunks)} chunks")
    check("indices are sequential", [c.index for c in chunks] == list(range(len(chunks))))
    check("empty text yields no chunks", chunk_text("   ") == [])

    print("\n== embeddings (stub) ==")
    embedder = StubEmbeddings(settings=Settings(embedding_dim=128))
    resp = await embedder.embed(EmbeddingRequest(texts=["idempotency key", "idempotency key"]))
    check("deterministic", resp.vectors[0] == resp.vectors[1])
    check("dimension respected", resp.dimension == 128 and len(resp.vectors[0]) == 128)
    norm = math.sqrt(sum(value * value for value in resp.vectors[0]))
    check("normalised", math.isclose(norm, 1.0, rel_tol=1e-9), f"|v|={norm:.6f}")
    check("tokens counted", resp.usage.total_tokens == 4, str(resp.usage.total_tokens))

    print("\n== vector store (memory) ==")
    store = VectorStore(client=AsyncQdrantClient(":memory:"), collection="check", dimension=128)
    from aicore.rag.store import VectorPoint, make_point_id

    await store.upsert(
        [
            VectorPoint(
                id=make_point_id("d", 0),
                vector=resp.vectors[0],
                payload={"document_id": "d", "chunk_index": 0, "text": "idempotency"},
            )
        ]
    )
    check("upsert stored a point", await store.count() == 1)
    await store.aclose()

    print("\n== retriever: index + hybrid retrieval ==")
    retriever = _memory_retriever(cfg)
    set_retriever(retriever)
    for document_id, text in DOCS.items():
        await retriever.index(
            document_id=document_id,
            chunks=chunk_text(text, chunk_size=200, chunk_overlap=20),
            title=document_id,
        )
    check("all documents indexed", await retriever.store.count() == len(DOCS))
    hits = await retriever.retrieve("redis stream acknowledge redelivery", limit=1)
    check("relevant chunk retrieved", bool(hits) and hits[0].document_id == "broker")
    hits2 = await retriever.retrieve("active cpu provisioned memory", limit=1)
    check("second query hits pricing", bool(hits2) and hits2[0].document_id == "pricing")

    print("\n== context ==")
    bundle = build_context(hits + hits2, max_chars=1000)
    check("context numbered", "[1]" in bundle.text and "[2]" in bundle.text)
    prompt = build_rag_prompt("what does the broker do?", bundle)
    check("prompt embeds context + question", "Context:" in prompt and "broker" in prompt.lower())

    print("\n== indexing task (DB) ==")
    db_path = ROOT / "stage4_check.sqlite3"
    if db_path.exists():
        db_path.unlink()
    db = Database(Settings(db_url=f"sqlite+aiosqlite:///{db_path.as_posix()}"))
    await db.connect()
    await db.create_all()
    set_database(db)
    reset_retriever()
    set_retriever(_memory_retriever(cfg))

    async with db.session() as session:
        document, created = await DocumentService(session).create_document(
            title="Doc", content="The broker retries failed tasks with backoff and jitter."
        )
        task, _ = await TaskService(session).create_task(
            task_type="index_document",
            payload={
                "document_id": str(document.id),
                "title": document.title,
                "content": "The broker retries failed tasks with backoff and jitter.",
            },
        )
    check("document created", created is True)

    await index_document(str(task.id))

    async with db.session() as session:
        reloaded = await DocumentService(session).get_document(document.id)
        reloaded_task = await TaskService(session).get_task(task.id)
    check("document indexed", reloaded is not None and reloaded.status is DocumentStatus.INDEXED)
    check("chunks recorded", reloaded is not None and reloaded.chunk_count >= 1)
    check(
        "task succeeded", reloaded_task is not None and reloaded_task.status is TaskStatus.SUCCEEDED
    )

    print("\n== answer_question with use_rag ==")
    set_llm_client(StubLLM(settings=cfg, model="stub-rag"))
    async with db.session() as session:
        qtask, _ = await TaskService(session).create_task(
            task_type="answer_question",
            payload={"question": "how does the broker handle failures", "use_rag": True},
        )
    result = await answer_question(str(qtask.id))
    check("answer used retrieved context", "Context:" in result.get("answer", ""))

    await db.disconnect()
    reset_database()
    reset_retriever()
    reset_llm_client()
    if db_path.exists():
        db_path.unlink()

    print("\n== telemetry ==")
    tel = configure_telemetry(Settings())
    check("langfuse off without keys", tel.langfuse is None)

    print()
    print("STAGE 4 " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
