"""Offline evaluation set for the agent layer.

Each case is a question, the tool the supervisor is expected to route through,
and keywords the final answer must contain (grounding in the retrieved
document). The dataset is intentionally tiny and deterministic: it runs on the
stub chat model and stub embeddings with no network, so it doubles as a
regression gate in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One evaluation case."""

    name: str
    question: str
    # Инструмент, который ОБЯЗАН вызвать супервизор (trajectory-проверка).
    expect_tool: str
    # Ключевые слова, которые обязаны остаться в финальном ответе (outcome).
    answer_keywords: tuple[str, ...] = field(default_factory=tuple)


KNOWLEDGE_BASE: dict[str, str] = {
    "idempotency": (
        "Idempotency keys stop duplicate task creation and duplicate LLM spend; "
        "a replayed request returns the original task instead of burning tokens again."
    ),
    "broker": (
        "The redis stream broker supports acknowledge and redelivery, so a task is "
        "re-delivered to another worker when a worker dies mid-processing."
    ),
    "pricing": (
        "Cloudflare Containers bill provisioned memory but only active CPU time, so "
        "an idle container is cheap while a busy one costs on cpu."
    ),
}

DEFAULT_DATASET: tuple[EvalCase, ...] = (
    EvalCase(
        name="idempotency_spend",
        question="how do idempotency keys stop duplicate llm spend",
        expect_tool="research_knowledge_base",
        answer_keywords=("idempotency",),
    ),
    EvalCase(
        name="broker_redelivery",
        question="redis stream acknowledge redelivery when a worker dies",
        expect_tool="research_knowledge_base",
        answer_keywords=("redelivery",),
    ),
    EvalCase(
        name="billing_cpu",
        question="cloudflare containers bill provisioned memory active cpu",
        expect_tool="research_knowledge_base",
        answer_keywords=("cpu",),
    ),
)
