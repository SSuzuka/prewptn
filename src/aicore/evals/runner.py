"""Run the agent against the offline eval set and aggregate the results.

The report tracks the metrics 2026 agent-eval guidance cares about: success rate
(+ per-check breakdown), tool-routing correctness, and cost — not just "did it
answer". It runs fully offline on the stub model, so a regression shows up in CI
without a network call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from aicore.agents import build_agent, reset_agent, set_agent
from aicore.agents.usage import AgentUsageCallbackHandler, final_answer
from aicore.config import Settings, get_settings
from aicore.evals.dataset import DEFAULT_DATASET, KNOWLEDGE_BASE, EvalCase
from aicore.evals.graders import (
    answer_contains,
    args_carry_query,
    called_tool,
    called_tools,
)
from aicore.rag import build_retriever, chunk_text, reset_retriever, set_retriever


@dataclass(slots=True)
class CaseResult:
    """Outcome of one case: pass/fail plus the raw checks."""

    name: str
    passed: bool
    checks: dict[str, bool]
    answer: str
    tools: list[str]
    model_calls: int
    tool_calls: int
    cost_usd: float
    # Set when the case raised (recursion limit, timeout, ...). A single failing
    # case must not abort the whole report, so its error is captured here.
    error: str | None = None


@dataclass(slots=True)
class EvalReport:
    """Aggregate over all cases."""

    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def _rate(self, check: str) -> float:
        if not self.results:
            return 0.0
        hits = sum(1 for result in self.results if result.checks.get(check))
        return hits / self.total

    @property
    def tool_routing_rate(self) -> float:
        return self._rate("tool_called")

    @property
    def grounded_rate(self) -> float:
        return self._rate("answer_grounded")

    @property
    def total_cost_usd(self) -> float:
        return round(sum(result.cost_usd for result in self.results), 8)

    @property
    def mean_tool_calls(self) -> float:
        return sum(result.tool_calls for result in self.results) / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": self.total,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "tool_routing_rate": round(self.tool_routing_rate, 4),
            "grounded_rate": round(self.grounded_rate, 4),
            "total_cost_usd": self.total_cost_usd,
            "mean_tool_calls": round(self.mean_tool_calls, 4),
            "results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "checks": result.checks,
                    "tools": result.tools,
                    "answer": result.answer,
                    "model_calls": result.model_calls,
                    "tool_calls": result.tool_calls,
                    "cost_usd": result.cost_usd,
                    "error": result.error,
                }
                for result in self.results
            ],
        }


async def run_evals(
    cases: Sequence[EvalCase] | None = None,
    *,
    settings: Settings | None = None,
) -> EvalReport:
    """Index the knowledge base, run every case through the agent, grade it."""
    cfg = settings or get_settings()
    dataset = tuple(cases) if cases is not None else DEFAULT_DATASET

    # Build the agent BEFORE installing the process-wide retriever/agent: if
    # building raises, the finally below still closes the retriever.
    retriever = build_retriever(cfg)
    try:
        agent = build_agent(cfg)
        # Process-wide singletons: not thread-safe, offline/serial runs only.
        set_retriever(retriever)
        set_agent(agent)

        for document_id, text in KNOWLEDGE_BASE.items():
            await retriever.index(
                document_id=document_id,
                chunks=chunk_text(
                    text, chunk_size=cfg.rag_chunk_size, chunk_overlap=cfg.rag_chunk_overlap
                ),
                title=document_id,
            )

        report = EvalReport()
        for case in dataset:
            handler = AgentUsageCallbackHandler()
            try:
                state = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": case.question}]},
                    config={
                        "configurable": {"thread_id": f"eval-{case.name}"},
                        "recursion_limit": cfg.agent_recursion_limit,
                        "callbacks": [handler],
                    },
                )
            except Exception as exc:  # noqa: BLE001 — one case must not kill the report
                # Partial report is the point of a regression gate: a recursion
                # limit or timeout on case N should still grade cases N+1...
                report.results.append(
                    CaseResult(
                        name=case.name,
                        passed=False,
                        checks={},
                        answer="",
                        tools=[],
                        model_calls=handler.usage.model_calls,
                        tool_calls=handler.usage.tool_calls,
                        cost_usd=handler.usage.cost_usd(cfg),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            messages = state.get("messages") or []
            answer = final_answer(messages)
            checks = {
                "tool_called": called_tool(messages, case.expect_tool),
                "args_query": args_carry_query(messages, case.expect_tool, case.question),
                "answer_grounded": answer_contains(answer, case.answer_keywords),
            }
            usage = handler.usage
            report.results.append(
                CaseResult(
                    name=case.name,
                    passed=all(checks.values()),
                    checks=checks,
                    answer=answer,
                    tools=called_tools(messages),
                    model_calls=usage.model_calls,
                    tool_calls=usage.tool_calls,
                    cost_usd=usage.cost_usd(cfg),
                )
            )
        return report
    finally:
        await retriever.aclose()
        reset_agent()
        reset_retriever()
