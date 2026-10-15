"""Тесты eval-харнесса агентов (trajectory + outcome) — офлайн."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from aicore.config import Settings
from aicore.evals import DEFAULT_DATASET, run_evals
from aicore.evals.graders import answer_contains, args_carry_query, called_tool, final_answer


class TestGraders:
    def test_called_tool_matches_name(self) -> None:
        messages = [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])]
        assert called_tool(messages, "x")
        assert not called_tool(messages, "y")

    def test_answer_contains_all_keywords(self) -> None:
        assert answer_contains("The broker redelivers", ("broker", "redeliver"))
        assert not answer_contains("The broker redelivers", ("broker", "cpu"))

    def test_args_carry_query(self) -> None:
        messages = [
            AIMessage(content="", tool_calls=[{"name": "x", "args": {"query": "Hi"}, "id": "1"}])
        ]
        assert args_carry_query(messages, "x", " hi ")
        assert not args_carry_query(messages, "x", "bye")

    def test_final_answer_is_last_ai_message(self) -> None:
        messages = [AIMessage(content="first"), AIMessage(content="last")]
        assert final_answer(messages) == "last"


class TestRunEvals:
    @pytest.mark.asyncio
    async def test_all_cases_pass_offline(self) -> None:
        report = await run_evals(settings=Settings(rag_retrieve_limit=1))
        assert report.total == len(DEFAULT_DATASET)
        assert report.pass_rate == 1.0
        assert report.tool_routing_rate == 1.0
        assert report.grounded_rate == 1.0
        assert report.total_cost_usd > 0
        assert report.mean_tool_calls >= 1.0

    @pytest.mark.asyncio
    async def test_report_to_dict_is_serialisable(self) -> None:
        report = await run_evals(settings=Settings(rag_retrieve_limit=1))
        payload = report.to_dict()
        assert payload["cases"] == report.total
        assert all("checks" in case for case in payload["results"])
        # Per-case metrics must be decomposable, not only aggregated.
        assert all(
            {"tool_calls", "cost_usd", "model_calls"} <= set(case) for case in payload["results"]
        )

    @pytest.mark.asyncio
    async def test_failing_case_does_not_abort_report(self) -> None:
        """Один упавший кейс не должен ронять весь отчёт (regression gate).

        `agent_recursion_limit=1` заставляет каждый кейс упасть с
        GraphRecursionError; отчёт обязан вернуться целиком, с ошибками по
        кейсам, а не выбросить исключение.
        """
        report = await run_evals(settings=Settings(rag_retrieve_limit=1, agent_recursion_limit=1))
        assert report.total == len(DEFAULT_DATASET)
        assert report.passed == 0
        assert all(case.error for case in report.results)
