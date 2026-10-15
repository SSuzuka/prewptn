"""Offline evaluation harness for the agent layer (trajectory + outcome)."""

from __future__ import annotations

from aicore.evals.dataset import DEFAULT_DATASET, KNOWLEDGE_BASE, EvalCase
from aicore.evals.runner import CaseResult, EvalReport, run_evals

__all__ = [
    "DEFAULT_DATASET",
    "KNOWLEDGE_BASE",
    "CaseResult",
    "EvalCase",
    "EvalReport",
    "run_evals",
]
