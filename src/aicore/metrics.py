"""Prometheus metrics for the task pipeline and the LLM layer.

One module, one registry: the API, the worker and the LLM clients all write
here, and `/metrics` exposes the result. Kept independent of the rest of the
app (only ``prometheus_client`` is imported) so any layer can record without
risk of an import cycle.
"""

from __future__ import annotations

import os

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.multiprocess import MultiProcessCollector

# --- tasks ---------------------------------------------------------------
TASKS_TOTAL = Counter(
    "aicore_tasks_total",
    "Task status transitions.",
    ["task_type", "status"],
)
TASK_LATENCY_MS = Histogram(
    "aicore_task_latency_ms",
    "End-to-end task processing latency in milliseconds.",
    ["task_type"],
    buckets=(5, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
)
ENQUEUE_FAILURES_TOTAL = Counter(
    "aicore_enqueue_failures_total",
    "Tasks committed to the DB but not enqueued (broker enqueue failed).",
    ["task_type"],
)

# --- LLM -----------------------------------------------------------------
LLM_CALLS_TOTAL = Counter(
    "aicore_llm_calls_total",
    "Successful LLM calls.",
    ["backend", "model"],
)
LLM_TOKENS_TOTAL = Counter(
    "aicore_llm_tokens_total",
    "LLM tokens by kind (prompt/completion).",
    ["kind"],
)
LLM_COST_USD_TOTAL = Counter(
    "aicore_llm_cost_usd_total",
    "Accumulated LLM cost in USD.",
    ["model"],
)
LLM_LATENCY_MS = Histogram(
    "aicore_llm_latency_ms",
    "LLM call latency in milliseconds.",
    ["backend"],
    buckets=(5, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000),
)

# --- embeddings / RAG ----------------------------------------------------
EMBEDDINGS_TOTAL = Counter(
    "aicore_embeddings_total",
    "Successful embedding calls.",
    ["backend", "model"],
)
EMBEDDING_TOKENS_TOTAL = Counter(
    "aicore_embedding_tokens_total",
    "Tokens embedded, by kind.",
    ["kind"],
)
EMBEDDING_COST_USD_TOTAL = Counter(
    "aicore_embedding_cost_usd_total",
    "Accumulated embedding cost in USD.",
    ["model"],
)
EMBEDDING_LATENCY_MS = Histogram(
    "aicore_embedding_latency_ms",
    "Embedding call latency in milliseconds.",
    ["backend"],
    buckets=(5, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000),
)
RAG_RETRIEVALS_TOTAL = Counter(
    "aicore_rag_retrievals_total",
    "Retrieval queries by hybrid mode.",
    ["hybrid"],
)
RAG_RETURNED_CHUNKS = Histogram(
    "aicore_rag_returned_chunks",
    "Number of chunks returned per retrieval query.",
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34, 55),
)
RAG_FALLBACKS_TOTAL = Counter(
    "aicore_rag_fallbacks_total",
    "Retrievals that degraded to answering without context.",
    ["reason"],
)


# --- multiprocess -----------------------------------------------------------
def multiprocess_registry() -> CollectorRegistry | None:
    """Registry that aggregates worker processes when multiprocess mode is on.

    Prometheus' multiprocess mode is enabled by setting
    ``PROMETHEUS_MULTIPROC_DIR`` **before** this module is imported; each worker
    then writes its counters/histograms to files in that directory instead of
    process memory. A single exporter (worker or the dedicated metrics server)
    reads them through ``MultiProcessCollector``, so with ``taskiq worker
    --workers N`` counters from all N processes are visible instead of only the
    one that won the :9100 bind.

    The directory must be shared between the worker processes and the process
    serving ``/metrics`` (same host or a mounted volume in Docker).
    """
    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return None
    registry = CollectorRegistry()
    try:
        MultiProcessCollector(registry)
    except (OSError, ValueError):
        # Directory is set but missing/unreadable: fail soft and let the caller
        # fall back to the in-process registry rather than 500 on /metrics.
        return None
    return registry


def record_task_status(task_type: str, status: str) -> None:
    """Count one task status transition (running/succeeded/pending/...)."""
    TASKS_TOTAL.labels(task_type=task_type, status=status).inc()


def record_task_latency(task_type: str, latency_ms: int) -> None:
    """Observe end-to-end latency for a finished task."""
    TASK_LATENCY_MS.labels(task_type=task_type).observe(latency_ms)


def record_enqueue_failed(task_type: str) -> None:
    """Count a task that was committed but could not be enqueued.

    The row stays PENDING with no queue message, so this counter is the alerting
    signal that a sweeper/outbox is needed.
    """
    ENQUEUE_FAILURES_TOTAL.labels(task_type=task_type).inc()


def record_llm_call(
    *,
    backend: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    latency_ms: int,
) -> None:
    """Record one successful LLM call: tokens, cost and latency."""
    LLM_CALLS_TOTAL.labels(backend=backend, model=model).inc()
    LLM_TOKENS_TOTAL.labels(kind="prompt").inc(prompt_tokens)
    LLM_TOKENS_TOTAL.labels(kind="completion").inc(completion_tokens)
    LLM_COST_USD_TOTAL.labels(model=model).inc(cost_usd)
    LLM_LATENCY_MS.labels(backend=backend).observe(latency_ms)


def record_embedding_call(
    *,
    backend: str,
    model: str,
    tokens: int,
    cost_usd: float,
    latency_ms: int,
) -> None:
    """Record one successful embedding call."""
    EMBEDDINGS_TOTAL.labels(backend=backend, model=model).inc()
    EMBEDDING_TOKENS_TOTAL.labels(kind="embedding").inc(tokens)
    EMBEDDING_COST_USD_TOTAL.labels(model=model).inc(cost_usd)
    EMBEDDING_LATENCY_MS.labels(backend=backend).observe(latency_ms)


def record_retrieval(*, hybrid: bool, returned: int) -> None:
    """Count one retrieval query and observe how many chunks it returned."""
    RAG_RETRIEVALS_TOTAL.labels(hybrid="on" if hybrid else "off").inc()
    RAG_RETURNED_CHUNKS.observe(returned)


def record_rag_fallback(*, reason: str) -> None:
    """Count one retrieval that degraded to a context-less answer."""
    RAG_FALLBACKS_TOTAL.labels(reason=reason).inc()
