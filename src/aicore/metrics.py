"""Prometheus metrics for the task pipeline and the LLM layer.

One module, one registry: the API, the worker and the LLM clients all write
here, and `/metrics` exposes the result. Kept independent of the rest of the
app (only ``prometheus_client`` is imported) so any layer can record without
risk of an import cycle.
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

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


def record_task_status(task_type: str, status: str) -> None:
    """Count one task status transition (running/succeeded/pending/...)."""
    TASKS_TOTAL.labels(task_type=task_type, status=status).inc()


def record_task_latency(task_type: str, latency_ms: int) -> None:
    """Observe end-to-end latency for a finished task."""
    TASK_LATENCY_MS.labels(task_type=task_type).observe(latency_ms)


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
