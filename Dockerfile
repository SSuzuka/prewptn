# syntax=docker/dockerfile:1

# One image runs all processes: the FastAPI API (default CMD), the taskiq
# worker (`taskiq worker aicore.broker:broker`), the scheduler
# (`taskiq scheduler aicore.broker:scheduler`) and Alembic migrations. Override
# the command to run each one. The production extras (redis/postgres/openai/
# embeddings) are installed so the image can reach Redis Streams, PostgreSQL, an
# OpenAI-compatible API and local (fastembed/ONNX) embeddings for RAG.
FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/

# onnxruntime (pulled in by fastembed) links libgomp at runtime, which the slim
# base image does not ship. Without it EMBEDDING_BACKEND=local fails on import.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dependencies first (cache-friendly): manifest + lock only, project not yet.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev \
        --extra redis --extra postgres --extra openai --extra embeddings

# Then the source and the project itself. README.md is required: hatchling
# reads it as the package readme while building the project metadata.
COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY alembic.ini ./
COPY migrations ./migrations
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra redis --extra postgres --extra openai --extra embeddings

ENV PATH="/app/.venv/bin:$PATH"

# Run as a non-root user. The fastembed/HF cache dirs must be writable: the
# local embedding model is downloaded once on first use. Prometheus multiprocess
# mode (PROMETHEUS_MULTIPROC_DIR) opens files in /tmp/prometheus at import, so
# the directory must exist and be owned by the runtime user even when a named
# volume is mounted over it.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/.cache/fastembed /app/.cache/huggingface /tmp/prometheus \
    && chown -R app:app /app /tmp/prometheus
USER app

EXPOSE 8000
CMD ["uvicorn", "aicore.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
