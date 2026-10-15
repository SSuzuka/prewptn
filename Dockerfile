# syntax=docker/dockerfile:1

# One image runs both processes: the FastAPI API (default CMD) and the taskiq
# worker (`taskiq worker aicore.broker:broker`). Override the command to run
# the worker. The production extras (redis/postgres/openai/embeddings) are
# installed so the image can reach Redis Streams, PostgreSQL, an
# OpenAI-compatible API and local (fastembed/ONNX) embeddings for RAG.
FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

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
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra redis --extra postgres --extra openai --extra embeddings

ENV PATH="/app/.venv/bin:$PATH"

# Run as a non-root user. The fastembed/HF cache dirs must be writable: the
# local embedding model is downloaded once on first use.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/.cache/fastembed /app/.cache/huggingface \
    && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "aicore.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
