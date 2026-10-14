# syntax=docker/dockerfile:1

# One image runs both processes: the FastAPI API (default CMD) and the taskiq
# worker (`taskiq worker aicore.broker:broker`). Override the command to run
# the worker. The production extras (redis/postgres/openai) are installed so
# the image can reach Redis Streams, PostgreSQL and an OpenAI-compatible API.
FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cache-friendly): manifest + lock only, project not yet.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev \
        --extra redis --extra postgres --extra openai

# Then the source and the project itself. README.md is required: hatchling
# reads it as the package readme while building the project metadata.
COPY pyproject.toml uv.lock ./
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra redis --extra postgres --extra openai

ENV PATH="/app/.venv/bin:$PATH"

# Run as a non-root user.
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "aicore.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
