"""Smoke-тесты: окружение собрано, ключевые API доступны.

Запуск без внешней инфраструктуры:
    uv run pytest
"""

import importlib
import importlib.metadata as md

import pytest

RUNTIME_MODULES = [
    "aicore",
    "fastapi",
    "starlette",
    "uvicorn",
    "pydantic",
    "pydantic_settings",
    "sqlalchemy",
    "aiosqlite",
    "taskiq",
    "sse_starlette",
    "langchain",
    "langchain_core",
    "langgraph",
    "langgraph.prebuilt",
    "qdrant_client",
    "langfuse",
    "mcp",
    "mcp.server.fastmcp",
    "structlog",
    "prometheus_client",
    "opentelemetry.sdk",
]

# Ключевые объекты, на которых держится архитектура проекта.
KEY_APIS = [
    ("mcp.server.fastmcp", "FastMCP"),
    ("langchain.agents", "create_agent"),
    ("langgraph.prebuilt", "ToolNode"),
    ("taskiq.middlewares", "SmartRetryMiddleware"),
    ("langfuse", "get_client"),
    ("qdrant_client", "AsyncQdrantClient"),
]


@pytest.mark.parametrize("module", RUNTIME_MODULES)
def test_module_importable(module: str) -> None:
    assert importlib.import_module(module) is not None


@pytest.mark.parametrize(("module", "attr"), KEY_APIS)
def test_key_api_present(module: str, attr: str) -> None:
    mod = importlib.import_module(module)
    assert hasattr(mod, attr), f"{module}.{attr} not found"


@pytest.mark.parametrize(
    ("dist", "min_version"),
    [
        ("fastapi", "0.142"),
        ("langchain", "1.0"),
        ("langgraph", "1.0"),
        ("langfuse", "4.17"),
        ("mcp", "1.28"),
        ("taskiq", "0.13"),
        ("qdrant-client", "1.10"),
    ],
)
def test_versions_match_intended_api_lines(dist: str, min_version: str) -> None:
    """Защита от молчаливого ухода на линии с breaking changes.

    Если uv подтянет fastapi 0.143.x / langfuse 5.x / mcp 2.x / sqlalchemy 2.1.x,
    тест упадёт — см. README -> Design decisions.
    """
    from packaging.version import Version

    assert Version(md.version(dist)) >= Version(min_version)


@pytest.mark.parametrize(
    ("dist", "max_version"),
    [
        ("fastapi", "0.143"),  # 0.143.0: OTel-экспортёры стали opt-in
        ("langfuse", "5"),  # 5.x: новый API, нужен сервер v4.59+
        ("mcp", "2"),  # 2.x: удалён mcp.server.fastmcp
        ("sqlalchemy", "2.1"),  # 2.1.x: слишком свежая async-линия
    ],
)
def test_versions_stay_below_breaking_lines(dist: str, max_version: str) -> None:
    from packaging.version import Version

    assert Version(md.version(dist)) < Version(max_version)
