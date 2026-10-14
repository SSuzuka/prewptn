"""Тесты API-каркаса: health-check отвечает без внешней инфраструктуры."""

import pytest
from fastapi.testclient import TestClient

from aicore.api.app import create_app
from aicore.config import reset_settings_cache


@pytest.fixture()
def client() -> TestClient:
    reset_settings_cache()
    with TestClient(create_app()) as c:
        yield c
    reset_settings_cache()


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "aicore"
    assert body["version"] == "0.1.0"


def test_health_reports_in_memory_backends(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["environment"] == "development"
    assert body["backends"] == {
        "broker": "memory",
        "db": "sqlite",
        "vector": "memory",
        "llm": "stub",
    }


def test_ready(client: TestClient) -> None:
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_version(client: TestClient) -> None:
    resp = client.get("/version")
    assert resp.status_code == 200
    assert resp.json() == {"name": "aicore", "version": "0.1.0"}


def test_openapi_schema_available(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["info"]["title"] == "AI Agent Platform"


def test_unknown_route_returns_404(client: TestClient) -> None:
    assert client.get("/nope").status_code == 404


def test_health_response_has_no_secrets(client: TestClient) -> None:
    """В ответе health не должно быть ни ключей, ни DSN с паролем."""
    blob = repr(client.get("/health").json())
    assert "sk-" not in blob
    assert "://" not in blob
