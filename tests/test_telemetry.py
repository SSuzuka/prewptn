"""Tests for the no-op-safe telemetry wiring (OTel + Langfuse)."""

from __future__ import annotations

import pytest

from aicore.config import Settings
from aicore.telemetry import (
    configure_telemetry,
    get_telemetry,
    record_generation,
    reset_telemetry,
    span,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_telemetry()
    yield
    reset_telemetry()


def test_langfuse_disabled_without_keys() -> None:
    tel = configure_telemetry(Settings())
    assert tel.langfuse is None
    assert tel.tracer is not None


def test_record_generation_is_noop_when_disabled() -> None:
    configure_telemetry(Settings())
    # Must not raise even with a fully populated payload.
    record_generation(
        name="llm.complete",
        model="m",
        input={"prompt": "x"},
        output="y",
        prompt_tokens=1,
        completion_tokens=2,
        cost_usd=0.0,
        latency_ms=3,
    )


def test_span_sets_attributes_and_ignores_none() -> None:
    configure_telemetry(Settings())
    with span("unit.test", **{"k": "v", "skip": None}) as current:
        assert current is not None


def test_get_telemetry_configures_on_demand() -> None:
    assert get_telemetry().service_name == "aicore"


def test_langfuse_enabled_uses_get_client(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep the env mirroring inside this test from leaking into other tests.
    for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)

    sentinel = object()
    called: dict[str, bool] = {}

    def fake_get_client() -> object:
        called["yes"] = True
        return sentinel

    import langfuse

    monkeypatch.setattr(langfuse, "get_client", fake_get_client)

    tel = configure_telemetry(Settings(langfuse_public_key="pk", langfuse_secret_key="sk"))
    assert called.get("yes") is True
    assert tel.langfuse is sentinel
