"""Тесты конфигурации и логирования.

Всё офлайн: без БД, очередей, Qdrant и LLM-ключей.
"""

import pytest
from pydantic import ValidationError

from aicore.config import Settings, get_settings, reset_settings_cache
from aicore.logging import (
    get_correlation_id,
    new_correlation_id,
    redact_secrets,
    reset_correlation_id,
    set_correlation_id,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    reset_settings_cache()
    yield
    reset_settings_cache()


class TestSettings:
    def test_singleton_is_cached(self) -> None:
        assert get_settings() is get_settings()

    def test_defaults_are_in_memory(self) -> None:
        cfg = Settings()
        assert cfg.broker_backend == "memory"
        assert cfg.llm_backend == "stub"
        assert cfg.vector_backend == "memory"
        assert cfg.uses_sqlite is True
        assert cfg.is_production is False

    def test_langfuse_disabled_without_keys(self) -> None:
        assert Settings().langfuse_enabled is False

    def test_langfuse_enabled_with_both_keys(self) -> None:
        cfg = Settings(langfuse_public_key="pk-lf-x", langfuse_secret_key="sk-lf-y")
        assert cfg.langfuse_enabled is True

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"llm_backend": "openai"}, True),
            ({"broker_backend": "redis"}, True),
            ({"vector_backend": "qdrant"}, True),
            ({}, False),
        ],
    )
    def test_is_production_reflects_backends(self, kwargs: dict, expected: bool) -> None:
        assert Settings(**kwargs).is_production is expected

    def test_api_key_alone_is_not_production(self) -> None:
        """Наличие ключа не делает прод — смотрят на бэкенды."""
        cfg = Settings(openai_api_key="sk-x")
        assert cfg.is_production is False

    def test_cors_accepts_comma_string(self) -> None:
        assert Settings(cors_origins="a, b ,c").cors_origins == ["a", "b", "c"]

    def test_cors_accepts_list(self) -> None:
        assert Settings(cors_origins=["a", "b"]).cors_origins == ["a", "b"]

    def test_log_level_is_upper_cased(self) -> None:
        assert Settings(log_level="debug").log_level == "DEBUG"

    def test_invalid_log_level_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(log_level="chatty")

    def test_invalid_broker_backend_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Settings(broker_backend="kafka")

    @pytest.mark.parametrize("field", ["embedding_dim", "task_max_retries"])
    def test_negative_numbers_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            Settings(**{field: -1})


class TestSecretRedaction:
    def test_redacted_dict_masks_keys(self) -> None:
        cfg = Settings(
            openai_api_key="sk-SECRET",
            langfuse_public_key="pk-lf-PUB",
            langfuse_secret_key="sk-lf-SEC",
        )
        blob = repr(cfg.redacted_dict())
        assert "sk-SECRET" not in blob
        assert "pk-lf-PUB" not in blob
        assert "sk-lf-SEC" not in blob

    def test_redacted_dict_masks_db_password_but_keeps_user(self) -> None:
        cfg = Settings(db_url="postgresql+asyncpg://bob:hunter2@db.internal:5432/aicore")
        result = cfg.redacted_dict()
        assert "hunter2" not in repr(result)
        assert "bob" in result["db_url"]

    def test_redacted_dict_safe_for_sqlite(self) -> None:
        cfg = Settings(db_url="sqlite+aiosqlite:///./local.db")
        assert cfg.redacted_dict()["db_url"] == "sqlite+aiosqlite:///./local.db"

    def test_redact_secrets_processor(self) -> None:
        event = {"event": "x", "api_key": "sk-leak", "password": "p", "answer": 42}
        out = redact_secrets(None, "info", event)
        assert out["api_key"] == "***"
        assert out["password"] == "***"
        assert out["answer"] == 42

    def test_redact_secrets_ignores_normal_fields(self) -> None:
        event = {"event": "x", "tokens": 10}
        assert redact_secrets(None, "info", event) == event


class TestCorrelationId:
    def test_set_and_reset(self) -> None:
        token = set_correlation_id(new_correlation_id())
        cid = get_correlation_id()
        assert isinstance(cid, str) and len(cid) == 32
        reset_correlation_id(token)
        assert get_correlation_id() is None

    def test_new_correlation_id_is_unique(self) -> None:
        assert new_correlation_id() != new_correlation_id()
