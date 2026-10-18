"""Helpers for turning settings values into Alembic/DB config values.

Kept separate from ``migrations/env.py`` (which needs an Alembic context to
import) so the escaping rule is unit-testable.
"""

from __future__ import annotations


def escape_percent(url: str) -> str:
    """Экранировать `%` для configparser (иначе `%` в пароле ломает Alembic).

    `Config.set_main_option` прогоняет значение через configparser-интерполяцию,
    поэтому DSN вида ``postgresql+asyncpg://user:p%40ss@host/db`` падает с
    ``ValueError: invalid interpolation syntax``. Удваивание ``%`` — штатный
    способ отключить интерполяцию для этого значения.
    """
    return url.replace("%", "%%")
