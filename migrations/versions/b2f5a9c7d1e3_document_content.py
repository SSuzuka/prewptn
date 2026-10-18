"""store document text on the row, not in the task payload

Revision ID: b2f5a9c7d1e3
Revises: fc1f60248123
Create Date: 2026-10-18 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b2f5a9c7d1e3"
down_revision: str | None = "fc1f60248123"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Текст документа (до 1 МБ) раньше ехал в payload задачи индексации и,
    # значит, через Redis и в JSONB `task_runs.payload`. Храним его на строке
    # документа: очередь несёт только document_id, а индексер читает текст из
    # БД. Колонка nullable — старые строки (и dev-схема через create_all) её
    # заполнят при повторной загрузке.
    op.add_column("documents", sa.Column("content", sa.Text(), nullable=True))


def downgrade() -> None:
    # batch_alter_table: SQLite не умеет DROP COLUMN вне batch-режима.
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_column("content")
