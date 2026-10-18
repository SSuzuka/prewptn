"""initial schema

Revision ID: fc1f60248123
Revises:
Create Date: 2026-10-17 20:42:01.460684
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "fc1f60248123"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NOW = sa.text("(CURRENT_TIMESTAMP)")


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("source", sa.String(length=1024), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "indexing", "indexed", "failed", name="document_status"),
            nullable=False,
        ),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("content_length", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_documents")),
        sa.UniqueConstraint("content_hash", name=op.f("uq_documents_content_hash")),
    )
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_documents_status"), ["status"], unique=False)

    op.create_table(
        "task_runs",
        sa.Column("task_type", sa.String(length=64), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "pending",
                "running",
                "succeeded",
                "failed",
                "dead_letter",
                "cancelled",
                name="task_status",
            ),
            nullable=False,
        ),
        sa.Column(
            "payload",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "result",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_usd", sa.Float(), nullable=True),
        sa.Column("queued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("broker_task_id", sa.String(length=255), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_task_runs")),
    )
    with op.batch_alter_table("task_runs", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_task_runs_broker_task_id"), ["broker_task_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_task_runs_idempotency_key"), ["idempotency_key"], unique=False
        )
        batch_op.create_index(
            "ix_task_runs_status_created_at", ["status", "created_at"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_task_runs_task_type"), ["task_type"], unique=False)

    op.create_table(
        "idempotency_keys",
        sa.Column("key", sa.String(length=255), nullable=False),
        sa.Column("scope", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.Uuid(), nullable=True),
        sa.Column("request_fingerprint", sa.String(length=64), nullable=True),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["task_runs.id"],
            name=op.f("fk_idempotency_keys_task_id_task_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_idempotency_keys")),
        sa.UniqueConstraint("key", "scope", name="uq_idempotency_keys_key_scope"),
    )
    with op.batch_alter_table("idempotency_keys", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_idempotency_keys_task_id"), ["task_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("idempotency_keys", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_idempotency_keys_task_id"))

    op.drop_table("idempotency_keys")
    with op.batch_alter_table("task_runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_task_runs_task_type"))
        batch_op.drop_index("ix_task_runs_status_created_at")
        batch_op.drop_index(batch_op.f("ix_task_runs_idempotency_key"))
        batch_op.drop_index(batch_op.f("ix_task_runs_broker_task_id"))

    op.drop_table("task_runs")
    with op.batch_alter_table("documents", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_documents_status"))

    op.drop_table("documents")

    # Postgres не удаляет enum-типы вместе с таблицами. Без явного DROP TYPE
    # повторный `alembic upgrade head` падает с `type "task_status" already
    # exists`. На SQLite native-enum нет — там шаг пропускается.
    if op.get_bind().dialect.name == "postgresql":
        op.execute("DROP TYPE IF EXISTS task_status")
        op.execute("DROP TYPE IF EXISTS document_status")
