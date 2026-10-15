"""Persistence for RAG documents and their indexing lifecycle.

``Document`` holds metadata only — the text lives in the vector store (per chunk)
and in the indexing task payload. Idempotency mirrors the task layer: a unique
``content_hash`` means re-uploading identical content returns the existing row
instead of re-embedding it (embeddings cost money; duplicates do not).

Status flow: ``pending → indexing → indexed`` (or ``failed`` on a permanent
error). The transition helpers are intentionally tiny so the same code runs from
the worker and from tests.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from aicore.db.models import Document, DocumentStatus
from aicore.logging import get_logger

logger = get_logger(__name__)


def normalize_content(content: str) -> str:
    """Canonical text (line endings, edges) — the identity used for hashing."""
    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


def content_hash(content: str) -> str:
    """Stable SHA-256 of the normalised content (line endings, edges)."""
    return hashlib.sha256(normalize_content(content).encode("utf-8")).hexdigest()


class DocumentService:
    """Operations over ``Document`` rows."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_document(
        self,
        *,
        title: str,
        content: str,
        source: str | None = None,
    ) -> tuple[Document, bool]:
        """Create a document row. Returns ``(document, created)``.

        If the same content is already known, the existing row is returned with
        ``created=False`` and nothing is re-indexed.
        """
        digest = content_hash(content)
        existing = await self.get_by_hash(digest)
        if existing is not None:
            logger.info("document.idempotent_replay", document_id=str(existing.id))
            return existing, False

        document = Document(
            title=title,
            source=source,
            content_hash=digest,
            status=DocumentStatus.PENDING,
            content_length=len(normalize_content(content)),
        )
        try:
            # Savepoint (not a full rollback): a lost race must not discard other
            # work the caller has already staged in this session.
            async with self.session.begin_nested():
                self.session.add(document)
                await self.session.flush()
        except IntegrityError:
            # A parallel upload won the unique(content_hash) race.
            existing = await self.get_by_hash(digest)
            if existing is None:
                raise
            logger.info("document.race_resolved", document_id=str(existing.id))
            return existing, False

        logger.info("document.created", document_id=str(document.id), title=title)
        return document, True

    async def get_document(self, document_id: UUID | str) -> Document | None:
        try:
            key = document_id if isinstance(document_id, UUID) else UUID(str(document_id))
        except ValueError:
            # A malformed id (e.g. a corrupted task payload) is "not found", not a
            # crash: callers such as the index worker must be able to dead-letter it.
            logger.warning("document.invalid_id", document_id=str(document_id))
            return None
        result = await self.session.execute(select(Document).where(Document.id == key))
        return result.scalar_one_or_none()

    async def get_by_hash(self, digest: str) -> Document | None:
        result = await self.session.execute(select(Document).where(Document.content_hash == digest))
        return result.scalar_one_or_none()

    async def list_documents(self, *, limit: int = 50, offset: int = 0) -> list[Document]:
        stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # ------------------------------------------------------------ transitions
    async def mark_indexing(self, document: Document) -> Document:
        document.status = DocumentStatus.INDEXING
        document.error = None
        await self.session.flush()
        logger.info("document.indexing", document_id=str(document.id))
        return document

    async def mark_indexed(self, document: Document, *, chunk_count: int) -> Document:
        document.status = DocumentStatus.INDEXED
        document.chunk_count = chunk_count
        document.error = None
        document.indexed_at = datetime.now(UTC)
        await self.session.flush()
        logger.info("document.indexed", document_id=str(document.id), chunks=chunk_count)
        return document

    async def mark_failed(self, document: Document, error: str) -> Document:
        document.status = DocumentStatus.FAILED
        document.error = error
        await self.session.flush()
        logger.error("document.failed", document_id=str(document.id), error=error)
        return document
