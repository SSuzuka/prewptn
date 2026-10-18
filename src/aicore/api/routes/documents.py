"""Роуты документов RAG: загрузка (→ задача индексации) и статус.

Идемпотентность — по содержимому: `content_hash` уникален, поэтому повторная
загрузка того же текста возвращает существующий документ и **не** жжёт
эмбеддинги повторно (та же идея, что `Idempotency-Key` у задач).
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from aicore.api.deps import rate_limit
from aicore.api.schemas import DocumentCreateRequest, DocumentView
from aicore.db.models import DocumentStatus
from aicore.db.session import get_session
from aicore.logging import get_logger
from aicore.rag.documents import DocumentService
from aicore.tasks.rag_index import enqueue_index_document
from aicore.tasks.service import TaskService

logger = get_logger(__name__)
router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def _require_uuid(document_id: str) -> None:
    try:
        UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="document_id must be a valid UUID") from None


@router.post(
    "/documents",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(rate_limit)],
    summary="Загрузить документ и поставить задачу индексации в RAG",
)
async def create_document(body: DocumentCreateRequest, session: SessionDep) -> DocumentView:
    docsvc = DocumentService(session)
    document, created = await docsvc.create_document(
        title=body.title, content=body.content, source=body.source
    )
    # Идемпотентный повтор не должен навсегда «запирать» упавший документ:
    # повторная загрузка того же текста после failed ставит новую задачу.
    if created or document.status == DocumentStatus.FAILED:
        # Текст документа НЕ кладём в payload: его читает индексер из строки
        # документа (`Document.content`). Иначе до 1 МБ ехало бы в Redis и в
        # JSONB `task_runs.payload` на каждый документ.
        task, _ = await TaskService(session).create_task(
            task_type="index_document",
            payload={
                "document_id": str(document.id),
                "title": document.title,
                "source": document.source,
            },
        )
        # Коммитим ДО постановки в очередь: задача читает документ отдельной
        # сессией и должна быть уже видна.
        await session.commit()
        await enqueue_index_document(str(task.id))
        await session.refresh(document)
    else:
        logger.info("document.idempotent_replay", document_id=str(document.id))
    return DocumentView.from_document(document)


@router.get(
    "/documents",
    dependencies=[Depends(rate_limit)],
    summary="Список документов RAG",
)
async def list_documents(
    session: SessionDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentView]:
    documents = await DocumentService(session).list_documents(limit=limit, offset=offset)
    return [DocumentView.from_document(document) for document in documents]


@router.get(
    "/documents/{document_id}",
    dependencies=[Depends(rate_limit)],
    summary="Статус индексации документа",
)
async def get_document(document_id: str, session: SessionDep) -> DocumentView:
    _require_uuid(document_id)
    document = await DocumentService(session).get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentView.from_document(document)
