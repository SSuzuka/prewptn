"""Vector store wrapper over Qdrant's async client.

One contract, two backends, chosen by ``vector_backend``:
  * ``memory`` — ``AsyncQdrantClient(":memory:")``: fully in-process, works
    offline in dev and tests, no server and no Docker;
  * ``qdrant`` — a real Qdrant server (``qdrant_url`` + optional ``qdrant_api_key``).

Chunk point IDs are deterministic (``uuid5(document_id, chunk_index)``), so
re-indexing the same document **overwrites** its points instead of duplicating
them — the same idempotency idea used for ``Document.content_hash``.

Only dense search lives here; hybrid re-ranking is in :mod:`aicore.rag.retriever`
so the store stays a thin, swappable adapter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from aicore.config import Settings, get_settings
from aicore.logging import get_logger

logger = get_logger(__name__)


def make_point_id(document_id: str, chunk_index: int) -> str:
    """Deterministic Qdrant point id for a chunk (enables idempotent re-index)."""
    return str(uuid5(NAMESPACE_URL, f"{document_id}:{chunk_index}"))


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """One chunk ready to be stored: id + vector + payload."""

    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchHit:
    """A retrieved chunk with its score and provenance."""

    id: str
    score: float
    text: str
    document_id: str
    chunk_index: int
    title: str | None = None
    source: str | None = None


class VectorStore:
    """Async wrapper around a Qdrant collection sized for ``dimension``."""

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        collection: str,
        dimension: int,
    ) -> None:
        self._client = client
        self._collection = collection
        self._dimension = dimension
        self._ensured = False

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> VectorStore:
        cfg = settings or get_settings()
        if cfg.vector_backend == "qdrant":
            client = AsyncQdrantClient(
                url=cfg.qdrant_url,
                api_key=cfg.qdrant_api_key or None,
                timeout=cfg.embedding_timeout_seconds,
            )
            logger.info(
                "rag.store.qdrant_server", url=cfg.qdrant_url, collection=cfg.qdrant_collection
            )
        else:
            # ":memory:" keeps the whole vector index inside the process.
            client = AsyncQdrantClient(":memory:")
            logger.warning(
                "rag.store.in_memory",
                note="vector_backend=memory is dev/test only: index is lost on restart",
            )
        return cls(client=client, collection=cfg.qdrant_collection, dimension=cfg.embedding_dim)

    async def ensure_collection(self) -> None:
        """Create the collection once, or validate an existing one."""
        if self._ensured:
            return
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            try:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=self._dimension, distance=Distance.COSINE),
                )
                logger.info(
                    "rag.store.collection_created",
                    collection=self._collection,
                    dimension=self._dimension,
                )
            except Exception:
                # Check-then-create is racy: another process (or coroutine on the
                # same instance) may have created it first. Tolerate that only
                # when the collection now exists; otherwise the error is real.
                if not await self._client.collection_exists(self._collection):
                    raise
                await self._assert_dimension()
        else:
            await self._assert_dimension()
        self._ensured = True

    async def _assert_dimension(self) -> None:
        """Fail fast if the existing collection's vector size differs from config.

        The collection is created from the static ``embedding_dim``, while a real
        embedder reports its own dimension. A mismatch (model/config drift) makes
        every upsert/search fail, so surface it here instead of on first write.
        """
        info = await self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        actual = getattr(vectors, "size", None)
        if actual is not None and int(actual) != self._dimension:
            raise RuntimeError(
                f"collection {self._collection!r} has vector size {actual}, but "
                f"embedding_dim is {self._dimension}; align the embedding model and "
                "config or recreate the collection"
            )

    async def upsert(self, points: list[VectorPoint]) -> int:
        """Insert/overwrite points; returns how many were written."""
        if not points:
            return 0
        await self.ensure_collection()
        await self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                for point in points
            ],
            wait=True,
        )
        return len(points)

    async def search(
        self,
        vector: list[float],
        *,
        limit: int,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        """Dense (cosine) search; optionally restrict to specific documents."""
        await self.ensure_collection()
        query_filter = None
        if document_ids:
            condition = (
                FieldCondition(key="document_id", match=MatchValue(value=document_ids[0]))
                if len(document_ids) == 1
                else FieldCondition(key="document_id", match=MatchAny(any=document_ids))
            )
            query_filter = Filter(must=[condition])
        response = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [self._to_hit(point) for point in response.points]

    async def delete_document(self, document_id: str) -> None:
        """Remove every chunk of a document (by payload filter)."""
        await self.ensure_collection()
        await self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
            wait=True,
        )

    async def document_point_ids(self, document_id: str) -> set[str]:
        """Ids of every stored point of one document (for resumable indexing)."""
        await self.ensure_collection()
        ids: set[str] = set()
        offset: Any = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                ),
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.update(str(point.id) for point in points)
            if offset is None:
                break
        return ids

    async def delete_points(self, ids: Iterable[str]) -> None:
        """Delete specific points by id (used to drop stale chunks)."""
        point_ids = [str(point_id) for point_id in ids]
        if not point_ids:
            return
        await self.ensure_collection()
        await self._client.delete(
            collection_name=self._collection,
            points_selector=point_ids,
            wait=True,
        )

    async def count(self) -> int:
        await self.ensure_collection()
        result = await self._client.count(collection_name=self._collection)
        return int(result.count)

    @staticmethod
    def _to_hit(point: Any) -> SearchHit:
        payload = dict(point.payload or {})
        return SearchHit(
            id=str(point.id),
            score=float(point.score),
            text=str(payload.get("text", "")),
            document_id=str(payload.get("document_id", "")),
            chunk_index=int(payload.get("chunk_index", 0)),
            title=payload.get("title"),
            source=payload.get("source"),
        )

    async def aclose(self) -> None:
        await self._client.close()
