"""Retrieval engine: index chunks and answer queries with hybrid ranking.

Two responsibilities, both needed to keep the pipeline honest:

  * :meth:`Retriever.index` — chunk → embed → upsert, with deterministic point
    ids so re-indexing a document overwrites rather than duplicates;
  * :meth:`Retriever.retrieve` — embed the query, run dense search, then
    optionally fuse a lexical ranking via **Reciprocal Rank Fusion (RRF)**.

Why the lexical half exists: purely semantic search misses exact tokens —
error codes, identifiers, function names — that share meaning-level similarity
with everything else. RRF merges the dense ranking with a token-overlap ranking
without needing score calibration, and it is deterministic and offline, so it
works on the in-memory store too. Full sparse-vector/BM25 fusion inside Qdrant
is a phase-2 upgrade; this is the cheap version that captures most of the win.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from aicore.config import Settings, get_settings
from aicore.logging import get_logger
from aicore.metrics import record_retrieval
from aicore.rag.base import EmbeddingClient, EmbeddingRequest
from aicore.rag.chunking import Chunk
from aicore.rag.store import SearchHit, VectorPoint, VectorStore, make_point_id

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# OpenAI-compatible /embeddings endpoints cap inputs per request; keep batches
# well under it so the same code works for local and hosted backends.
_EMBED_BATCH = 64


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class Retriever:
    """Chunk index + hybrid retrieval over one vector collection."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingClient,
        store: VectorStore,
        settings: Settings | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._store = store
        self._cfg = settings or get_settings()

    @property
    def store(self) -> VectorStore:
        return self._store

    async def index(
        self,
        *,
        document_id: str,
        chunks: Iterable[Chunk],
        title: str | None = None,
        source: str | None = None,
    ) -> int:
        """Embed and store the chunks of one document; returns the count."""
        chunk_list = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunk_list:
            return 0

        # Возобновляемая индексация: point ids детерминированы, поэтому ретрай
        # переиспользует уже записанные чанки и эмбеддит только недостающие —
        # временный сбой не жжёт эмбеддинги повторно.
        existing = await self._store.document_point_ids(document_id)
        current_ids = {make_point_id(document_id, chunk.index) for chunk in chunk_list}
        pending = [
            chunk for chunk in chunk_list if make_point_id(document_id, chunk.index) not in existing
        ]

        for start in range(0, len(pending), _EMBED_BATCH):
            batch = pending[start : start + _EMBED_BATCH]
            response = await self._embeddings.embed(
                EmbeddingRequest(texts=[chunk.text for chunk in batch])
            )
            points = [
                VectorPoint(
                    id=make_point_id(document_id, chunk.index),
                    vector=vector,
                    payload={
                        "document_id": document_id,
                        "chunk_index": chunk.index,
                        "text": chunk.text,
                        "title": title,
                        "source": source,
                    },
                )
                for chunk, vector in zip(batch, response.vectors, strict=True)
            ]
            await self._store.upsert(points)

        # Точки, которых больше нет в новой версии документа, — устаревшие.
        stale = existing - current_ids
        if stale:
            await self._store.delete_points(stale)

        # Число актуальных чанков документа (не «записанных сейчас»: при
        # полном идемпотентном ре-индексе новых точек нет, но документ остаётся
        # проиндексированным целиком).
        chunk_count = len(current_ids)
        logger.info("rag.indexed", document_id=document_id, chunks=chunk_count)
        return chunk_count

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        """Return the most relevant chunks for ``query`` (hybrid ranking)."""
        query = query.strip()
        if not query:
            return []

        top_k = limit or self._cfg.rag_retrieve_limit
        fetch_k = max(self._cfg.rag_retrieve_fetch_k, top_k)

        embedding = await self._embeddings.embed(EmbeddingRequest(texts=[query]))
        if not embedding.vectors:
            return []
        hits = await self._store.search(
            embedding.vectors[0], limit=fetch_k, document_ids=document_ids
        )
        hits = [hit for hit in hits if hit.score >= self._cfg.rag_min_score]

        if self._cfg.rag_hybrid_enabled:
            hits = _fuse(hits, query, self._cfg.rag_rrf_k)

        result = hits[:top_k]
        record_retrieval(hybrid=self._cfg.rag_hybrid_enabled, returned=len(result))
        return result

    async def aclose(self) -> None:
        await self._embeddings.aclose()
        await self._store.aclose()


def _fuse(hits: list[SearchHit], query: str, rrf_k: int) -> list[SearchHit]:
    """Reciprocal Rank Fusion of the dense ranking and a token-overlap ranking."""
    if len(hits) < 2:
        return hits

    query_tokens = _tokens(query)
    if not query_tokens:
        return hits

    dense_rank = {hit.id: rank for rank, hit in enumerate(hits)}
    lexical = sorted(
        hits,
        key=lambda hit: len(query_tokens & _tokens(hit.text)) / len(query_tokens),
        reverse=True,
    )
    lexical_rank = {hit.id: rank for rank, hit in enumerate(lexical)}

    def score(hit: SearchHit) -> float:
        return 1.0 / (rrf_k + dense_rank[hit.id] + 1) + 1.0 / (rrf_k + lexical_rank[hit.id] + 1)

    return sorted(hits, key=score, reverse=True)
