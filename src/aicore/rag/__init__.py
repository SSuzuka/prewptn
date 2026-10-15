"""RAG layer: chunking, embeddings, vector store, hybrid retrieval and context.

Provider choice is a config switch, exactly like the LLM layer:
  * ``embedding_backend`` — ``stub`` (deterministic, offline), ``local``
    (fastembed ONNX, free and key-less), ``openai`` (any ``/embeddings``);
  * ``vector_backend`` — ``memory`` (in-process Qdrant) or ``qdrant`` (server).

Callers depend on :class:`~aicore.rag.base.EmbeddingClient` and
:class:`~aicore.rag.retriever.Retriever`, never on a concrete backend, so dev,
tests and prod run the same code path with different wiring.
"""

from __future__ import annotations

from aicore.config import Settings, get_settings
from aicore.rag.base import (
    EmbeddingClient,
    EmbeddingError,
    EmbeddingPermanentError,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingTransientError,
    EmbeddingUsage,
    compute_embedding_cost_usd,
)
from aicore.rag.chunking import Chunk, chunk_text, estimate_tokens
from aicore.rag.context import Citation, ContextBundle, build_context, build_rag_prompt
from aicore.rag.documents import DocumentService, content_hash
from aicore.rag.retriever import Retriever
from aicore.rag.store import SearchHit, VectorPoint, VectorStore, make_point_id

__all__ = [
    "Chunk",
    "Citation",
    "ContextBundle",
    "DocumentService",
    "EmbeddingClient",
    "EmbeddingError",
    "EmbeddingPermanentError",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "EmbeddingTransientError",
    "EmbeddingUsage",
    "Retriever",
    "SearchHit",
    "VectorPoint",
    "VectorStore",
    "build_context",
    "build_embedding_client",
    "build_rag_prompt",
    "build_retriever",
    "chunk_text",
    "compute_embedding_cost_usd",
    "content_hash",
    "estimate_tokens",
    "get_retriever",
    "make_point_id",
    "reset_retriever",
    "set_retriever",
]


def build_embedding_client(settings: Settings | None = None) -> EmbeddingClient:
    """Build an embedding client from settings.

    The heavy ``fastembed`` import is lazy, so a stub-only install never needs the
    ``embeddings`` extra. ``httpx`` (openai backend) is not a concern: it already
    ships with the base stack as a hard dependency of ``qdrant-client``.
    """
    cfg = settings or get_settings()
    if cfg.embedding_backend == "local":
        from aicore.rag.embeddings import LocalEmbeddings

        return LocalEmbeddings(settings=cfg)
    if cfg.embedding_backend == "openai":
        from aicore.rag.embeddings import OpenAICompatibleEmbeddings

        return OpenAICompatibleEmbeddings(settings=cfg)

    from aicore.rag.stub import StubEmbeddings

    return StubEmbeddings(settings=cfg)


def build_retriever(settings: Settings | None = None) -> Retriever:
    """Build a retriever from settings (embedding client + vector store)."""
    cfg = settings or get_settings()
    return Retriever(
        embeddings=build_embedding_client(cfg),
        store=VectorStore.from_settings(cfg),
        settings=cfg,
    )


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """Process-wide retriever, built on first use."""
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()
    return _retriever


def set_retriever(retriever: Retriever) -> None:
    global _retriever
    _retriever = retriever


def reset_retriever() -> None:
    global _retriever
    _retriever = None
