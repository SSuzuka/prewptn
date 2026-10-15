"""Turn retrieved chunks into a bounded, citable context block.

The naive "concatenate the top-k chunks" approach is what makes RAG slow and
lossy: vanilla pipelines can push 10k–30k tokens of context into a model whose
useful window is far smaller, and answer quality drops past a "context cliff".
So context assembly is its own step with two jobs:

  * **budget** — never emit more than ``max_chars`` of context;
  * **provenance** — every chunk is numbered ``[1]``, ``[2]`` … and carries its
    source, so the answer can cite where a claim came from.

Duplicate chunks (the same text retrieved twice, or overlap re-surfacing) are
collapsed before they eat the budget.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from aicore.rag.store import SearchHit


@dataclass(frozen=True, slots=True)
class Citation:
    """Where a numbered context block came from."""

    marker: int
    document_id: str
    chunk_index: int
    score: float
    title: str | None = None
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """A ready-to-inject context block plus its citations."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    used_chunks: int = 0
    truncated: bool = False

    def __bool__(self) -> bool:
        return bool(self.text)


def build_context(hits: Iterable[SearchHit], *, max_chars: int = 6000) -> ContextBundle:
    """Assemble a deduplicated, budget-bounded context from retrieved chunks."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    seen: set[str] = set()
    blocks: list[str] = []
    citations: list[Citation] = []
    used = 0
    consumed = 0
    truncated = False

    for hit in hits:
        text = hit.text.strip()
        if not text:
            continue
        fingerprint = text.lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        marker = used + 1
        label = hit.title or hit.source or hit.document_id
        block = f"[{marker}] ({label}) {text}"
        remaining = max_chars - consumed
        if len(block) > remaining:
            if blocks:
                truncated = True
                break
            # Первый блок сам больше бюджета: обрезаем, но не превышаем его.
            block = block[:remaining]
            truncated = True

        blocks.append(block)
        citations.append(
            Citation(
                marker=marker,
                document_id=hit.document_id,
                chunk_index=hit.chunk_index,
                score=hit.score,
                title=hit.title,
                source=hit.source,
            )
        )
        # +2 — разделитель "\n\n" между блоками, чтобы join не вылез за бюджет.
        consumed += len(block) + 2
        used += 1

    return ContextBundle(
        text="\n\n".join(blocks),
        citations=citations,
        used_chunks=used,
        truncated=truncated,
    )


def build_rag_prompt(question: str, context: ContextBundle) -> str:
    """Build the grounded prompt: retrieved context first, question last."""
    if not context:
        return question
    return (
        "Use the numbered context below to answer the question. "
        "Cite the sources you used with their [n] markers. "
        "If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context.text}\n\n"
        f"Question: {question}"
    )
