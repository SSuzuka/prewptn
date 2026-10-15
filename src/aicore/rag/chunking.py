"""Recursive character chunking for indexing.

Chunking is a first-class quality lever, not plumbing: the size and the
boundaries decide what the retriever can ever find. This module implements the
"safe default" — recursive splitting on paragraph / sentence / word boundaries,
then merging the pieces back up to ``chunk_size`` with a small ``chunk_overlap``.

Sizes are in **characters** and the token estimate is ~4 chars/token, so the
default 1000/100 lands close to the ~200–800 token range that code/text
retrieval benchmarks report as the sweet spot, without adding a tokenizer
dependency. Overlap is kept small on purpose: it helps preserve a boundary-
straddling sentence, but large overlaps mostly duplicate the index and cost
embeddings for no measured recall gain.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ordered from the strongest boundary to the weakest; "" is the hard-split
# fallback so a pathological single "word" longer than chunk_size still splits.
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    ". ",
    "! ",
    "? ",
    "; ",
    ", ",
    " ",
    "",
)


@dataclass(frozen=True, slots=True)
class Chunk:
    """One indexable piece of a document."""

    index: int
    text: str


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token); no tokenizer dependency."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[Chunk]:
    """Split ``text`` into ordered chunks.

    Raises :class:`ValueError` on a nonsensical configuration (non-positive
    size, or overlap >= size, which would never advance).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = text.strip()
    if not normalized:
        return []

    pieces = _split(normalized, chunk_size, separators)
    merged = _merge(pieces, chunk_size, chunk_overlap)
    return [Chunk(index=i, text=chunk) for i, chunk in enumerate(merged) if chunk.strip()]


def _split(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """Recursively split ``text`` until every piece is at most ``size`` chars."""
    if len(text) <= size:
        return [text]

    separator = separators[0]
    rest = separators[1:]

    if separator == "" or not rest:
        # No finer boundary left: hard-split by length.
        return [text[i : i + size] for i in range(0, len(text), size)]

    pieces = _split_keep(text, separator)
    out: list[str] = []
    for piece in pieces:
        if len(piece) <= size:
            out.append(piece)
        else:
            out.extend(_split(piece, size, rest))
    return out


def _split_keep(text: str, separator: str) -> list[str]:
    """Split on ``separator`` but re-attach it, so no text is lost."""
    parts = text.split(separator)
    return [part + separator for part in parts[:-1]] + [parts[-1]]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack pieces into chunks, seeding each new chunk with the tail."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if current and len(current) + len(piece) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = tail + piece
        else:
            current += piece
    if current.strip():
        chunks.append(current)
    return chunks
