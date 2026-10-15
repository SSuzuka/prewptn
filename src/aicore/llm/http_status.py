"""Shared HTTP response classification for OpenAI-compatible clients.

Both the chat client (:mod:`aicore.llm.openai_compat`) and the embeddings client
(:mod:`aicore.rag.embeddings`) speak to OpenAI-compatible endpoints and must
classify upstream failures identically: retry timeouts / 429 / the 5xx family,
fail fast on other 4xx, and treat a malformed 2xx body as a provider glitch.
Keeping the status set and the decode/status helpers here stops the two layers
from silently drifting apart in retry behaviour.
"""

from __future__ import annotations

from typing import Any

import httpx

# Statuses worth retrying: request timeout/conflict, too-early, rate limit and
# the transient 5xx family. Everything else in 4xx is a caller problem.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def raise_for_status(
    response: httpx.Response,
    *,
    label: str,
    transient_error: type[Exception],
    permanent_error: type[Exception],
) -> None:
    """Raise the caller's transient/permanent error for a non-2xx response."""
    if response.status_code < 400:
        return
    message = f"{label} HTTP {response.status_code}: {response.text[:500]}"
    if response.status_code in RETRYABLE_STATUS:
        raise transient_error(message)
    raise permanent_error(message)


def decode_json_object(
    response: httpx.Response,
    *,
    label: str,
    transient_error: type[Exception],
    permanent_error: type[Exception],
) -> dict[str, Any]:
    """Decode a JSON object body, classifying bad JSON as transient."""
    try:
        data = response.json()
    except ValueError as exc:
        # A 2xx with a truncated body is a provider glitch, not our bug.
        raise transient_error(f"{label} returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise permanent_error(f"{label} returned a non-object JSON body")
    return data
