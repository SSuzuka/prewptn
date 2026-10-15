"""Tools the agents can call.

Every tool is ``async`` on purpose: the agent graph is driven through the async
path (``ainvoke``) inside the worker, and a sync-only tool would raise
``StructuredTool does not support sync invocation``. The RAG tool reuses the
existing :class:`~aicore.rag.retriever.Retriever`, so "knowledge base access" is
the same hybrid retrieval the ``/v1/documents`` pipeline builds — the agent does
not get a second, divergent retrieval stack.
"""

from __future__ import annotations

from langchain_core.tools import tool

from aicore.logging import get_logger
from aicore.metrics import record_rag_fallback
from aicore.rag import get_retriever

logger = get_logger(__name__)


@tool
async def search_knowledge_base(query: str) -> str:
    """Search the indexed knowledge base and return the most relevant passages.

    Use it for factual questions that must be grounded in the ingested
    documents. Returns numbered passages with their source document, or an
    explicit "no results" message when nothing matches.
    """
    try:
        hits = await get_retriever().retrieve(query)
    except Exception:  # noqa: BLE001 — retrieval blip must not fail the whole run
        # Same degradation rule as `tasks/llm.py:_with_rag`: a broken vector
        # store should not abort the agent run and trigger a full broker retry
        # that re-bills every model call already made. Return a string so the
        # supervisor can finish and say the lookup failed. Don't log the query
        # itself (may contain PII); its length is enough to distinguish cases.
        logger.exception("rag.tool_retrieve_failed", query_chars=len(query))
        record_rag_fallback(reason="agent_tool_retrieve_error")
        return (
            "Knowledge base is temporarily unavailable. Answer from general "
            "knowledge and state that the knowledge-base lookup failed."
        )
    if not hits:
        return "No relevant passages were found in the knowledge base."
    return "\n\n".join(
        f"[{index}] (doc {hit.document_id}#{hit.chunk_index}) {hit.text}"
        for index, hit in enumerate(hits, start=1)
    )
