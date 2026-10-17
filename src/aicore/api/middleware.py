"""Чистые ASGI-мидлвари приложения.

`CorrelationIdMiddleware` реализован на голом ASGI, а не через
`BaseHTTPMiddleware`: последний оборачивает ответ и исторически ломает
стриминговые ответы (SSE). Здесь мы только читаем заголовок запроса, ставим
`correlation_id` в contextvar (его подхватывает structlog-процессор) и
дописываем его в заголовки ответа.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from aicore.logging import get_correlation_id, reset_correlation_id, set_correlation_id

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

_INCOMING_HEADERS = (b"x-correlation-id", b"x-request-id")
_RESPONSE_HEADER = b"x-correlation-id"


class CorrelationIdMiddleware:
    """Ставит correlation_id из заголовка (или новый) и возвращает его клиенту."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming: str | None = None
        for name, value in scope.get("headers", []):
            if name.lower() in _INCOMING_HEADERS:
                incoming = value.decode("latin-1")
                break

        token = set_correlation_id(incoming)

        async def send_with_correlation(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                cid = get_correlation_id() or ""
                headers.append((_RESPONSE_HEADER, cid.encode("latin-1")))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_with_correlation)
        finally:
            reset_correlation_id(token)
