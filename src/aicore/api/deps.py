"""FastAPI-зависимости: rate limiting по клиенту.

Лимитер в памяти процесса: скользящее окно на клиента. Для одного инстанса
этого достаточно; при нескольких репликах счётчик надо вынести в Redis (фаза 2).
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status

from aicore.config import get_settings


class RateLimiter:
    """Sliding-window лимитер запросов на клиента."""

    def __init__(self, per_minute: int, window: float = 60.0) -> None:
        self.per_minute = per_minute
        self.window = window
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        async with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.per_minute:
                return False
            hits.append(now)
            return True


_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(get_settings().api_rate_limit_per_minute)
    return _limiter


def reset_rate_limiter() -> None:
    """Сбросить лимитер (тесты / смена настроек)."""
    global _limiter
    _limiter = None


async def rate_limit(request: Request) -> None:
    """Зависимость: 429, если клиент превысил лимит запросов в минуту."""
    limiter = get_rate_limiter()
    client = request.client.host if request.client else "anonymous"
    if not await limiter.allow(client):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": str(int(limiter.window))},
        )
