"""FastAPI-зависимости: rate limiting по клиенту.

Лимитер в памяти процесса: скользящее окно на клиента. Для одного инстанса
этого достаточно; при нескольких репликах счётчик надо вынести в Redis (фаза 2).
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections import deque

from fastapi import HTTPException, Request, status

from aicore.config import get_settings


class RateLimiter:
    """Sliding-window лимитер запросов на клиента.

    The per-client deques are pruned periodically: without it the dict grows with
    every distinct client ever seen (even after their window empties) and a bot
    sweep turns it into an unbounded memory leak.
    """

    def __init__(
        self,
        per_minute: int,
        window: float = 60.0,
        *,
        max_clients: int = 10_000,
        sweep_every: int = 1_000,
    ) -> None:
        self.per_minute = per_minute
        self.window = window
        # `max_clients` is a safety valve, not an exact limit: under a flood of
        # unique IPs the oldest keys (by insertion order) are evicted, which can
        # drop a legitimate active client and hand it a false 429. That is the
        # intended trade-off — availability over an OOM — so keep it well above
        # the real working set, and move the counter to Redis for multi-replica.
        self._max_clients = max_clients
        self._sweep_every = max(1, sweep_every)
        self._hits: dict[str, deque[float]] = {}
        self._ops_since_sweep = 0
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        async with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = self._hits[key] = deque()
            while hits and now - hits[0] > self.window:
                hits.popleft()
            allowed = len(hits) < self.per_minute
            if allowed:
                hits.append(now)

            self._ops_since_sweep += 1
            if self._ops_since_sweep >= self._sweep_every:
                self._ops_since_sweep = 0
                self._sweep(now)
            return allowed

    def _sweep(self, now: float) -> None:
        """Drop clients whose window is empty, then cap the total key count."""
        stale = [
            key for key, hits in self._hits.items() if not hits or now - hits[-1] > self.window
        ]
        for key in stale:
            del self._hits[key]

        overflow = len(self._hits) - self._max_clients
        if overflow > 0:
            # dict preserves insertion order: drop the oldest keys first.
            for key in list(self._hits)[:overflow]:
                del self._hits[key]


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
    """Зависимость: 429, если клиент превысил лимит запросов в минуту.

    Клиент определяется по IP. За доверенным прокси (proxy_headers) берётся
    **правый** адрес из `X-Forwarded-For` — тот, который дописал ближайший
    доверенный прокси. Левые адреса задаёт сам клиент, поэтому брать их значит
    позволять обойти лимит подделкой заголовка.
    """
    limiter = get_rate_limiter()
    client = _client_key(request)
    if not await limiter.allow(client):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate limit exceeded",
            headers={"Retry-After": str(int(limiter.window))},
        )


def _client_key(request: Request) -> str:
    """Ключ клиента для лимитера: IP, с учётом доверенного прокси.

    `X-Forwarded-For` — это цепочка `client, proxy1, proxy2, ...`, куда КАЖДЫЙ
    узел дописывает источник справа. Клиент может подделать левые значения
    (или весь заголовок), но правый элемент дописан доверенным прокси — только
    ему и можно верить. Раньше брался левый (`split(",", 1)[0]`), и лимит
    обходился запросами с разными фейковыми XFF.
    """
    if get_settings().trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            last = forwarded.rsplit(",", 1)[-1].strip()
            if last:
                return last
    return request.client.host if request.client else "anonymous"


async def require_api_key(request: Request) -> None:
    """401, если включена аутентификация по ключу, а ключ не подошёл.

    Пустой список `api_keys` = аутентификация выключена: проект остаётся
    открытым для локального запуска (как и остальные in-memory дефолты), но
    закрытый режим включается одной переменной окружения. Сравнение —
    constant-time, чтобы не течь по таймингу.

    Сравниваем байты, а не `str`: HTTP-заголовок может содержать не-ASCII
    (например, `café`), а `secrets.compare_digest` для `str` требует
    ASCII-only и на таком ключе бросал `TypeError` -> 500 вместо 401.
    """
    keys = get_settings().api_keys
    if not keys:
        return
    presented = request.headers.get("x-api-key")
    if presented is None or not _key_matches(presented, keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def _key_matches(presented: str, keys: list[str]) -> bool:
    """Constant-time сравнение ключа; не-ASCII не роняет запрос."""
    presented_bytes = presented.encode("utf-8")
    return any(secrets.compare_digest(presented_bytes, key.encode("utf-8")) for key in keys)
