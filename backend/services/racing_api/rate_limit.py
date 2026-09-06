"""Client-side rate limiting.

The Racing API meters requests per plan. Being throttled mid-backfill is
expensive: a 429 costs a round trip, a retry delay, and — on a long historical
import — can cascade into hours of lost throughput. Pacing ourselves *below* the
limit is strictly cheaper than discovering it.

A token bucket is used rather than a fixed sleep so that short bursts (fetching
the eight races of a single meeting) proceed at full speed, while the sustained
average stays under the configured ceiling.
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType

from backend.utils.logging import get_logger

logger = get_logger(__name__, channel="api")


class AsyncTokenBucket:
    """Asynchronous token bucket.

    ``rate`` tokens are added per second up to a maximum of ``burst``. Each
    request consumes one token; when the bucket is empty, :meth:`acquire` sleeps
    exactly long enough for the next token to appear.
    """

    def __init__(self, rate: float, burst: int | None = None) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = float(rate)
        self.capacity = float(burst if burst is not None else max(1, int(rate)))
        self._tokens = self.capacity
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()
        self.total_waited_seconds = 0.0
        self.wait_count = 0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._updated_at
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._updated_at = now

    async def acquire(self, tokens: float = 1.0) -> float:
        """Block until ``tokens`` are available. Returns the seconds waited."""
        if tokens > self.capacity:
            raise ValueError(f"cannot acquire {tokens} tokens; capacity is {self.capacity}")

        waited = 0.0
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    if waited > 0:
                        self.total_waited_seconds += waited
                        self.wait_count += 1
                    return waited
                deficit = tokens - self._tokens
                delay = deficit / self.rate

            await asyncio.sleep(delay)
            waited += delay

    @property
    def available_tokens(self) -> float:
        self._refill()
        return self._tokens

    async def __aenter__(self) -> AsyncTokenBucket:
        await self.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


__all__ = ["AsyncTokenBucket"]
