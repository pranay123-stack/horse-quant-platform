"""Token bucket tests."""

from __future__ import annotations

import asyncio
import time

import pytest

from backend.services.racing_api.rate_limit import AsyncTokenBucket

pytestmark = pytest.mark.unit


def test_rate_must_be_positive():
    with pytest.raises(ValueError, match="rate must be positive"):
        AsyncTokenBucket(0)


async def test_burst_is_served_without_waiting():
    bucket = AsyncTokenBucket(rate=10, burst=5)
    started = time.monotonic()
    for _ in range(5):
        assert await bucket.acquire() == 0.0
    assert time.monotonic() - started < 0.05


async def test_exceeding_the_burst_waits():
    bucket = AsyncTokenBucket(rate=100, burst=2)
    await bucket.acquire()
    await bucket.acquire()

    waited = await bucket.acquire()
    assert waited > 0
    assert bucket.wait_count == 1
    assert bucket.total_waited_seconds > 0


async def test_sustained_rate_is_respected():
    """Six requests through a 2-capacity bucket at 50/s must take ~80ms."""
    bucket = AsyncTokenBucket(rate=50, burst=2)
    started = time.monotonic()
    for _ in range(6):
        await bucket.acquire()
    elapsed = time.monotonic() - started

    # 4 requests beyond the burst / 50 per second = 0.08s, with generous slack
    # for scheduler jitter on a loaded machine.
    assert 0.05 < elapsed < 0.6


async def test_tokens_refill_over_time():
    bucket = AsyncTokenBucket(rate=100, burst=2)
    await bucket.acquire()
    await bucket.acquire()
    assert bucket.available_tokens < 1

    await asyncio.sleep(0.05)
    assert bucket.available_tokens > 1


async def test_cannot_request_more_than_capacity():
    bucket = AsyncTokenBucket(rate=10, burst=2)
    with pytest.raises(ValueError, match="capacity"):
        await bucket.acquire(5)


async def test_concurrent_acquire_is_serialised():
    """Parallel callers must not double-spend the same token."""
    bucket = AsyncTokenBucket(rate=1000, burst=3)
    results = await asyncio.gather(*(bucket.acquire() for _ in range(10)))
    assert len(results) == 10
    assert bucket.available_tokens < 3


async def test_usable_as_a_context_manager():
    bucket = AsyncTokenBucket(rate=100, burst=2)
    async with bucket:
        pass
    assert bucket.available_tokens < 2


def test_default_capacity_tracks_the_rate():
    assert AsyncTokenBucket(rate=5).capacity == 5
    assert AsyncTokenBucket(rate=0.5).capacity == 1  # never below one
