"""A rate limiter that never waits, and counts the requests it admitted.

The real process-fallback limiter paces Google at ``RPM * 0.8 / 60`` requests a
second (about 5 s per acquire after the first token at 15 RPM), so any unit test
that sends several provider requests through it waits for real. The root
``_instant_process_limiters`` fixture in ``tests/conftest.py`` installs this class
in its place for every non-integration test; a test that needs a pacing limiter
builds an ``InMemoryRateLimiter`` itself.
"""

from __future__ import annotations

from langchain_core.rate_limiters import BaseRateLimiter


class InstantLimiter(BaseRateLimiter):
    """Admits every request at once; ``acquired`` counts them."""

    def __init__(self) -> None:
        self.acquired = 0

    def acquire(self, *, blocking: bool = True) -> bool:
        self.acquired += 1
        return True

    async def aacquire(self, *, blocking: bool = True) -> bool:
        self.acquired += 1
        return True
