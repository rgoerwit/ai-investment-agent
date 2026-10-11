"""The root isolation fixtures: no real limiter wait, no shared-embedding leak.

Both guard process-global state that used to make a test's behaviour depend on
the tests before it: the Google process-fallback limiter (about 5 s per acquire
after its first token, 100+ s across three memory modules) and
``FinancialSituationMemory``'s class-level embeddings and Chroma clients.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.rate_limiters import InMemoryRateLimiter

from src import llms
from src.config import config
from src.llm_runtime import rate_limits
from src.memory import FinancialSituationMemory
from tests.helpers.instant_limiter import InstantLimiter

SENTINEL = object()


# --- limiters -----------------------------------------------------------------


def test_the_google_process_fallback_admits_every_request_at_once():
    limiter = rate_limits.limiter_for_binding(config, "google", None)

    assert isinstance(limiter, InstantLimiter)
    for _ in range(25):  # the real limiter would need minutes for these
        assert limiter.acquire() is True
    assert limiter.acquired == 25


def test_every_rate_limited_vendor_gets_an_instant_fallback_that_counts():
    for vendor in ("google", *rate_limits.NON_GOOGLE_RATE_LIMITED_VENDORS):
        limiter = rate_limits.limiter_for_binding(config, vendor, None)
        assert isinstance(limiter, InstantLimiter), vendor
        assert asyncio.run(limiter.aacquire(blocking=True)) is True
        assert limiter.acquired == 1, vendor


def test_the_fallback_table_is_keyed_and_reused_within_a_test():
    first = rate_limits.limiter_for_binding(config, "google", None)
    assert rate_limits.limiter_for_binding(config, "google", None) is first
    assert rate_limits.limiter_for_binding(config, "openai", None) is not first


def test_a_limiter_acquired_in_one_test_does_not_carry_its_count_into_the_next_a():
    rate_limits.limiter_for_binding(config, "google", None).acquire()


def test_a_limiter_acquired_in_one_test_does_not_carry_its_count_into_the_next_b():
    assert rate_limits.limiter_for_binding(config, "google", None).acquired == 0


def test_the_global_gemini_limiter_proxy_resolves_to_an_instant_limiter():
    assert llms.GLOBAL_RATE_LIMITER._instance is None  # still lazy
    assert isinstance(llms.GLOBAL_RATE_LIMITER._get_instance(), InstantLimiter)
    assert asyncio.run(llms.GLOBAL_RATE_LIMITER.aacquire()) is True


def test_the_lazy_openai_limiter_is_instant_when_a_ceiling_is_configured():
    assert config.openai_rpm_limit is not None
    assert isinstance(llms._get_openai_rate_limiter(), InstantLimiter)


@pytest.mark.real_rate_limiter
def test_the_opt_out_marker_keeps_the_real_pacing_limiter():
    limiter = rate_limits.create_process_rate_limiter(15)

    assert isinstance(limiter, InMemoryRateLimiter)
    assert limiter.requests_per_second == pytest.approx(0.2)
    assert isinstance(
        rate_limits.limiter_for_binding(config, "google", None), InMemoryRateLimiter
    )


# --- shared memory state ------------------------------------------------------
# Order-dependent by design: _a leaves state behind, _b (collected after it, same
# module, no -p randomly) must find the class clean.


def test_a_test_may_populate_the_shared_memory_clients_a():
    FinancialSituationMemory._shared_embeddings = SENTINEL
    FinancialSituationMemory._shared_embeddings_available = True
    FinancialSituationMemory._shared_embeddings_key = ("p", "m", 1, "k")
    FinancialSituationMemory._shared_chroma_client = SENTINEL
    FinancialSituationMemory._shared_chroma_key = "somewhere"


def test_the_next_test_finds_the_shared_memory_clients_reset_b():
    assert FinancialSituationMemory._shared_embeddings is None
    assert FinancialSituationMemory._shared_embeddings_available is False
    assert FinancialSituationMemory._shared_embeddings_key is None
    assert FinancialSituationMemory._shared_chroma_client is None
    assert FinancialSituationMemory._shared_chroma_key is None
