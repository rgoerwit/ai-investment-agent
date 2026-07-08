"""Regression tests for the hard wall-clock timeout wrap around LLM ainvoke.

Background
----------
A single ``runnable.ainvoke()`` against ``langchain_google_genai`` plumbs
``api_timeout`` and ``api_retry_attempts`` into the underlying
``HttpOptions(timeout=…, retry_options=HttpRetryOptions(attempts=…))``.  When
``api_retry_attempts`` was 10 and ``api_timeout`` was 300s, a stuck Gemini
upstream could legitimately consume up to 10 × 300s ≈ 50 minutes per call
before raising — multiplied by our outer ``max_attempts=3`` retry loop, that
produced the multi-hour ticker stalls observed on HK tickers in May 2026.

These tests pin two defenses against that pathology:

1. ``config.api_retry_attempts`` defaults to a small value so SDK-level retry
   storms can't park a worker for an hour.
2. ``invoke_with_rate_limit_handling`` wraps ``runnable.ainvoke`` in
   ``run_with_hard_timeout`` using ``config.llm_call_hard_timeout_seconds`` so a
   hung provider SDK is force-cancelled and the outer retry loop can either
   reattempt or surface the failure.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.agents import invoke_with_rate_limit_handling
from src.config import config as settings_config


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """The breaker is process-global. Reset before each test so accumulated
    state from earlier hard-timeout exercises does not silently open a
    circuit for a later test reusing the same (context, provider, model)
    key."""
    from src.agents.circuit_breaker import reset_circuit_breaker_for_tests

    reset_circuit_breaker_for_tests()
    yield
    reset_circuit_breaker_for_tests()


class TestApiRetryAttemptsDefault:
    """The SDK retry budget must stay small to bound worst-case call duration.

    These tests check the *code default* on the Settings field, not the runtime
    resolved value (which may be overridden by the developer's local .env).
    """

    def _field_default(self, name: str):
        from src.config import Settings

        return Settings.model_fields[name].default

    def test_code_default_is_small_enough_to_bound_worst_case(self):
        # 10 retries × 300s timeout (api_timeout) = 50 min per single LLM call.
        # We cap retries low so the SDK can't hang on its own retries; outer
        # invoke_with_rate_limit_handling already retries 3× with backoff.
        default = self._field_default("api_retry_attempts")
        assert default <= 3, (
            f"api_retry_attempts default must stay small (≤3); got {default}. "
            "At higher values the Gemini SDK can park a single ainvoke() for "
            "tens of minutes via internal retries, defeating wall-clock "
            "timeouts. See May 2026 HK ticker slow-tail incident."
        )

    def test_hard_timeout_default_present_and_bounded(self):
        # The hard timeout is the load-bearing safety net; if it disappears or
        # is cranked unreasonably high the slow-tail returns.
        default = self._field_default("llm_call_hard_timeout_seconds")
        assert 60.0 <= default <= 1800.0, (
            f"llm_call_hard_timeout_seconds default {default} is out of the "
            "sane range [60s, 1800s]."
        )
        assert default == 120.0
        # Also verify the field exists on the runtime object (so the wrap can
        # actually look it up).
        assert hasattr(settings_config, "llm_call_hard_timeout_seconds")


class TestApexQuickHardTimeout:
    """The two gate-critical APEX seats get a larger --quick per-call budget so
    the flat quick cap (sized for cheap flash agents) can't guillotine the
    DATA_BLOCK / PM_BLOCK the hard gates depend on (2026-07-06 fix)."""

    def test_apex_seat_gets_apex_budget(self, monkeypatch):
        from src.agents.runtime import (
            APEX_SEAT_CONTEXTS,
            quick_mode_hard_timeout_seconds,
        )

        monkeypatch.setattr(
            settings_config, "apex_quick_llm_call_hard_timeout_seconds", 180.0
        )
        monkeypatch.setattr(
            settings_config, "quick_llm_call_hard_timeout_seconds", 60.0
        )
        for context in APEX_SEAT_CONTEXTS:
            assert quick_mode_hard_timeout_seconds(context, settings_config) == 180.0

    def test_non_apex_agent_gets_flat_quick_budget(self, monkeypatch):
        from src.agents.runtime import quick_mode_hard_timeout_seconds

        monkeypatch.setattr(
            settings_config, "apex_quick_llm_call_hard_timeout_seconds", 180.0
        )
        monkeypatch.setattr(
            settings_config, "quick_llm_call_hard_timeout_seconds", 60.0
        )
        assert quick_mode_hard_timeout_seconds("News Analyst", settings_config) == 60.0

    def test_apex_seat_contexts_match_prompt_agent_names(self):
        # APEX_SEAT_CONTEXTS must equal the on-disk agent_name of the two APEX
        # seats, or the larger budget silently stops applying after a rename.
        import json

        from src.agents.runtime import APEX_SEAT_CONTEXTS
        from src.llms import APEX_SEATS

        names = set()
        for seat in APEX_SEATS:
            key = "fundamentals_analyst" if seat == "senior_fundamentals" else seat
            with open(f"prompts/{key}.json") as fh:
                names.add(json.load(fh)["agent_name"])
        assert APEX_SEAT_CONTEXTS == names


class TestHardTimeoutWrap:
    """`invoke_with_rate_limit_handling` must enforce a wall-clock cap."""

    @pytest.mark.asyncio
    async def test_hung_ainvoke_is_force_cancelled_after_hard_timeout(self):
        """A coroutine that never resolves must not block longer than the cap."""
        from src.token_tracker import get_tracker

        tracker = get_tracker()
        tracker.reset()

        async def never_resolves(*_args, **_kwargs):
            # Awaiting an unresolved Future never returns; this simulates a
            # hung HTTP read inside the provider SDK.
            await asyncio.get_event_loop().create_future()
            return AIMessage(content="never")

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=never_resolves)

        # Patch the cap to a tiny value so the test runs fast.
        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 0.05):
            with pytest.raises(Exception) as exc_info:
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "x"},
                    max_attempts=1,  # don't burn time on retries
                    context="HardTimeoutTest",
                )

        # The hard-timeout path raises asyncio.TimeoutError; the outer wrapper
        # logs it as failure_kind='timeout' and re-raises.
        assert (
            isinstance(exc_info.value, asyncio.TimeoutError)
            or "timeout" in str(exc_info.value).lower()
        )
        attempt = tracker.get_total_stats()["call_attempts"][-1]
        assert attempt["failure_kind"] == "timeout"
        assert attempt["failure_origin"] == "hard_timeout"

    @pytest.mark.asyncio
    async def test_fast_ainvoke_is_unaffected(self):
        """Calls that resolve quickly must succeed normally even with cap set."""
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 1.0):
            result = await invoke_with_rate_limit_handling(
                runnable,
                {"input": "x"},
                max_attempts=1,
                context="HardTimeoutTest",
            )

        assert getattr(result, "content", None) == "ok"

    @pytest.mark.asyncio
    async def test_wrap_is_present_in_invoke_path(self):
        """Guard against silent removal of run_with_hard_timeout from runtime.py.

        Without this test, a refactor that drops the wrap would still pass the
        other tests because AsyncMock returns instantly. Here we observe that
        the wrap is invoked exactly once per attempt with the configured cap.
        """
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        seen_calls: list[dict] = []

        async def spy(coro, *, timeout, label):
            seen_calls.append({"timeout": timeout, "label": label})
            return await coro

        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 12.5):
            with patch("src.agents.runtime.run_with_hard_timeout", new=spy):
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "x"},
                    max_attempts=1,
                    context="WrapPresenceTest",
                )

        assert len(seen_calls) == 1, (
            "run_with_hard_timeout must wrap every ainvoke; if this fails the "
            "wall-clock guarantee was silently removed."
        )
        assert seen_calls[0]["timeout"] == 12.5
        # Label should encode context/provider/model so hard_timeout_exceeded
        # warnings are attributable.
        assert seen_calls[0]["label"].startswith("llm:WrapPresenceTest:")

    @pytest.mark.asyncio
    async def test_overall_timeout_shortens_effective_timeout(self):
        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        seen_calls: list[dict] = []

        async def spy(coro, *, timeout, label):
            seen_calls.append({"timeout": timeout, "label": label})
            return await coro

        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 12.5):
            with patch("src.agents.runtime.run_with_hard_timeout", new=spy):
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "x"},
                    max_attempts=1,
                    context="OverallTimeoutTest",
                    overall_timeout_seconds=0.5,
                )

        assert len(seen_calls) == 1
        assert 0 < seen_calls[0]["timeout"] <= 0.5

    @pytest.mark.asyncio
    async def test_hard_timeout_retries_via_outer_loop(self):
        """When attempt 1 hits the hard timeout, attempt 2 should run with backoff."""
        call_count = {"n": 0}

        async def first_hangs_then_succeeds(*_args, **_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                await asyncio.get_event_loop().create_future()
            return AIMessage(content="recovered")

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=first_hangs_then_succeeds)

        # Patch only the runtime module's asyncio.sleep so the outer backoff
        # is instant; the inner hang awaits an unresolved future, not sleep,
        # so it remains hung until the hard timeout fires.
        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 0.05):
            with patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock):
                with patch("src.agents.runtime.random.uniform", return_value=2.0):
                    result = await invoke_with_rate_limit_handling(
                        runnable,
                        {"input": "x"},
                        max_attempts=2,
                        context="HardTimeoutTest",
                    )

        assert call_count["n"] == 2
        assert getattr(result, "content", None) == "recovered"

    @pytest.mark.asyncio
    async def test_timeout_uses_transient_attempt_budget(self):
        async def always_times_out(*_args, **_kwargs):
            await asyncio.get_event_loop().create_future()

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=always_times_out)

        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 0.01):
            with patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(TimeoutError):
                    await invoke_with_rate_limit_handling(
                        runnable,
                        {"input": "x"},
                        max_attempts=3,
                        max_transient_attempts=2,
                        context="TransientBudgetTest",
                    )

        assert runnable.ainvoke.call_count == 2


def test_no_external_asyncio_timeout_wrapping_shared_llm_helper():
    """LLM call deadlines should flow through the shared runtime helper."""
    source_path = Path("src/agents/consultant_nodes.py")
    tree = ast.parse(source_path.read_text())
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncWith):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            is_asyncio_timeout = (
                isinstance(func, ast.Attribute)
                and func.attr == "timeout"
                and isinstance(func.value, ast.Name)
                and func.value.id == "asyncio"
            )
            if is_asyncio_timeout:
                offenders.append(node.lineno)

    assert offenders == []


# ---------------------------------------------------------------------------
# Provider partial-response detection (May 2026 2382.HK auditor truncation)
#
# Some providers occasionally return a "200 OK" with truncated content and
# either a missing or `length` finish_reason — the call looks "successful"
# but the response is half-baked. We detect the partial via finish_reason
# and surface it as a transient failure so the existing retry loop can
# reattempt before passing partial output downstream.
# ---------------------------------------------------------------------------


def _resp(content: str, finish_reason: object = "stop") -> AIMessage:
    """Construct an AIMessage with response_metadata.finish_reason set
    the way LangChain wraps OpenAI / Anthropic / Gemini responses."""
    msg = AIMessage(content=content)
    metadata: dict = {}
    if finish_reason is not _SENTINEL:
        metadata["finish_reason"] = finish_reason
    msg.response_metadata = metadata
    return msg


_SENTINEL = object()


class TestQuickModeFlexLatencyGuard:
    """Fix A3 step 4 (runtime quick-guard): a flex-tier latency timeout that
    reaches the retry loop in quick mode (transport fallback disabled, or the
    client-timeout/outer-cap race) must be non-retryable AND excluded from the
    circuit breaker — it is a queue event, not a provider fault. See the
    flex-fallback x timeout matrix rows 6-8."""

    @pytest.mark.asyncio
    async def test_quick_flex_timeout_is_non_retryable_and_not_breaker_counted(
        self, monkeypatch
    ):
        from unittest.mock import MagicMock

        from src.agents.circuit_breaker import (
            get_circuit_breaker,
            reset_circuit_breaker_for_tests,
        )
        from src.runtime_config import RuntimeConfig, use_runtime_config

        reset_circuit_breaker_for_tests()
        # Activate gemini flex + quick mode so the guard condition is met.
        monkeypatch.setattr(settings_config, "gemini_service_tier", "flex")
        rc = RuntimeConfig.from_config(settings_config).with_overrides(
            quick_mode_active=True
        )

        async def always_hangs(*_a, **_kw):
            await asyncio.get_event_loop().create_future()

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=always_hangs)

        breaker = get_circuit_breaker()
        record_spy = MagicMock(wraps=breaker.record_outcome)
        monkeypatch.setattr(breaker, "record_outcome", record_spy)

        with (
            patch.object(settings_config, "quick_llm_call_hard_timeout_seconds", 0.02),
            patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock),
            use_runtime_config(rc),
        ):
            with pytest.raises(TimeoutError):
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "x"},
                    max_attempts=3,
                    max_transient_attempts=2,
                    context="QuickFlexGuard",
                    provider="google",
                    model_name="gemini-3.5-flash",
                )

        # Non-retryable: exactly one SDK attempt despite the transient budget.
        assert runnable.ainvoke.call_count == 1
        # Breaker never told about a failure for this queue-timeout.
        failure_calls = [
            c for c in record_spy.call_args_list if c.kwargs.get("ok") is False
        ]
        assert failure_calls == []
        reset_circuit_breaker_for_tests()

    @pytest.mark.asyncio
    async def test_full_mode_flex_timeout_still_retries(self, monkeypatch):
        """Contrast: the guard is quick-gated. In full mode a flex timeout is a
        normal transient failure and still consumes the retry budget.

        The SDK raises the timeout directly here (rather than relying on the
        outer wrap) because in full mode the flex floor stretches the hard cap
        to 1350s — the point under test is the retry classification, not the
        wrap firing."""
        from src.agents.circuit_breaker import reset_circuit_breaker_for_tests

        reset_circuit_breaker_for_tests()
        monkeypatch.setattr(settings_config, "gemini_service_tier", "flex")
        # No quick RuntimeConfig bound => full mode.

        async def raises_timeout(*_a, **_kw):
            raise TimeoutError("flex attempt exceeded SDK client timeout")

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=raises_timeout)

        with patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError):
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "x"},
                    max_attempts=3,
                    max_transient_attempts=2,
                    context="FullFlexRetry",
                    provider="google",
                    model_name="gemini-3.5-flash",
                )

        assert runnable.ainvoke.call_count == 2
        reset_circuit_breaker_for_tests()


class TestProviderPartialResponseDetection:
    """Pin the finish_reason classifier behavior so future provider quirks
    don't quietly slip through."""

    def test_clean_stop_is_not_partial(self):
        from src.agents.runtime import _detect_provider_partial_response

        assert (
            _detect_provider_partial_response(_resp("done", finish_reason="stop"))
            is None
        )

    def test_anthropic_end_turn_is_not_partial(self):
        from src.agents.runtime import _detect_provider_partial_response

        assert (
            _detect_provider_partial_response(_resp("done", finish_reason="end_turn"))
            is None
        )

    def test_tool_calls_finish_reason_is_not_partial(self):
        """Agent loop continues with tool results; not a retry signal."""
        from src.agents.runtime import _detect_provider_partial_response

        assert (
            _detect_provider_partial_response(
                _resp("calling tool", finish_reason="tool_calls")
            )
            is None
        )

    def test_content_filter_is_not_partial(self):
        """Provider intentionally stopped — retry would just loop."""
        from src.agents.runtime import _detect_provider_partial_response

        assert (
            _detect_provider_partial_response(
                _resp("blocked", finish_reason="content_filter")
            )
            is None
        )

    def test_missing_finish_reason_is_partial(self):
        """The May 2026 2382.HK case: stream cut without finalizing."""
        from src.agents.runtime import _detect_provider_partial_response

        assert (
            _detect_provider_partial_response(_resp("partial", finish_reason=None))
            is not None
        )

    def test_length_finish_reason_is_partial(self):
        """Hit max_completion_tokens before finishing — surface to caller."""
        from src.agents.runtime import _detect_provider_partial_response

        result = _detect_provider_partial_response(
            _resp("ran out", finish_reason="length")
        )
        assert result is not None
        assert "length" in result

    def test_empty_content_with_no_metadata_is_partial(self):
        """Defensive: empty body and no metadata → suspect partial."""
        from src.agents.runtime import _detect_provider_partial_response

        msg = AIMessage(content="")
        # No response_metadata at all
        assert _detect_provider_partial_response(msg) is not None

    def test_bare_aimessage_with_content_is_NOT_partial(self):
        """Pin the regression that an over-aggressive partial detector
        (one that flags ANY missing finish_reason as partial) would have
        broken `tests/config/test_rate_limit_handling.py
        ::test_success_on_first_try`. Many call sites and tests construct
        plain AIMessages without setting response_metadata; those must
        flow through as clean responses, not trigger a phantom retry.
        """
        from src.agents.runtime import _detect_provider_partial_response

        msg = AIMessage(content="some normal content")
        # Default response_metadata is the empty dict; no provider tail.
        assert _detect_provider_partial_response(msg) is None

    def test_tool_call_response_with_empty_content_is_NOT_partial(self):
        """An active tool-loop step is not a partial response. The
        consultant + auditor agent loops produce responses where
        ``content`` is empty and ``tool_calls`` is populated — the next
        loop turn feeds tool results back into the model. Flagging these
        as partial would short-circuit the loop and break the consultant
        verification mitigations (see
        tests/advanced/test_verification_mitigations.py
        ::test_consultant_with_tools_executes_loop)."""
        from unittest.mock import MagicMock as _MagicMock

        from src.agents.runtime import _detect_provider_partial_response

        msg = _MagicMock()
        msg.content = ""
        msg.tool_calls = [{"name": "spot_check_metric", "args": {}, "id": "call_1"}]
        # MagicMock's response_metadata attribute returns another MagicMock
        # (not a dict). The detector tolerates that and falls through to
        # the tool_calls early-return.
        assert _detect_provider_partial_response(msg) is None

    def test_responses_api_completed_status_is_NOT_partial(self):
        """OpenAI's Responses API (`use_responses_api=True` +
        `output_version="responses/v1"`, used by the consultant, auditor,
        and editor LLMs) populates `response_metadata` with `status`
        rather than `finish_reason`. A `status="completed"` response is
        clean and must not trigger a retry.

        Pre-fix, every gpt-4o-mini auditor call was getting flagged as
        `provider_partial_response` with `finish_reason_missing` because
        the detector only knew about Chat-Completions-shaped metadata.
        """
        from src.agents.runtime import _detect_provider_partial_response

        msg = AIMessage(content="FORENSIC_DATA_BLOCK: ...")
        msg.response_metadata = {
            "id": "resp_abc",
            "object": "response",
            "model_name": "gpt-4o-mini",
            "model": "gpt-4o-mini",
            "status": "completed",
            "service_tier": "default",
        }
        assert _detect_provider_partial_response(msg) is None

    def test_responses_api_incomplete_status_is_partial_with_reason(self):
        """`status="incomplete"` with `incomplete_details.reason` is the
        Responses-API equivalent of `finish_reason="length"` or similar
        early-stop conditions. Surface the reason."""
        from src.agents.runtime import _detect_provider_partial_response

        msg = AIMessage(content="cut off mid-")
        msg.response_metadata = {
            "id": "resp_abc",
            "object": "response",
            "model_name": "gpt-4o-mini",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }
        result = _detect_provider_partial_response(msg)
        assert result is not None
        assert "max_output_tokens" in result

    def test_responses_api_failed_status_is_partial(self):
        """`status="failed"` is a hard provider error reported via the
        Responses-API metadata; treat as partial so the outer retry loop
        reattempts."""
        from src.agents.runtime import _detect_provider_partial_response

        msg = AIMessage(content="")
        msg.response_metadata = {
            "id": "resp_abc",
            "object": "response",
            "status": "failed",
        }
        result = _detect_provider_partial_response(msg)
        assert result is not None
        assert "failed" in result


class TestProviderPartialResponseRetry:
    @pytest.mark.asyncio
    async def test_partial_response_triggers_retry_and_recovers(self):
        """First call returns a partial; second returns a clean stop."""
        call_count = {"n": 0}

        async def fake_invoke(_input):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Stream-cut style: no finish_reason at all.
                return _resp("partial header", finish_reason=None)
            return _resp("clean output", finish_reason="stop")

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=fake_invoke)

        with patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agents.runtime.random.uniform", return_value=2.0):
                result = await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "x"},
                    max_attempts=2,
                    context="PartialRetryTest",
                )

        assert call_count["n"] == 2
        assert result.content == "clean output"

    @pytest.mark.asyncio
    async def test_clean_response_is_not_retried(self):
        """Regression guard: a clean response must NOT trigger an extra
        invoke. Otherwise every successful call costs 2x the API quota."""
        call_count = {"n": 0}

        async def fake_invoke(_input):
            call_count["n"] += 1
            return _resp("clean", finish_reason="stop")

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=fake_invoke)

        result = await invoke_with_rate_limit_handling(
            runnable,
            {"input": "x"},
            max_attempts=3,
            context="CleanResponseTest",
        )

        assert call_count["n"] == 1
        assert result.content == "clean"

    @pytest.mark.asyncio
    async def test_persistent_partial_after_max_attempts_returns_partial(self):
        """If every attempt is partial, return the last partial rather
        than retrying forever. The structural validators downstream
        (output_validation.detect_truncation, agent_invalid_structure)
        will then correctly fail-closed on it."""
        call_count = {"n": 0}

        async def fake_invoke(_input):
            call_count["n"] += 1
            return _resp("always partial", finish_reason=None)

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=fake_invoke)

        with patch("src.agents.runtime.asyncio.sleep", new_callable=AsyncMock):
            with patch("src.agents.runtime.random.uniform", return_value=2.0):
                result = await invoke_with_rate_limit_handling(
                    runnable,
                    {"input": "x"},
                    max_attempts=2,
                    context="PersistentPartialTest",
                )

        # On the FINAL attempt, the partial-response check is bypassed
        # (`attempt < max_attempts - 1` is False), so we return the
        # partial content rather than burning all retries.
        assert call_count["n"] == 2
        assert result.content == "always partial"


class TestCircuitBreakerIntegration:
    """P2-7: The runtime wires the breaker so chronic timeouts fast-fail."""

    @pytest.mark.asyncio
    async def test_three_timeouts_open_circuit_and_fourth_call_is_fast(self):
        from src.agents.circuit_breaker import (
            CircuitOpenError,
            get_circuit_breaker,
            reset_circuit_breaker_for_tests,
        )
        from src.token_tracker import get_tracker

        reset_circuit_breaker_for_tests()
        breaker = get_circuit_breaker()
        # Force a tight, deterministic threshold for the test.
        breaker.threshold = 3
        breaker.window_seconds = 300.0
        breaker.cool_off_seconds = 60.0
        breaker.reset()

        tracker = get_tracker()
        tracker.reset()

        async def always_hangs(*_a, **_kw):
            await asyncio.get_event_loop().create_future()
            return AIMessage(content="never")

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=always_hangs)

        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 0.05):
            with patch.object(settings_config, "llm_circuit_breaker_enabled", True):
                # Three attempts: each hangs to hard-timeout and counts as a
                # breaker-eligible failure. The hard-timeout wrap surfaces
                # TimeoutError (asyncio.TimeoutError aliases to builtin TimeoutError
                # on 3.11+).
                for _ in range(3):
                    with pytest.raises(TimeoutError):
                        await invoke_with_rate_limit_handling(
                            runnable,
                            {"input": "x"},
                            max_attempts=1,
                            context="BreakerAgent",
                            provider="google",
                            model_name="gemini-3.1-flash-lite",
                        )

                # Fourth call: breaker is open, must raise CircuitOpenError
                # immediately without invoking ainvoke.
                before_calls = runnable.ainvoke.call_count
                with pytest.raises(CircuitOpenError):
                    await invoke_with_rate_limit_handling(
                        runnable,
                        {"input": "x"},
                        max_attempts=1,
                        context="BreakerAgent",
                        provider="google",
                        model_name="gemini-3.1-flash-lite",
                    )
                assert runnable.ainvoke.call_count == before_calls

        attempts = tracker.get_total_stats()["call_attempts"]
        # Last attempt was the fast-fail; record_call_attempt tagged it
        # with the canonical circuit_open kind/origin.
        assert attempts[-1]["failure_kind"] == "circuit_open"
        assert attempts[-1]["failure_origin"] == "circuit_breaker"

        reset_circuit_breaker_for_tests()

    @pytest.mark.asyncio
    async def test_disabled_breaker_does_not_short_circuit(self):
        from src.agents.circuit_breaker import (
            CircuitOpenError,
            reset_circuit_breaker_for_tests,
        )

        reset_circuit_breaker_for_tests()

        async def always_hangs(*_a, **_kw):
            await asyncio.get_event_loop().create_future()
            return AIMessage(content="never")

        runnable = AsyncMock()
        runnable.ainvoke = AsyncMock(side_effect=always_hangs)

        with patch.object(settings_config, "llm_call_hard_timeout_seconds", 0.05):
            with patch.object(settings_config, "llm_circuit_breaker_enabled", False):
                # With breaker disabled, repeated timeouts still time out
                # normally — never as CircuitOpenError.
                for _ in range(5):
                    with pytest.raises(TimeoutError) as info:
                        await invoke_with_rate_limit_handling(
                            runnable,
                            {"input": "x"},
                            max_attempts=1,
                            context="BreakerDisabledAgent",
                            provider="google",
                            model_name="gemini-3.1-flash-lite",
                        )
                    assert not isinstance(info.value, CircuitOpenError)

        reset_circuit_breaker_for_tests()
