"""Provider refusal / safety-block diagnostics.

A returned response that the provider blocked or refused (Gemini
``finish_reason="SAFETY"``/``RECITATION``, OpenAI ``content_filter`` /
structured ``.refusal``, Anthropic ``stop_reason="refusal"``, Gemini prompt
block) must be surfaced as an explicit, non-retryable ``provider_safety_block``
failure — not silently returned as a clean (empty) success. A refusal that
*raises* (e.g. a Kimi/Moonshot HTTP 400 ``content_filter``) must classify the
same way rather than falling into ``bad_request``/``unknown_provider_error``.

This is the observability seam: the block flows through the shared failure
machinery (dedicated ``llm_refusal_detected`` event, one diagnostic-ledger
entry under ``provider_safety_block``, and — via the node-level
``except -> failure_artifact`` wrapper — the saved artifact), with no prompt
contents in the logs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

import src.agents.runtime as runtime_mod
from src.agents import invoke_with_rate_limit_handling
from src.agents.runtime import (
    ProviderRefusalError,
    RefusalSignal,
    _detect_provider_refusal,
)
from src.config import config as settings_config
from src.runtime_diagnostics import classify_failure


@pytest.fixture(autouse=True)
def _reset_state():
    """Both breakers and the token tracker are process-global; reset around
    each test so accumulated state can't bleed across tests."""
    from src.agents.circuit_breaker import reset_circuit_breaker_for_tests
    from src.token_tracker import get_tracker

    reset_circuit_breaker_for_tests()
    get_tracker().reset()
    yield
    reset_circuit_breaker_for_tests()
    get_tracker().reset()


def _msg(content: str, *, metadata: dict | None = None, **attrs) -> AIMessage:
    msg = AIMessage(content=content)
    msg.response_metadata = metadata or {}
    for key, value in attrs.items():
        setattr(msg, key, value)
    return msg


class TestDetectProviderRefusal:
    """Metadata-only detector — one row per real provider shape."""

    def test_gemini_safety_response(self):
        signal = _detect_provider_refusal(
            _msg("", metadata={"finish_reason": "SAFETY"})
        )
        assert signal == RefusalSignal("safety", "response")

    def test_gemini_recitation_response(self):
        signal = _detect_provider_refusal(
            _msg("", metadata={"finish_reason": "RECITATION"})
        )
        assert signal == RefusalSignal("recitation", "response")

    def test_gemini_prohibited_content_response(self):
        signal = _detect_provider_refusal(
            _msg("", metadata={"finish_reason": "PROHIBITED_CONTENT"})
        )
        assert signal == RefusalSignal("prohibited_content", "response")

    def test_openai_chat_content_filter_finish_reason(self):
        signal = _detect_provider_refusal(
            _msg("", metadata={"finish_reason": "content_filter"})
        )
        assert signal == RefusalSignal("content_filter", "response")

    def test_openai_responses_api_refusal(self):
        signal = _detect_provider_refusal(
            _msg(
                "",
                metadata={
                    "status": "incomplete",
                    "incomplete_details": {"reason": "refusal"},
                },
            )
        )
        assert signal == RefusalSignal("refusal", "response")

    def test_openai_chat_structured_refusal_field(self):
        signal = _detect_provider_refusal(
            _msg("", additional_kwargs={"refusal": "I can't help with that."})
        )
        assert signal == RefusalSignal("refusal", "response")

    def test_anthropic_stop_reason_refusal(self):
        signal = _detect_provider_refusal(_msg("", metadata={"stop_reason": "refusal"}))
        assert signal == RefusalSignal("refusal", "response")

    def test_gemini_prompt_block_is_request_stage(self):
        signal = _detect_provider_refusal(
            _msg("", metadata={"prompt_feedback": {"block_reason": "SAFETY"}})
        )
        assert signal == RefusalSignal("prompt_blocked:SAFETY", "request")

    def test_normal_stop_is_not_a_refusal(self):
        assert (
            _detect_provider_refusal(_msg("done", metadata={"finish_reason": "stop"}))
            is None
        )

    def test_responses_api_length_incomplete_is_not_a_refusal(self):
        # An early-stop for length is a partial, not a content refusal — it
        # must stay on the existing partial/retry path.
        assert (
            _detect_provider_refusal(
                _msg(
                    "cut off",
                    metadata={
                        "status": "incomplete",
                        "incomplete_details": {"reason": "max_output_tokens"},
                    },
                )
            )
            is None
        )

    def test_bare_message_is_not_a_refusal(self):
        assert _detect_provider_refusal(AIMessage(content="normal content")) is None


class TestClassifyRefusalException:
    """A refusal that surfaces as a raised exception classifies as
    provider_safety_block — the Kimi/Moonshot 400 content_filter case."""

    def test_kimi_content_filter_400(self):
        exc = RuntimeError(
            "Error code: 400 - content_filter: The request was rejected "
            "because it was considered high risk"
        )
        details = classify_failure(exc, provider="openai", model_name="kimi-k3")
        assert details.kind == "provider_safety_block"
        assert details.retryable is False

    def test_our_refusal_error_classifies(self):
        details = classify_failure(
            ProviderRefusalError(RefusalSignal("safety", "response"))
        )
        assert details.kind == "provider_safety_block"
        assert details.retryable is False

    def test_plain_bad_request_still_bad_request(self):
        exc = RuntimeError("Error code: 400 - invalid_request_error: bad param")
        assert classify_failure(exc).kind == "bad_request"

    def test_unknown_error_unchanged(self):
        assert classify_failure(RuntimeError("something odd")).kind == (
            "unknown_provider_error"
        )


class TestRefusalArtifactPersistence:
    """A refused seat degrades through the node-level failure_artifact wrapper,
    which stamps the kind + reason into the persisted ArtifactStatus — with no
    schema change (Step 4, verify-only)."""

    def test_failure_artifact_carries_provider_safety_block(self):
        from src.runtime_diagnostics import failure_artifact

        result = failure_artifact(
            "fundamentals_report",
            ProviderRefusalError(RefusalSignal("safety", "response")),
            provider="google",
        )
        status = result["artifact_statuses"]["fundamentals_report"]
        assert status["error_kind"] == "provider_safety_block"
        assert status["ok"] is False
        assert status["retryable"] is False
        assert "provider_safety_block" in status["message"]


class _FakeRunnable:
    """Minimal ainvoke stand-in that returns a fixed response and counts calls."""

    def __init__(self, response):
        self._response = response
        self.call_count = 0

    async def ainvoke(self, _input_data):
        self.call_count += 1
        return self._response


class TestRefusalInvokeIntegration:
    @pytest.mark.asyncio
    async def test_safety_response_raises_and_records_once(self):
        from src.token_tracker import get_tracker

        runnable = _FakeRunnable(_msg("", metadata={"finish_reason": "SAFETY"}))
        secret_prompt = "PRIVATE_PROMPT_TEXT_do_not_log"

        # Patch the module logger directly (deterministic regardless of the
        # global structlog cache-on-first-use config, which defeats
        # structlog.testing.capture_logs once the logger has been used).
        with patch.object(runtime_mod, "logger", MagicMock()) as mock_logger:
            with pytest.raises(ProviderRefusalError) as exc_info:
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"messages": [AIMessage(content=secret_prompt)]},
                    context="Portfolio Manager",
                    provider="google",
                    model_name="gemini-3.1-pro-preview",
                )

        # Non-retryable: exactly one provider call.
        assert runnable.call_count == 1
        assert exc_info.value.reason_code == "safety"
        assert exc_info.value.stage == "response"

        # No success was logged.
        info_events = [c.args[0] for c in mock_logger.info.call_args_list if c.args]
        assert "llm_call_success" not in info_events

        # Exactly one dedicated refusal event, with safe fields and no prompt.
        refusal_calls = [
            c
            for c in mock_logger.warning.call_args_list
            if c.args and c.args[0] == "llm_refusal_detected"
        ]
        assert len(refusal_calls) == 1
        kwargs = refusal_calls[0].kwargs
        assert kwargs["reason_code"] == "safety"
        assert kwargs["stage"] == "response"
        assert kwargs["provider"] == "google"
        assert secret_prompt not in repr(kwargs)

        stats = get_tracker().get_total_stats()
        assert stats["failed_by_kind"].get("provider_safety_block") == 1
        last = stats["call_attempts"][-1]
        assert last["failure_kind"] == "provider_safety_block"
        assert last["retryable"] is False

    @pytest.mark.asyncio
    async def test_tool_call_turn_with_empty_content_is_not_a_refusal(self):
        # A paused-for-tools turn legitimately has empty content; it must not
        # be mistaken for a safety block.
        response = _msg(
            "",
            metadata={"finish_reason": "tool_calls"},
            tool_calls=[{"name": "spot_check", "args": {}, "id": "call_1"}],
        )
        runnable = _FakeRunnable(response)
        result = await invoke_with_rate_limit_handling(
            runnable,
            {"messages": []},
            context="External Consultant",
            provider="openai",
            model_name="gpt-5.4",
        )
        assert result is response
        assert runnable.call_count == 1

    @pytest.mark.asyncio
    async def test_refusals_do_not_open_the_circuit_breaker(self):
        # A content block is not a provider-health fault; repeated refusals
        # must not trip the breaker and fast-fail sibling agents.
        for _ in range(6):
            runnable = _FakeRunnable(_msg("", metadata={"finish_reason": "SAFETY"}))
            with pytest.raises(ProviderRefusalError):
                await invoke_with_rate_limit_handling(
                    runnable,
                    {"messages": []},
                    context="Portfolio Manager",
                    provider="google",
                    model_name="gemini-3.1-pro-preview",
                )
            # Never a CircuitOpenError — the call was actually dispatched.
            assert runnable.call_count == 1
