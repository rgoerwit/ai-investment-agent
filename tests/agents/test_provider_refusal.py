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

from types import SimpleNamespace
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
from src.runtime_diagnostics import classify_failure, is_provider_content_block


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


# Refusal bodies as the vendors actually emit them. The first two are verbatim
# from saved artifacts (AGS.SI 2026-07-19 and 6782.TW 2026-07-29) — before this
# suite existed, only the Moonshot one classified correctly.
OBSERVED_REFUSAL_BODIES = {
    "zai_glm_1301": (
        "Error code: 400 - {'contentFilter': [{'level': 1, 'role': 'assistant'}], "
        "'error': {'code': '1301', 'message': 'System detected potentially unsafe "
        "or sensitive content in input or generation.'}}"
    ),
    "zai_glm_chinese_only": "Error code 400: {'code': '1301', 'message': '敏感内容'}",
    "moonshot_high_risk": (
        "Error code: 400 - {'error': {'code': 400, 'message': 'The request was "
        "rejected because it was considered high risk', 'param': 'prompt', "
        "'type': 'content_filter'}}"
    ),
    "deepseek_content_risk": (
        "Error code: 400 - {'error': {'message': 'Content Exists Risk', "
        "'type': 'invalid_request_error'}}"
    ),
    "qwen_data_inspection": (
        "Error code: 400 - {'error': {'code': 'data_inspection_failed'}}"
    ),
    "openai_content_filter": "Error code: 400 - {'error': {'type': 'content_filter'}}",
    "azure_content_policy": (
        "Error code: 400 - The response was filtered due to the prompt triggering "
        "Azure OpenAI's content management policy."
    ),
    "vertex_403_prohibited": (
        "403 Forbidden: response blocked, finish_reason: PROHIBITED_CONTENT"
    ),
}


class TestChineseProviderRefusalVocabulary:
    """One predicate recognizes every vendor reachable through OPENAI_API_BASE.

    The refusal machinery was built against OpenAI/Gemini wording only, so a
    Z.AI/GLM or DeepSeek refusal fell through to ``bad_request`` — which is *not*
    breaker-excluded, meaning repeated refusals fast-failed sibling agents.
    """

    @pytest.mark.parametrize("name", sorted(OBSERVED_REFUSAL_BODIES))
    def test_observed_body_is_a_content_block(self, name: str):
        assert is_provider_content_block(OBSERVED_REFUSAL_BODIES[name]) is True

    @pytest.mark.parametrize("name", sorted(OBSERVED_REFUSAL_BODIES))
    def test_observed_body_classifies_non_retryable(self, name: str):
        details = classify_failure(
            RuntimeError(OBSERVED_REFUSAL_BODIES[name]), provider="openai"
        )
        assert details.kind == "provider_safety_block"
        assert details.retryable is False

    @pytest.mark.parametrize(
        "spelling",
        ["content_filter", "contentFilter", "content filter", "Content-Filter"],
    )
    def test_punctuation_and_camel_case_fold_to_one_marker(self, spelling: str):
        """Normalization is why the marker list does not carry per-vendor spellings."""
        assert is_provider_content_block(f"Error code: 400 - {spelling}: blocked")

    def test_safety_block_outranks_auth_when_both_markers_present(self):
        """A 403-shaped refusal must not read as an auth problem: auth_error is
        breaker-counted and sends the operator to check API keys."""
        details = classify_failure(
            RuntimeError(OBSERVED_REFUSAL_BODIES["vertex_403_prohibited"])
        )
        assert details.kind == "provider_safety_block"


class TestStatusCodesAreMatchedAsNumbers:
    """Bare substring status matching found codes inside unrelated numbers."""

    def test_executive_order_14032_is_not_an_auth_error(self):
        """'executive order 14032' contains '403' and is verbatim CMIC prompt
        text — exactly the span a content filter echoes back in its rejection."""
        details = classify_failure(
            RuntimeError(
                "Error code: 400 - request rejected; prompt referenced "
                "executive order 14032"
            )
        )
        assert details.kind != "auth_error"
        assert details.kind == "bad_request"

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("Error code: 401 - Incorrect API key provided", "auth_error"),
            ("Error code: 403 - Forbidden", "auth_error"),
            ("Error code: 429 - rate limit reached", "rate_limit"),
            ("Error code: 500 - internal error", "server_error"),
            ("Error code: 503 - service unavailable", "server_error"),
            ("Error code: 400 - unknown parameter", "bad_request"),
        ],
    )
    def test_real_status_codes_still_classify(self, body: str, expected: str):
        assert classify_failure(RuntimeError(body)).kind == expected

    def test_embedded_digits_do_not_trigger_a_status_branch(self):
        assert classify_failure(RuntimeError("job 15000 failed")).kind == (
            "unknown_provider_error"
        )

    @pytest.mark.parametrize(
        "body",
        [
            "revenue was 500 million",
            "debt was 403 million",
            "operating income 502 million yen",
            "EPS 400 KRW",
            "value 429",
            # Structured magnitudes: a generic ':'/'='/'-' introducer read all of
            # these as statuses. Anchoring the class-name arm to ^identifier: is
            # what keeps a JSON key or an assignment from qualifying.
            '{"revenue": 500}',
            '{"operating_income": 502}',
            "debt = 403 million",
            "metric=429",
            "pe_ratio=400",
        ],
    )
    def test_financial_prose_is_not_an_http_status(self, body: str):
        """This system's exception text is full of three-digit financial
        magnitudes, and a content filter's rejection body echoes prompt spans
        back. Reading those as statuses changes retry and breaker behaviour."""
        assert classify_failure(RuntimeError(body)).kind == "unknown_provider_error"

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            # The introducer shapes actually observed across the 685 error
            # messages persisted in results/.
            ("Error code: 400 - bad param", "bad_request"),
            ("{'error': {'code': 500, 'message': 'x'}}", "server_error"),
            ("{'error': {'status': 503}}", "server_error"),
            ("ClientError: 429 Too Many Requests", "rate_limit"),
            ("APIError: 401 unauthorized", "auth_error"),
            ("HTTP 504 upstream", "server_error"),
        ],
    )
    def test_real_introducer_shapes_still_classify(self, body: str, expected: str):
        assert classify_failure(RuntimeError(body)).kind == expected

    @pytest.mark.parametrize(
        "class_name", ["Exception", "ServiceUnavailable", "APIStatusError"]
    )
    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            ("429 rate limited", "rate_limit"),
            ("503 The service is currently unavailable", "server_error"),
            ("500 Internal error encountered", "server_error"),
        ],
    )
    def test_status_leading_the_message_survives_any_class_name(
        self, class_name: str, body: str, expected: str
    ):
        """``classify_failure`` matches ``combined``, which prefixes the exception
        class name — so a bare-status message is never at position 0, and classes
        like ``ServiceUnavailable`` carry no code/error keyword. Missing these
        makes a real 429/503 non-retryable, which is worse than the mis-read the
        introducer requirement prevents.
        """
        exc_type = type(class_name, (Exception,), {})
        details = classify_failure(exc_type(body))
        assert details.kind == expected
        assert details.retryable is True


class TestOneRefusalPredicate:
    """The APAC node's detector and the global classifier share one vocabulary."""

    def test_apac_detector_delegates(self):
        from src.agents.apac_specialist_node import _is_glm_1301_policy_block

        for body in OBSERVED_REFUSAL_BODIES.values():
            assert _is_glm_1301_policy_block(RuntimeError(body)) is True

    def test_bare_policy_code_without_context_is_not_a_block(self):
        assert is_provider_content_block("order 1301 completed") is False

    def test_code_embedded_in_a_longer_number_is_not_a_block(self):
        assert is_provider_content_block("processed 131013 sensitive records") is False

    def test_bare_word_safety_is_not_a_block(self):
        """A report discussing safety must never read as a provider refusal."""
        assert is_provider_content_block("workplace safety metrics improved") is False


class TestRefusalDetectionAdaptsToRewording:
    """Detection keys on token proximity and codes, never on whole sentences.

    Provider safety layers are bolted on and reworded often, so a phrase list
    goes stale silently — the failure being detected here is a *missed* refusal,
    which is the expensive direction (it counts against the circuit breakers and
    fast-fails sibling agents).
    """

    @pytest.mark.parametrize(
        "body",
        [
            # None of these are wordings the detector was built against.
            "Request denied: generated content violates our usage guidelines",
            "{'code': 'E_MODERATION', 'message': 'output withheld'}",
            "{'contentPolicyViolation': true}",
            "output suppressed: content flagged by upstream review",
            "content refused by the provider review layer",
        ],
    )
    def test_unseen_wordings_still_detected(self, body: str):
        assert is_provider_content_block(body) is True

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("content_filter", True),
            ("contentFilter", True),
            ("content filter", True),
            ("Content-Filter", True),
            ("CONTENT_FILTER", True),
        ],
    )
    def test_spelling_variants_need_no_separate_entry(self, spelling, expected):
        assert is_provider_content_block(f"Error code: 400 - {spelling}") is expected


class TestRefusalDetectionDoesNotCollideWithEquityVocabulary:
    """This system analyses risk and safety for a living.

    "safety" and "risk" are ordinary subject matter here, and a content filter's
    rejection body routinely echoes the prompt span that offended it — so the
    weaker rejection words are admitted only next to "content", never next to
    "safety".
    """

    @pytest.mark.parametrize(
        "body",
        [
            "product safety risk assessment unavailable",
            "echo: workplace safety policy disclosure missing",
            "ValueError: risk_penalty missing for flag CONTENT_GAP",
            "ToolException: fetch_reference_content failed after 3 attempts",
            "ValueError: INSUFFICIENT_CONTENT returned for url",
            "http error: content_length mismatch in response body",
            "JSONDecodeError: Expecting value: line 1 column 1",
        ],
    )
    def test_domain_prose_is_not_a_refusal(self, body: str):
        assert is_provider_content_block(body) is False

    @pytest.mark.parametrize(
        "body",
        [
            # src/tooling/ has its own content-inspection pipeline, ~245 mentions
            # of "inspection". A failure in *our* inspector is not a provider
            # refusal, so "inspection" only counts beside the vendor's "data"
            # qualifier (Qwen's data_inspection_failed).
            "AttributeError: 'InspectionService' object has no attribute 'execute'",
            "RuntimeError: ContentInspectionHook failed to initialize",
            "ValueError: inspection_service returned None",
            "TypeError: guardrail config missing",
        ],
    )
    def test_our_own_inspection_pipeline_is_not_a_provider_refusal(self, body: str):
        assert is_provider_content_block(body) is False

    def test_the_vendor_qualifier_still_matches(self):
        assert is_provider_content_block("{'code': 'data_inspection_failed'}") is True

    @pytest.mark.parametrize(
        ("body", "expected_kind"),
        [
            ("Error code: 401 - Incorrect API key provided", "auth_error"),
            ("ReadTimeout: request timed out after 120s", "timeout"),
            ("Error code: 429 - rate limit reached", "rate_limit"),
            ("Error code: 500 - internal server error", "server_error"),
        ],
    )
    def test_ordinary_failures_keep_their_kind(self, body: str, expected_kind: str):
        assert classify_failure(RuntimeError(body)).kind == expected_kind


class TestEndpointHostIdentifiesTheVendor:
    """'Which vendor served this?' without a vendor lookup table.

    ``provider`` names the transport family and is "openai" for every
    OpenAI-compatible endpoint, so it cannot answer this question.
    """

    @pytest.mark.parametrize(
        ("base_url", "expected"),
        [
            ("https://api.z.ai/api/paas/v4", "api.z.ai"),
            ("https://api.moonshot.cn/v1", "api.moonshot.cn"),
            ("https://api.deepseek.com", "api.deepseek.com"),
            (
                "https://a-vendor-we-have-never-seen.example/v1",
                "a-vendor-we-have-never-seen.example",
            ),
        ],
    )
    def test_host_extracted_from_any_endpoint(self, base_url: str, expected: str):
        details = classify_failure(
            RuntimeError("Error code: 400 - content_filter"),
            provider="openai",
            base_url=base_url,
        )
        assert details.endpoint_host == expected
        assert details.provider == "openai"

    def test_absent_base_url_is_none_not_an_error(self):
        assert classify_failure(RuntimeError("boom")).endpoint_host is None

    def test_get_base_url_reads_the_live_client(self):
        from src.runtime_diagnostics import get_base_url

        assert get_base_url(SimpleNamespace(openai_api_base="https://x.test/v1")) == (
            "https://x.test/v1"
        )
        assert (
            get_base_url(
                SimpleNamespace(
                    root_client=SimpleNamespace(base_url="https://y.test/v1")
                )
            )
            == "https://y.test/v1"
        )
        assert get_base_url(SimpleNamespace()) is None

    def test_endpoint_host_never_carries_a_credential_or_path(self):
        """``get_base_url`` returns the FULL url. A base URL may embed a key
        (``https://user:key@host/v1``) or a query string, so any log field named
        endpoint_host must be the parsed host — one definition, no leak."""
        from src.runtime_diagnostics import get_base_url, get_endpoint_host

        llm = SimpleNamespace(
            openai_api_base="https://user:s3cret@api.z.ai/api/paas/v4?k=1"
        )
        assert get_endpoint_host(llm) == "api.z.ai"
        assert "s3cret" in get_base_url(llm)  # the raw value is unsafe by design
        assert get_endpoint_host(SimpleNamespace()) is None

    def test_failure_details_and_log_helper_agree(self):
        from src.runtime_diagnostics import get_endpoint_host

        url = "https://user:s3cret@api.moonshot.cn/v1"
        details = classify_failure(RuntimeError("boom"), base_url=url)
        assert details.endpoint_host == get_endpoint_host(
            SimpleNamespace(openai_api_base=url)
        )
        assert "s3cret" not in str(details.endpoint_host)


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
