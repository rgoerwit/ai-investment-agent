import socket

from src.runtime_diagnostics import (
    FUNDAMENTALS_SYNC_FIELDS,
    OPTIONAL_PUBLISHABLE_ARTIFACTS,
    QUICK_OPTIONAL_PUBLISHABLE_ARTIFACTS,
    QUICK_REQUIRED_PUBLISHABLE_ARTIFACTS,
    REQUIRED_PUBLISHABLE_ARTIFACTS,
    SYNC_CHECK_FIELDS,
    build_analysis_validity,
    classify_failure,
    failure_artifact,
    get_artifact_status,
    get_optional_publishable_artifacts,
    get_required_publishable_artifacts,
    is_artifact_complete,
)


class TestRuntimeFailureClassification:
    def test_classifies_dns_resolution(self):
        exc = socket.gaierror(8, "nodename nor servname provided, or not known")

        details = classify_failure(
            exc,
            provider="google",
            model_name="gemini-3-flash-preview",
        )

        assert details.kind == "dns_resolution"
        assert details.provider == "google"
        assert details.retryable is True

    def test_classifies_rate_limit(self):
        exc = Exception("HTTP 429: Too Many Requests")

        details = classify_failure(exc, provider="openai", model_name="gpt-4o")

        assert details.kind == "rate_limit"
        assert details.provider == "openai"
        assert details.retryable is True

    def test_classifies_local_rate_limiter_type_error_as_application_error(self):
        exc = TypeError(
            "'_LazyRateLimiterProxy' object does not support the asynchronous context manager protocol"
        )

        details = classify_failure(
            exc, provider="google", model_name="gemini-embedding-001"
        )

        assert details.kind == "application_error"
        assert details.provider == "google"
        assert details.retryable is False


class TestAnalysisValidity:
    @staticmethod
    def _make_result(*, quick_mode: bool, value_trap_ok: bool = True) -> dict:
        value_trap_status = {"ok": value_trap_ok, "content": "value trap"}
        if not value_trap_ok:
            value_trap_status = {
                "complete": True,
                "ok": False,
                "error_kind": "dns_resolution",
                "provider": "google",
            }

        return {
            "market_report": "market",
            "sentiment_report": "sentiment",
            "news_report": "news",
            "value_trap_report": "value trap" if value_trap_ok else "",
            "pre_screening_result": "PASS",
            "fundamentals_report": "### --- START DATA_BLOCK ---\nSECTOR: Industrials\nRAW_HEALTH_SCORE: 5/12\nADJUSTED_HEALTH_SCORE: 41.7%\nRAW_GROWTH_SCORE: 1/6\nADJUSTED_GROWTH_SCORE: 16.7%\nUS_REVENUE_PERCENT: Not disclosed\n### --- END DATA_BLOCK ---",
            "final_trade_decision": "VERDICT: BUY",
            "run_summary": {"quick_mode": quick_mode},
            "artifact_statuses": {
                "market_report": {"ok": True, "content": "market"},
                "sentiment_report": {"ok": True, "content": "sentiment"},
                "news_report": {"ok": True, "content": "news"},
                "value_trap_report": value_trap_status,
                "fundamentals_report": {
                    "ok": True,
                    "content": "### --- START DATA_BLOCK ---\nSECTOR: Industrials\nRAW_HEALTH_SCORE: 5/12\nADJUSTED_HEALTH_SCORE: 41.7%\nRAW_GROWTH_SCORE: 1/6\nADJUSTED_GROWTH_SCORE: 16.7%\nUS_REVENUE_PERCENT: Not disclosed\n### --- END DATA_BLOCK ---",
                },
                "final_trade_decision": {"ok": True, "content": "VERDICT: BUY"},
            },
        }

    def test_publishable_requires_valid_pm_and_fundamentals(self):
        result = self._make_result(quick_mode=False)

        validity = build_analysis_validity(result)

        assert validity["publishable"] is True
        assert validity["has_data_block"] is True

    def test_publishable_ignores_unparseable_datablock_mentions(self):
        result = {
            "market_report": "market",
            "sentiment_report": "sentiment",
            "news_report": "news",
            "value_trap_report": "value trap",
            "pre_screening_result": "PASS",
            "fundamentals_report": "The analyst discusses the DATA_BLOCK but never emits the fenced section.",
            "final_trade_decision": "VERDICT: BUY",
            "run_summary": {"quick_mode": False},
            "artifact_statuses": {
                "market_report": {"ok": True, "content": "market"},
                "sentiment_report": {"ok": True, "content": "sentiment"},
                "news_report": {"ok": True, "content": "news"},
                "value_trap_report": {"ok": True, "content": "value trap"},
                "fundamentals_report": {
                    "ok": True,
                    "content": "The analyst discusses the DATA_BLOCK but never emits the fenced section.",
                },
                "final_trade_decision": {"ok": True, "content": "VERDICT: BUY"},
            },
        }

        validity = build_analysis_validity(result)

        assert validity["has_data_block"] is False

    def test_publishable_fails_with_invalid_pm_artifact(self):
        result = self._make_result(quick_mode=False)
        result["final_trade_decision"] = ""
        result["artifact_statuses"]["final_trade_decision"] = {
            "ok": False,
            "error_kind": "dns_resolution",
            "provider": "google",
        }

        validity = build_analysis_validity(result)

        assert validity["publishable"] is False
        assert "final_trade_decision" in validity["required_failures"]

    def test_publishable_allows_optional_failure(self):
        result = self._make_result(quick_mode=False)
        result["artifact_statuses"]["auditor_report"] = {
            "complete": True,
            "ok": False,
            "error_kind": "timeout",
            "provider": "openai",
        }

        validity = build_analysis_validity(result)

        assert validity["publishable"] is True
        assert validity["required_failures"] == {}
        assert "auditor_report" in validity["optional_failures"]

    def test_publishable_quick_mode_does_not_require_value_trap_report(self):
        result = self._make_result(quick_mode=True, value_trap_ok=False)

        validity = build_analysis_validity(result)

        assert validity["publishable"] is True
        assert "value_trap_report" not in validity["required_failures"]
        assert "value_trap_report" in validity["optional_failures"]

    def test_publishable_deep_mode_still_requires_value_trap_report(self):
        result = self._make_result(quick_mode=False, value_trap_ok=False)

        validity = build_analysis_validity(result)

        assert validity["publishable"] is False
        assert "value_trap_report" in validity["required_failures"]


class TestArtifactStatus:
    def test_failure_artifact_is_complete_but_invalid(self):
        result = failure_artifact(
            "legal_report",
            "Legal counsel unavailable",
            fallback_content='{"pfic_status":"UNCERTAIN"}',
        )

        status = get_artifact_status(result, "legal_report")

        assert status.complete is True
        assert status.ok is False
        assert status.error_kind == "application_error"
        assert is_artifact_complete(result, "legal_report") is True
        assert result["legal_report"] == '{"pfic_status":"UNCERTAIN"}'


class TestArtifactPolicy:
    def test_graph_artifacts_are_covered_by_policy(self):
        assert FUNDAMENTALS_SYNC_FIELDS == {
            "raw_fundamentals_data",
            "foreign_language_report",
            "legal_report",
        }
        assert SYNC_CHECK_FIELDS == {
            "market_report",
            "sentiment_report",
            "news_report",
            "value_trap_report",
            "auditor_report",
        }

    def test_publishability_policy_lists_required_and_optional_artifacts(self):
        assert REQUIRED_PUBLISHABLE_ARTIFACTS == {
            "market_report",
            "sentiment_report",
            "news_report",
            "value_trap_report",
            "fundamentals_report",
            "final_trade_decision",
        }
        assert QUICK_REQUIRED_PUBLISHABLE_ARTIFACTS == {
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            "final_trade_decision",
        }
        assert OPTIONAL_PUBLISHABLE_ARTIFACTS == {
            "auditor_report",
            "consultant_review",
            "valuation_params",
        }
        assert QUICK_OPTIONAL_PUBLISHABLE_ARTIFACTS == {
            "auditor_report",
            "consultant_review",
            "valuation_params",
            "value_trap_report",
        }

    def test_required_publishable_artifacts_follow_mode(self):
        assert get_required_publishable_artifacts(
            {"run_summary": {"quick_mode": True}}
        ) == {
            "market_report",
            "sentiment_report",
            "news_report",
            "fundamentals_report",
            "final_trade_decision",
        }
        assert get_required_publishable_artifacts(
            {"run_summary": {"quick_mode": False}}
        ) == {
            "market_report",
            "sentiment_report",
            "news_report",
            "value_trap_report",
            "fundamentals_report",
            "final_trade_decision",
        }
        assert get_optional_publishable_artifacts(
            {"run_summary": {"quick_mode": True}}
        ) == {
            "auditor_report",
            "consultant_review",
            "valuation_params",
            "value_trap_report",
        }
