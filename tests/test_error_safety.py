from __future__ import annotations

from pathlib import Path

from src.error_safety import (
    redact_sensitive_text,
    safe_error_payload,
    safe_metadata,
    safe_trace_input,
)


def test_redact_sensitive_text_redacts_secret_patterns():
    redacted = redact_sensitive_text(
        "https://example.com?q=ok&api_key=supersecretvalue1234567890",
        max_chars=200,
    )
    bearer_redacted = redact_sensitive_text(
        "Authorization=Bearer sk-abcdefghijklmnopqrstuvwxyz123456",
        max_chars=200,
    )

    assert "supersecretvalue1234567890" not in redacted
    assert "api_key=[REDACTED]" in redacted
    assert "abcdefghijklmnopqrstuvwxyz123456" not in bearer_redacted
    assert "[REDACTED]" in bearer_redacted


def test_safe_trace_input_allowlists_and_masks_paths():
    payload = safe_trace_input(
        {
            "ticker": "6005.T",
            "workflow": "analysis",
            "results_dir": Path("/tmp/results"),
            "api_token": "secret-token",
        },
        allowlist={"ticker", "workflow", "results_dir", "api_token"},
    )

    assert payload == {
        "ticker": "6005.T",
        "workflow": "analysis",
        "results_dir": "[path]",
    }


def test_safe_metadata_filters_sensitive_keys_and_truncates_values():
    payload = safe_metadata(
        {
            "ticker": "6005.T",
            "password": "supersecret",
            "notes": "x" * 100,
        },
        max_chars=16,
    )

    assert "password" not in payload
    assert payload["ticker"] == "6005.T"
    assert payload["notes"].endswith("...")
    assert len(payload["notes"]) <= 16


def test_safe_error_payload_sanitizes_exception_text():
    exc = RuntimeError(
        "request failed for api_token=secret1234567890 on /tmp/private/results.json"
    )

    payload = safe_error_payload(exc, operation="fetch news")

    assert payload["error_type"] == "RuntimeError"
    assert payload["failure_kind"] == "unknown_provider_error"
    assert payload["retryable"] is False
    assert payload["error"].startswith("Error in fetch news: RuntimeError")
    assert "secret1234567890" not in payload["error"]
    assert "results.json" not in payload["error"]
    assert "secret1234567890" not in payload.get("message_preview", "")
