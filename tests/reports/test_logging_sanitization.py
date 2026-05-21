"""Verify the new memo + quality-judge logging routes through summarize_exception
(Tranche 5, Step 9).

These tests intercept the structlog logger directly (caplog can't see
structlog warnings without bridging configuration) and assert that the
warning kwargs come from ``summarize_exception``'s structured summary —
``operation`` / ``error_type`` / ``failure_kind`` / ``message_preview`` —
rather than a bare ``error=str(exc)`` field that could leak raw secrets.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def _capture_warnings(target_module: str) -> tuple[object, list[dict]]:
    """Patch ``target_module.logger`` and return a (patcher, captured-list) pair."""
    captured: list[dict] = []

    class _RecorderLogger:
        def warning(self, event: str, **kwargs) -> None:
            captured.append({"event": event, **kwargs})

        # The rest of the structlog logger surface — no-op for these tests.
        def info(self, *a, **k) -> None: ...
        def debug(self, *a, **k) -> None: ...
        def error(self, *a, **k) -> None: ...

    patcher = patch(f"{target_module}.logger", _RecorderLogger())
    return patcher, captured


def test_memo_render_failure_uses_summarize_exception_keys() -> None:
    from src.reporting import memo as memo_mod

    patcher, captured = _capture_warnings("src.reporting.memo")
    secret = "sk-FAKE-API-KEY-SHOULD-NOT-LEAK"
    with (
        patcher,
        patch("src.reporting.memo.build_memo", side_effect=RuntimeError(secret)),
    ):
        out = memo_mod.render_memo_for_state({})

    assert "UNAVAILABLE" in out
    events = [c for c in captured if c["event"] == "memo_render_failed"]
    assert events, f"memo_render_failed event missing — captured: {captured}"
    record = events[0]
    # Sanitized keys produced by summarize_exception.
    assert record.get("operation") == "render_memo_for_state"
    assert "error_type" in record
    assert "failure_kind" in record
    # Critically: no raw `error=<exc string>` field.
    assert "error" not in record or record.get("error") != secret


def test_quality_judge_corrupt_json_uses_summarize_exception(tmp_path: Path) -> None:
    from src.eval import report_quality_judge as judge_mod

    bad = tmp_path / "broken_analysis.json"
    bad.write_text("{not json", encoding="utf-8")

    patcher, captured = _capture_warnings("src.eval.report_quality_judge")
    secret = "/private/path/should/not/leak/credentials.json"
    with patcher, patch("json.loads", side_effect=ValueError(secret)):
        result = judge_mod.score_saved_analysis(bad)

    assert result is None
    events = [c for c in captured if c["event"] == "quality_judge_json_read_failed"]
    assert events, f"event missing — captured: {captured}"
    record = events[0]
    assert record.get("operation") == "quality_judge_load_json"
    assert "error_type" in record
    # No raw `error=<exc string>` field.
    assert "error" not in record or record.get("error") != secret
