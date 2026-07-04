"""Focused tests for extracted persistence helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import ToolMessage


def test_build_run_summary_tracks_finished_successful_artifacts(monkeypatch):
    from src.persistence import build_run_summary

    class StubTracker:
        def get_total_stats(self):
            return {"failed_attempts": 2, "total_calls": 3}

    monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())

    result = {
        "pre_screening_result": "PASS",
        "investment_debate_state": {"count": 1},
        "analysis_validity": {
            "publishable": True,
            "required_failures": {"fundamentals_report": {}},
            "optional_failures": {"consultant_review": {}},
        },
        "artifact_statuses": {
            "consultant_review": {"complete": True, "ok": False},
            # ok=True AND a parseable non-caveated STATUS → a genuine successful audit.
            "auditor_report": {
                "complete": True,
                "ok": True,
                "content": "STATUS: CLEAN\nNo anomalies detected.",
            },
            "apac_regional_report": {"complete": True, "ok": True},
        },
        "messages": [ToolMessage(content="done", tool_call_id="call_1", name="tool")],
    }

    summary = build_run_summary(
        result,
        quick_mode=True,
        article_requested=False,
        provider_preflight={"google": {"dns": "ok"}},
    )

    assert summary["consultant_finished"] is True
    assert summary["consultant_successful"] is False
    assert summary["auditor_finished"] is True
    assert summary["auditor_successful"] is True
    assert summary["apac_specialist_completed"] is True
    assert summary["apac_specialist_successful"] is True
    assert summary["apac_specialist_status"] == "ok"
    assert summary["required_failures"] == ["fundamentals_report"]
    assert summary["optional_failures"] == ["consultant_review"]
    assert summary["llm_attempts"] == 5
    assert summary["llm_failures"] == 2


@pytest.mark.parametrize(
    "report, expected_successful, expected_status",
    [
        (
            "## FORENSIC AUDITOR REPORT\n\nSTATUS: INSUFFICIENT_DATA\n\nReason: ...",
            False,
            "INSUFFICIENT_DATA",
        ),
        (
            "## FORENSIC AUDITOR REPORT\n\nSTATUS: UNAVAILABLE\n",
            False,
            "UNAVAILABLE",
        ),
        (
            # Filings retrieved but audit opinion unverified → caveated, not a clean pass.
            "## FORENSIC AUDITOR REPORT\n\nSTATUS: PARTIAL_DATA\n\nFY2025 20-F located.",
            False,
            "PARTIAL_DATA",
        ),
        (
            "## FORENSIC AUDITOR REPORT\n\nSTATUS: CLEAN\n\nNo anomalies detected.",
            True,
            "CLEAN",
        ),
        (
            # No parseable STATUS line → unknown, must NOT read as a clean pass.
            "## FORENSIC AUDITOR REPORT\n\nSome prose without a status field.",
            False,
            None,
        ),
    ],
)
def test_build_run_summary_auditor_success_requires_data(
    monkeypatch, report, expected_successful, expected_status
):
    """A well-formed but data-less auditor report must not count as successful."""
    from src.persistence import build_run_summary

    class StubTracker:
        def get_total_stats(self):
            return {"failed_attempts": 0, "total_calls": 1}

    monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())

    result = {
        "pre_screening_result": "PASS",
        "investment_debate_state": {"count": 1},
        "artifact_statuses": {
            # ok=True because the prose is structurally valid; content drives success.
            "auditor_report": {"complete": True, "ok": True, "content": report},
        },
        "messages": [],
    }

    summary = build_run_summary(
        result, quick_mode=False, article_requested=False, provider_preflight={}
    )

    assert summary["auditor_finished"] is True
    assert summary["auditor_successful"] is expected_successful
    assert summary["auditor_status"] == expected_status


def test_save_results_to_file_preserves_macro_context_metadata(tmp_path, monkeypatch):
    from src.persistence import save_results_to_file

    monkeypatch.setattr("src.persistence.config.results_dir", str(tmp_path))
    monkeypatch.setattr("src.persistence.config.enable_memory", False)
    monkeypatch.setattr("src.prompts.get_all_prompts", lambda: {})

    class StubTracker:
        def get_total_stats(self):
            return {
                "failed_attempts": 0,
                "total_calls": 1,
                "total_agents": 1,
                "total_prompt_tokens": 10,
                "total_completion_tokens": 5,
                "total_tokens": 15,
                "total_cost_usd": 0.1,
                "session_start": "2026-03-14T00:00:00",
                "agents": {"Macro Context Analyst": {"calls": 1}},
                "failed_by_provider": {},
                "failed_by_kind": {},
            }

    monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())

    result = {
        "market_report": "ok",
        "sentiment_report": "ok",
        "news_report": "ok",
        "fundamentals_report": "DATA_BLOCK",
        "apac_regional_report": "### APAC REGIONAL AUDIT: 7203.T",
        "final_trade_decision": "BUY",
        "analysis_validity": {"publishable": True},
        "artifact_statuses": {},
        "prompts_used": {
            "macro_context_analyst": {
                "agent_name": "Macro Context Analyst",
                "version": "1.0",
            }
        },
        "run_summary": {
            "quick_mode": True,
            "tool_calls": 0,
            "publishable": True,
            "macro_context_status": "generated",
            "macro_context_region": "JAPAN",
            "macro_context_report_present": True,
            "macro_context_injected_into_news": True,
        },
        "macro_context_llm_invoked": True,
        "macro_context_generated_at": "2026-04-18T00:00:00+00:00",
        "macro_context_injected_into_news": True,
        "macro_context_report": "brief",
        "macro_regime_block": {
            "risk_appetite": "RISK_OFF",
            "shock_type": "ENERGY",
            "shock_phase": "ACUTE",
            "equity_transmission": "EARNINGS_PRESSURE",
            "dip_posture": "WAIT_FOR_CONFIRMATION",
            "confidence": "MEDIUM",
            "present": True,
        },
        "macro_regime_raw": "MACRO_REGIME_BLOCK:\nRISK_APPETITE: RISK_OFF",
    }

    output_path = save_results_to_file(result, "7203.T", quick_mode=True)
    payload = json.loads(output_path.read_text())

    assert payload["macro_context"]["status"] == "generated"
    assert payload["macro_context"]["region"] == "JAPAN"
    assert payload["macro_context"]["report_present"] is True
    assert payload["macro_context"]["injected_into_news"] is True
    assert payload["macro_context"]["llm_invoked"] is True
    assert payload["macro_context"]["cache_dir"] == str(
        tmp_path / ".macro_context_cache"
    )
    assert payload["macro_regime_block"]["risk_appetite"] == "RISK_OFF"
    assert payload["macro_regime_raw"].startswith("MACRO_REGIME_BLOCK:")
    assert payload["reports"]["apac_regional_report"].startswith(
        "### APAC REGIONAL AUDIT"
    )


def test_save_results_to_file_canonicalizes_prediction_snapshot_sector(
    tmp_path, monkeypatch
):
    from src.persistence import save_results_to_file

    monkeypatch.setattr("src.persistence.config.results_dir", str(tmp_path))
    monkeypatch.setattr("src.persistence.config.enable_memory", False)
    monkeypatch.setattr("src.prompts.get_all_prompts", lambda: {})

    class StubTracker:
        def get_total_stats(self):
            return {
                "failed_attempts": 0,
                "total_calls": 0,
                "total_agents": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "session_start": "2026-03-14T00:00:00",
                "agents": {},
                "failed_by_provider": {},
                "failed_by_kind": {},
            }

    monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())

    result = {
        "market_report": "ok",
        "sentiment_report": "ok",
        "news_report": "ok",
        "fundamentals_report": "DATA_BLOCK",
        "final_trade_decision": "BUY",
        "analysis_validity": {"publishable": True},
        "artifact_statuses": {},
        "prompts_used": {},
        "run_summary": {"quick_mode": True, "tool_calls": 0, "publishable": True},
    }

    with patch(
        "src.retrospective.extract_snapshot",
        return_value={
            "ticker": "7203.T",
            "verdict": "BUY",
            "analysis_date": "2026-04-25",
            "sector": "Consumer Cyclical",
        },
    ):
        output_path = save_results_to_file(result, "7203.T", quick_mode=True)

    payload = json.loads(output_path.read_text())
    assert payload["prediction_snapshot"]["sector"] == "Consumer Discretionary"


@pytest.mark.asyncio
async def test_maybe_save_rejection_record_canonicalizes_snapshot_sector(
    monkeypatch,
):
    """Note: tests/conftest.py sets ENABLE_MEMORY=false session-wide; this
    test asserts the memory-enabled code path so it must opt back in.

    We mutate `config.__dict__["enable_memory"]` via `monkeypatch.setitem`
    rather than `monkeypatch.setattr(config, ...)`. The latter goes through
    pydantic-settings BaseModel `__setattr__`, which under full-suite test
    ordering has been observed not to take effect reliably (see the May
    2026 cross-test leakage incident — three otherwise-isolated tests
    failed only when run alongside the broader suite). `setitem` bypasses
    pydantic's setter entirely and is properly tracked/restored by
    monkeypatch.
    """
    from src.config import config
    from src.persistence import _maybe_save_rejection_record

    monkeypatch.setitem(config.__dict__, "enable_memory", True)

    args = SimpleNamespace(ticker="7203.T", quick=False)
    logger = MagicMock()

    with (
        patch(
            "src.retrospective.extract_snapshot",
            return_value={
                "ticker": "7203.T",
                "verdict": "HOLD",
                "analysis_date": "2026-04-25",
                "sector": "Technology",
            },
        ),
        patch(
            "src.retrospective.create_lessons_memory",
            return_value=MagicMock(available=True),
        ),
        patch(
            "src.retrospective.save_rejection_record",
            new=AsyncMock(return_value=True),
        ) as save_rejection_record,
    ):
        await _maybe_save_rejection_record(
            {}, args, trace_id="trace-1", logger_obj=logger
        )

    snapshot = save_rejection_record.await_args.args[0]
    assert snapshot["sector"] == "Information Technology"


@pytest.mark.asyncio
async def test_maybe_save_rejection_record_skips_when_memory_disabled(monkeypatch):
    """`--no-memory` (config.enable_memory=False) must skip the rejection
    record save entirely. Otherwise the path tries to spin up the global
    `lessons_learned` ChromaDB collection and emits the
    `add_situations_failed` / `rejection_record_storage_failed` noise that
    the May 2026 HK-pipeline run produced."""
    from src.config import config
    from src.persistence import _maybe_save_rejection_record

    # Conftest already sets enable_memory=False session-wide; pin it
    # explicitly so the test is robust to future conftest changes. Use
    # setitem to bypass pydantic-settings' BaseModel __setattr__ for
    # consistency with peer tests (see canonicalizes_snapshot_sector
    # docstring for full background).
    monkeypatch.setitem(config.__dict__, "enable_memory", False)

    args = SimpleNamespace(ticker="2099.HK", quick=True)
    logger = MagicMock()

    with (
        patch("src.retrospective.extract_snapshot") as extract_snapshot,
        patch("src.retrospective.create_lessons_memory") as create_lessons_memory,
        patch("src.retrospective.save_rejection_record") as save_rejection_record,
    ):
        await _maybe_save_rejection_record(
            {}, args, trace_id="trace-1", logger_obj=logger
        )

    extract_snapshot.assert_not_called()
    create_lessons_memory.assert_not_called()
    save_rejection_record.assert_not_called()
    # Skipped at debug level — no warnings, no errors.
    logger.debug.assert_called_once()
    assert logger.debug.call_args.args[0] == "rejection_record_save_skipped_no_memory"
    logger.warning.assert_not_called()
    logger.error.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_save_rejection_record_runs_when_memory_enabled(monkeypatch):
    """Regression guard: when memory is enabled, the rejection-save path
    still fires for non-BUY verdicts (don't accidentally short-circuit it
    in the no-memory check)."""
    from src.config import config
    from src.persistence import _maybe_save_rejection_record

    # See canonicalizes_snapshot_sector docstring re setitem vs setattr.
    monkeypatch.setitem(config.__dict__, "enable_memory", True)

    args = SimpleNamespace(ticker="7203.T", quick=False)
    logger = MagicMock()

    with (
        patch(
            "src.retrospective.extract_snapshot",
            return_value={
                "ticker": "7203.T",
                "verdict": "HOLD",
                "analysis_date": "2026-04-25",
                "sector": "Technology",
            },
        ),
        patch(
            "src.retrospective.create_lessons_memory",
            return_value=MagicMock(available=True),
        ),
        patch(
            "src.retrospective.save_rejection_record",
            new=AsyncMock(return_value=True),
        ) as save_rejection_record,
    ):
        await _maybe_save_rejection_record(
            {}, args, trace_id="trace-1", logger_obj=logger
        )

    save_rejection_record.assert_awaited_once()


def test_persist_analysis_outputs_surfaces_formatted_warning():
    from src.persistence import _persist_analysis_outputs

    console = MagicMock()
    logger = MagicMock()
    args = SimpleNamespace(ticker="7203.T", quick=True, quiet=False, brief=False)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("save failed")

    monkeypatch = None
    # Local import keeps the patch scope tight.
    from unittest.mock import patch

    with patch("src.persistence.save_results_to_file", side_effect=_raise):
        _persist_analysis_outputs(
            {"analysis_validity": {"publishable": True}},
            args,
            trace_id="trace-1",
            logger_obj=logger,
            console_obj=console,
            cost_suffix_fn=lambda: "",
            error_message_formatter=lambda op, exc: f"{op}:{type(exc).__name__}",
        )

    console.print.assert_called_once()
    assert "saving analysis results:RuntimeError" in console.print.call_args.args[0]


def test_persist_analysis_outputs_records_saved_path(tmp_path):
    from src.persistence import _persist_analysis_outputs

    args = SimpleNamespace(ticker="7203.T", quick=True, quiet=True, brief=False)
    result: dict = {}
    saved = tmp_path / "7203.T_20260704_analysis.json"

    with patch("src.persistence.save_results_to_file", return_value=saved):
        _persist_analysis_outputs(
            result,
            args,
            logger_obj=MagicMock(),
            console_obj=None,
        )

    assert result["_saved_analysis_path"] == str(saved)


def test_patch_saved_run_summary_merges_fields(tmp_path):
    from src.persistence import patch_saved_run_summary

    path = tmp_path / "a_analysis.json"
    path.write_text(
        json.dumps({"run_summary": {"existing": 1}, "metadata": {"ticker": "7203.T"}}),
        encoding="utf-8",
    )

    patch_saved_run_summary(
        path,
        {"article_writer_model": "gemini-3.5-flash", "article_writer_fell_back": True},
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_summary"]["existing"] == 1
    assert data["run_summary"]["article_writer_model"] == "gemini-3.5-flash"
    assert data["run_summary"]["article_writer_fell_back"] is True
    assert data["metadata"]["ticker"] == "7203.T"


def test_patch_saved_run_summary_creates_missing_run_summary(tmp_path):
    from src.persistence import patch_saved_run_summary

    path = tmp_path / "a_analysis.json"
    path.write_text(json.dumps({"metadata": {}}), encoding="utf-8")

    patch_saved_run_summary(path, {"article_writer_fell_back": False})

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_summary"] == {"article_writer_fell_back": False}


def test_patch_saved_run_summary_missing_file_fails_open(tmp_path):
    from src.persistence import patch_saved_run_summary

    logger = MagicMock()

    patch_saved_run_summary(tmp_path / "missing.json", {"a": 1}, logger_obj=logger)

    logger.warning.assert_called_once()
    assert logger.warning.call_args.args[0] == "run_summary_patch_failed"
