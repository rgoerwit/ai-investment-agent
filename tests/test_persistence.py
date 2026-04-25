"""Focused tests for extracted persistence helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

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
            "auditor_report": {"complete": True, "ok": True},
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
    assert summary["required_failures"] == ["fundamentals_report"]
    assert summary["optional_failures"] == ["consultant_review"]
    assert summary["llm_attempts"] == 5
    assert summary["llm_failures"] == 2


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
