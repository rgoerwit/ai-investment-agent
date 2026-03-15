"""
Tests for src.main CLI argument parsing.

Covers --strict flag parsing and composability with other flags.
"""

import json
import logging
import sys
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def restore_cli_logger_levels():
    from src.main import (
        CLI_APP_DEBUG_LOGGERS,
        CLI_NOISY_DEPENDENCY_LOGGERS,
        HTTP_TRACE_LOGGERS,
    )

    logger_names = {
        *CLI_APP_DEBUG_LOGGERS,
        *CLI_NOISY_DEPENDENCY_LOGGERS.keys(),
        *HTTP_TRACE_LOGGERS,
    }
    saved_levels = {name: logging.getLogger(name).level for name in logger_names}
    saved_root_level = logging.getLogger().level

    yield

    logging.getLogger().setLevel(saved_root_level)
    for name, level in saved_levels.items():
        logging.getLogger(name).setLevel(level)


class TestStrictModeCLI:
    """Test --strict CLI flag is wired correctly."""

    def test_strict_flag_parsed_from_cli(self):
        """--strict sets args.strict = True."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--strict"])
        assert args.strict is True

    def test_no_strict_flag_defaults_false(self):
        """Without --strict, args.strict = False."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK"])
        assert args.strict is False

    def test_strict_and_quick_composable(self):
        """--strict --quick can be combined without conflict."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--strict", "--quick"])
        assert args.strict is True
        assert args.quick is True

    def test_strict_with_quiet_composable(self):
        """--strict --quiet can be combined (batch use case)."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--strict", "--quiet"])
        assert args.strict is True
        assert args.quiet is True

    def test_strict_with_output_composable(self):
        """--strict with --output is valid."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            ["--ticker", "0005.HK", "--strict", "--output", "results/test.md"]
        )
        assert args.strict is True
        assert args.output == "results/test.md"

    def test_strict_quick_quiet_all_composable(self):
        """--strict --quick --quiet can all be combined (pipeline batch mode)."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            ["--ticker", "0005.HK", "--strict", "--quick", "--quiet"]
        )
        assert args.strict is True
        assert args.quick is True
        assert args.quiet is True


class TestStrictAddendaContent:
    """Sanity-check the content of _STRICT_PM_ADDENDUM and _STRICT_RM_ADDENDUM.

    No mocking needed — these are pure string checks on module-level constants.
    If content changes break these, it's a signal to update the plan doc too.
    """

    def test_pm_addendum_has_tighter_health_threshold(self):
        """PM addendum must require Financial Health ≥ 60% (tighter than normal 50%)."""
        from src.agents import _STRICT_PM_ADDENDUM

        assert "Financial Health ≥ 60%" in _STRICT_PM_ADDENDUM

    def test_pm_addendum_rejects_pfic_and_vie(self):
        """PM addendum must explicitly disqualify both PFIC and VIE."""
        from src.agents import _STRICT_PM_ADDENDUM

        assert "PFIC" in _STRICT_PM_ADDENDUM
        assert "VIE" in _STRICT_PM_ADDENDUM

    def test_rm_addendum_has_catalyst_requirement(self):
        """RM addendum must require a near-term catalyst in strict mode."""
        from src.agents import _STRICT_RM_ADDENDUM

        assert "catalyst" in _STRICT_RM_ADDENDUM.lower()

    def test_rm_addendum_weights_bear_arguments(self):
        """RM addendum must instruct to weight bear arguments more heavily."""
        from src.agents import _STRICT_RM_ADDENDUM

        assert "bear" in _STRICT_RM_ADDENDUM.lower()


class TestToolAuditLogging:
    def test_debug_flag_implies_verbose(self, monkeypatch):
        from src.main import build_arg_parser, parse_arguments

        parser = build_arg_parser()
        parsed = parser.parse_args(["--ticker", "6083.T", "--debug"])
        assert parsed.debug is True

        monkeypatch.setattr(sys, "argv", ["prog", "--ticker", "6083.T", "--debug"])
        args = parse_arguments()

        assert args.debug is True
        assert args.verbose is True

    def test_configure_tool_audit_logging_is_opt_in(self):
        from src.main import configure_tool_audit_logging
        from src.tooling.runtime import TOOL_SERVICE

        configure_tool_audit_logging(False)
        assert TOOL_SERVICE.hooks == []

        configure_tool_audit_logging(True)
        try:
            assert len(TOOL_SERVICE.hooks) == 1
        finally:
            configure_tool_audit_logging(False)

    @patch("src.main.socket.getaddrinfo", side_effect=OSError("dns down"))
    def test_provider_preflight_logs_failures(self, _mock_dns):
        from src.main import run_provider_preflight

        result = run_provider_preflight()
        assert result["openai"]["dns"] == "failed"

    def test_configure_cli_logging_keeps_transport_logs_suppressed_in_verbose(
        self, monkeypatch
    ):
        from src.main import configure_cli_logging

        args = type(
            "Args",
            (),
            {"quiet": False, "brief": False, "verbose": True, "debug": False},
        )()

        monkeypatch.setattr(
            "src.main.configure_tool_audit_logging", lambda enabled: None
        )
        monkeypatch.setattr("src.main.run_provider_preflight", lambda: {"ok": True})

        result = configure_cli_logging(args)

        assert result == {"ok": True}
        assert logging.getLogger("src").level == logging.DEBUG
        assert logging.getLogger("openai").level >= logging.WARNING
        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("httpcore").level >= logging.WARNING

    def test_configure_cli_logging_allows_http_trace_only_in_debug(self, monkeypatch):
        from src.main import configure_cli_logging

        args = type(
            "Args",
            (),
            {"quiet": False, "brief": False, "verbose": True, "debug": True},
        )()

        monkeypatch.setenv("INVESTMENT_AGENT_TRACE_HTTP", "1")
        monkeypatch.setattr(
            "src.main.configure_tool_audit_logging", lambda enabled: None
        )
        monkeypatch.setattr("src.main.run_provider_preflight", lambda: {"ok": True})

        configure_cli_logging(args)

        assert logging.getLogger("openai").level == logging.DEBUG
        assert logging.getLogger("httpx").level == logging.DEBUG
        assert logging.getLogger("httpcore").level == logging.DEBUG


class TestSavedDiagnostics:
    def test_build_run_summary_tracks_finished_vs_successful_artifacts(
        self, monkeypatch
    ):
        from src.main import build_run_summary

        class StubTracker:
            def get_total_stats(self):
                return {
                    "failed_attempts": 2,
                    "total_calls": 3,
                }

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
        }

        summary = build_run_summary(
            result,
            quick_mode=True,
            article_requested=False,
            provider_preflight={"google": {"dns": "ok"}},
        )

        assert summary["consultant_completed"] is True
        assert summary["consultant_finished"] is True
        assert summary["consultant_successful"] is False
        assert summary["auditor_completed"] is True
        assert summary["auditor_finished"] is True
        assert summary["auditor_successful"] is True
        assert summary["required_failures"] == ["fundamentals_report"]
        assert summary["optional_failures"] == ["consultant_review"]
        assert summary["llm_attempts"] == 5
        assert summary["llm_failures"] == 2

    def test_save_results_includes_pre_screening_and_run_summary(
        self, tmp_path, monkeypatch
    ):
        from langchain_core.messages import ToolMessage

        from src.main import save_results_to_file

        monkeypatch.setattr("src.main.config.results_dir", str(tmp_path))
        monkeypatch.setattr("src.main.config.enable_memory", False)
        monkeypatch.setattr("src.prompts.get_all_prompts", lambda: {})

        class StubTracker:
            def get_total_stats(self):
                return {
                    "failed_attempts": 1,
                    "total_calls": 2,
                    "total_agents": 1,
                    "total_prompt_tokens": 10,
                    "total_completion_tokens": 5,
                    "total_tokens": 15,
                    "total_cost_usd": 0.1,
                    "session_start": "2026-03-14T00:00:00",
                    "agents": {},
                    "failed_by_provider": {"google": 1},
                    "failed_by_kind": {"timeout": 1},
                }

        monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())

        result = {
            "market_report": "ok",
            "sentiment_report": "ok",
            "news_report": "ok",
            "fundamentals_report": "DATA_BLOCK",
            "final_trade_decision": "BUY",
            "pre_screening_result": "PASS",
            "investment_debate_state": {"count": 1},
            "analysis_validity": {"publishable": True},
            "artifact_statuses": {},
            "prompts_used": {},
            "run_summary": {
                "quick_mode": True,
                "tool_calls": 1,
                "publishable": True,
            },
            "messages": [
                ToolMessage(content="done", tool_call_id="call_1", name="get_news")
            ],
        }

        output_path = save_results_to_file(result, "6083.T", quick_mode=True)
        payload = json.loads(output_path.read_text())

        assert payload["pre_screening_result"] == "PASS"
        assert payload["run_summary"]["quick_mode"] is True
        assert payload["run_summary"]["tool_calls"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
