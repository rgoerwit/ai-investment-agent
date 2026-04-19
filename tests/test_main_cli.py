"""
Tests for src.main CLI argument parsing.

Covers --strict flag parsing and composability with other flags.
"""

import asyncio
import json
import logging
import sys
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest


async def _async_none():
    return None


async def _async_result(value):
    return value


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


class TestOutputCompanyNameLookup:
    def test_load_company_name_for_output_retries_normalized_alias(self):
        from src.main import _load_company_name_for_output

        requested_symbols = []

        class _Ticker:
            def __init__(self, symbol):
                self.info = (
                    {"longName": "Truecaller AB"} if symbol == "TRUE-B.ST" else {}
                )

        class _Future:
            def __init__(self, fn):
                self._fn = fn

            def result(self, timeout=None):
                return self._fn()

        class _Executor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn):
                return _Future(fn)

        fake_yfinance = MagicMock()

        def _ticker_factory(symbol):
            requested_symbols.append(symbol)
            return _Ticker(symbol)

        fake_yfinance.Ticker.side_effect = _ticker_factory

        with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
            with patch("src.main.ThreadPoolExecutor", return_value=_Executor()):
                assert _load_company_name_for_output("TRUE.B.ST") == "Truecaller AB"

        assert requested_symbols == ["TRUE.B.ST", "TRUE-B.ST"]

    def test_load_company_name_for_output_returns_none_on_timeout(self):
        from src.main import _load_company_name_for_output

        fake_yfinance = MagicMock()
        fake_yfinance.Ticker.return_value.info = {"longName": "Should Not Return"}

        mock_future = MagicMock()
        mock_future.result.side_effect = FuturesTimeoutError()

        mock_executor = MagicMock()
        mock_executor.__enter__.return_value.submit.return_value = mock_future

        with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
            with patch("src.main.ThreadPoolExecutor", return_value=mock_executor):
                assert _load_company_name_for_output("SNTIA.OL") is None

    def test_run_analysis_warns_with_lookup_candidates_after_company_name_exhaustion(
        self,
    ):
        from src.main import run_analysis
        from src.ticker_utils import CompanyNameResult

        fake_tracker = MagicMock()
        fake_graph = MagicMock()
        fake_graph.ainvoke = AsyncMock(return_value={})

        with (
            patch("src.main.logger") as mock_logger,
            patch(
                "src.ticker_utils.resolve_company_name",
                new=AsyncMock(
                    return_value=CompanyNameResult(
                        name="TRUE.B.ST",
                        source="unresolved",
                        is_resolved=False,
                    )
                ),
            ),
            patch("src.main._fetch_market_context", new=AsyncMock(return_value="")),
            patch("src.graph.create_trading_graph", return_value=fake_graph),
            patch("src.token_tracker.get_tracker", return_value=fake_tracker),
            patch("src.main.build_analysis_validity", return_value={"ok": True}),
        ):
            result = asyncio.run(
                run_analysis(
                    ticker="TRUE.B.ST",
                    quick_mode=True,
                    strict_mode=False,
                    skip_charts=True,
                )
            )

        assert result["analysis_validity"] == {"ok": True}
        assert result["macro_context_status"] == "failed"
        assert result["macro_context_region"]
        warning_calls = [
            call
            for call in mock_logger.warning.call_args_list
            if call.args and call.args[0] == "company_name_unresolved_at_startup"
        ]
        assert len(warning_calls) == 1
        assert warning_calls[0].kwargs["requested_ticker"] == "TRUE.B.ST"
        assert warning_calls[0].kwargs["lookup_candidates"] == [
            "TRUE.B.ST",
            "TRUE-B.ST",
            "TRUE.ST",
        ]

    def test_run_analysis_prefetches_macro_context_into_trading_context(self):
        from src.main import run_analysis
        from src.ticker_utils import CompanyNameResult

        fake_tracker = MagicMock()
        captured_context = {}

        async def _capture_ainvoke(_state, *, config):
            captured_context["context"] = config["configurable"]["context"]
            return {}

        fake_graph = MagicMock()
        fake_graph.ainvoke = AsyncMock(side_effect=_capture_ainvoke)

        macro_result = MagicMock()
        macro_result.report = "### EQUITY REGIME\n- Summary: Risk appetite is mixed."
        macro_result.region = "JAPAN"
        macro_result.status = "cached"
        macro_result.generated_at = None
        macro_result.llm_invoked = False
        macro_result.prompt_used = None

        with (
            patch(
                "src.ticker_utils.resolve_company_name",
                new=AsyncMock(
                    return_value=CompanyNameResult(
                        name="Toyota Motor",
                        source="test",
                        is_resolved=True,
                    )
                ),
            ),
            patch("src.main._fetch_market_context", new=AsyncMock(return_value="")),
            patch(
                "src.macro_context.get_macro_context",
                new=AsyncMock(return_value=macro_result),
            ),
            patch("src.graph.create_trading_graph", return_value=fake_graph),
            patch("src.token_tracker.get_tracker", return_value=fake_tracker),
            patch("src.main.build_analysis_validity", return_value={"ok": True}),
        ):
            result = asyncio.run(
                run_analysis(
                    ticker="7203.T",
                    quick_mode=True,
                    strict_mode=False,
                    skip_charts=True,
                )
            )

        assert result["analysis_validity"] == {"ok": True}
        assert result["macro_context_status"] == "cached"
        assert result["macro_context_region"] == "JAPAN"
        assert result["macro_context_injected_into_news"] is False
        context = captured_context["context"]
        assert context.macro_context_report == macro_result.report
        assert context.macro_context_region == "JAPAN"
        assert context.macro_context_status == "cached"

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

    def test_capture_baseline_flag_parsed_from_cli(self):
        """--capture-baseline enables baseline capture mode."""
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--capture-baseline"])
        assert args.capture_baseline is True

    def test_capture_baseline_cleanup_flag_parsed_from_cli(self):
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--capture-baseline-cleanup"])
        assert args.capture_baseline_cleanup is True

    def test_enable_langfuse_flag_parsed_from_cli(self):
        from src.main import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--enable-langfuse"])
        assert args.enable_langfuse is True

    def test_resolve_langfuse_session_id_prefers_env_override(self, monkeypatch):
        from src.main import _resolve_langfuse_session_id

        monkeypatch.setenv("LANGFUSE_SESSION_ID", "batch-session-123")

        assert _resolve_langfuse_session_id("default-session") == "batch-session-123"

    def test_resolve_langfuse_session_id_uses_default_when_env_missing(
        self, monkeypatch
    ):
        from src.main import _resolve_langfuse_session_id

        monkeypatch.delenv("LANGFUSE_SESSION_ID", raising=False)

        assert _resolve_langfuse_session_id("default-session") == "default-session"

    def test_parse_arguments_allows_cleanup_without_ticker(self, monkeypatch):
        from src.main import parse_arguments

        monkeypatch.setattr(sys, "argv", ["prog", "--capture-baseline-cleanup"])
        args = parse_arguments()
        assert args.capture_baseline_cleanup is True
        assert args.ticker is None


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


class TestTracingMetadataFlow:
    def test_run_analysis_uses_passed_tracing_metadata_without_rebuilding(self):
        from src.main import run_analysis
        from src.ticker_utils import CompanyNameResult

        fake_tracker = MagicMock()
        fake_graph = MagicMock()
        fake_graph.ainvoke = AsyncMock(return_value={})
        tracing_metadata = {"ticker": "0005.HK", "session_id": "session-1"}

        with (
            patch(
                "src.ticker_utils.resolve_company_name",
                new=AsyncMock(
                    return_value=CompanyNameResult(
                        name="HSBC Holdings",
                        source="lookup",
                        is_resolved=True,
                    )
                ),
            ),
            patch("src.main._fetch_market_context", new=AsyncMock(return_value="")),
            patch("src.graph.create_trading_graph", return_value=fake_graph),
            patch("src.token_tracker.get_tracker", return_value=fake_tracker),
            patch("src.main.build_analysis_validity", return_value={"ok": True}),
            patch(
                "src.main._build_analysis_trace_metadata",
                side_effect=AssertionError("should not rebuild metadata"),
            ),
        ):
            result = asyncio.run(
                run_analysis(
                    ticker="0005.HK",
                    quick_mode=True,
                    strict_mode=False,
                    skip_charts=True,
                    session_id="session-1",
                    tracing_metadata=tracing_metadata,
                )
            )

        assert result["analysis_validity"] == {"ok": True}
        assert (
            fake_graph.ainvoke.await_args.kwargs["config"]["metadata"]
            == tracing_metadata
        )

    def test_run_analysis_builds_metadata_when_tracing_metadata_missing(self):
        from src.main import run_analysis
        from src.ticker_utils import CompanyNameResult

        fake_tracker = MagicMock()
        fake_graph = MagicMock()
        fake_graph.ainvoke = AsyncMock(return_value={})

        with (
            patch(
                "src.ticker_utils.resolve_company_name",
                new=AsyncMock(
                    return_value=CompanyNameResult(
                        name="HSBC Holdings",
                        source="lookup",
                        is_resolved=True,
                    )
                ),
            ),
            patch("src.main._fetch_market_context", new=AsyncMock(return_value="")),
            patch("src.graph.create_trading_graph", return_value=fake_graph),
            patch("src.token_tracker.get_tracker", return_value=fake_tracker),
            patch("src.main.build_analysis_validity", return_value={"ok": True}),
            patch(
                "src.main._build_analysis_trace_metadata",
                return_value={"ticker": "0005.HK", "session_id": "session-1"},
            ) as mock_builder,
        ):
            result = asyncio.run(
                run_analysis(
                    ticker="0005.HK",
                    quick_mode=True,
                    strict_mode=False,
                    skip_charts=True,
                    session_id="session-1",
                )
            )

        assert result["analysis_validity"] == {"ok": True}
        mock_builder.assert_called_once_with(
            ticker="0005.HK",
            session_id="session-1",
            quick_mode=True,
        )
        assert fake_graph.ainvoke.await_args.kwargs["config"]["metadata"] == {
            "ticker": "0005.HK",
            "session_id": "session-1",
        }

    def test_run_analysis_forwards_tracing_callbacks_to_macro_context(self):
        from src.main import run_analysis
        from src.ticker_utils import CompanyNameResult

        fake_tracker = MagicMock()
        fake_graph = MagicMock()
        fake_graph.ainvoke = AsyncMock(return_value={})
        tracing_callback = MagicMock()
        macro_result = MagicMock(
            report="brief",
            region="GLOBAL",
            status="generated",
            generated_at="2026-04-18T00:00:00+00:00",
            llm_invoked=True,
            prompt_used={"agent_name": "Macro Context Analyst", "version": "1.0"},
        )

        with (
            patch(
                "src.ticker_utils.resolve_company_name",
                new=AsyncMock(
                    return_value=CompanyNameResult(
                        name="HSBC Holdings",
                        source="lookup",
                        is_resolved=True,
                    )
                ),
            ),
            patch("src.main._fetch_market_context", new=AsyncMock(return_value="")),
            patch(
                "src.macro_context.get_macro_context",
                new=AsyncMock(return_value=macro_result),
            ) as get_macro_context,
            patch("src.graph.create_trading_graph", return_value=fake_graph),
            patch("src.token_tracker.get_tracker", return_value=fake_tracker),
            patch("src.main.build_analysis_validity", return_value={"ok": True}),
        ):
            result = asyncio.run(
                run_analysis(
                    ticker="0005.HK",
                    quick_mode=True,
                    strict_mode=False,
                    skip_charts=True,
                    tracing_callbacks=[tracing_callback],
                )
            )

        assert result["analysis_validity"] == {"ok": True}
        assert get_macro_context.await_count == 1
        assert get_macro_context.await_args.args[0] == "0005.HK"
        assert get_macro_context.await_args.kwargs["callbacks"] == [tracing_callback]
        assert result["prompts_used"]["macro_context_analyst"]["agent_name"] == (
            "Macro Context Analyst"
        )

    def test_run_analysis_logs_macro_context_prefetch_complete(self):
        from src.main import run_analysis
        from src.ticker_utils import CompanyNameResult

        fake_tracker = MagicMock()
        fake_graph = MagicMock()
        fake_graph.ainvoke = AsyncMock(return_value={})
        macro_result = MagicMock(
            report="brief",
            region="AUSTRALIA",
            status="generated",
            generated_at="2026-04-19T13:49:10.364729+00:00",
            llm_invoked=True,
            prompt_used={"agent_name": "Macro Context Analyst", "version": "1.0"},
        )

        with (
            patch("src.main.logger") as mock_logger,
            patch(
                "src.ticker_utils.resolve_company_name",
                new=AsyncMock(
                    return_value=CompanyNameResult(
                        name="Ridley",
                        source="lookup",
                        is_resolved=True,
                    )
                ),
            ),
            patch("src.main._fetch_market_context", new=AsyncMock(return_value="")),
            patch(
                "src.macro_context.get_macro_context",
                new=AsyncMock(return_value=macro_result),
            ),
            patch("src.graph.create_trading_graph", return_value=fake_graph),
            patch("src.token_tracker.get_tracker", return_value=fake_tracker),
            patch("src.main.build_analysis_validity", return_value={"ok": True}),
        ):
            asyncio.run(
                run_analysis(
                    ticker="RIC.AX",
                    quick_mode=True,
                    strict_mode=False,
                    skip_charts=True,
                )
            )

        mock_logger.info.assert_any_call(
            "macro_context_prefetch_complete",
            ticker="RIC.AX",
            trade_date=ANY,
            region="AUSTRALIA",
            status="generated",
            llm_invoked=True,
            generated_at="2026-04-19T13:49:10.364729+00:00",
            prompt_recorded=True,
        )


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

    def test_configure_tool_audit_logging_preserves_content_inspection_hook(self):
        from src.main import configure_tool_audit_logging
        from src.tooling.inspection_hook import ContentInspectionHook
        from src.tooling.runtime import TOOL_SERVICE

        TOOL_SERVICE.set_hooks([ContentInspectionHook()])
        try:
            configure_tool_audit_logging(True)
            hook_types = [type(h).__name__ for h in TOOL_SERVICE.hooks]
            assert "LoggingToolAuditHook" in hook_types
            assert "ContentInspectionHook" in hook_types

            configure_tool_audit_logging(False)
            hook_types = [type(h).__name__ for h in TOOL_SERVICE.hooks]
            assert "LoggingToolAuditHook" not in hook_types
            assert "ContentInspectionHook" in hook_types
        finally:
            TOOL_SERVICE.clear_hooks()

    def test_configure_content_inspection_from_config_installs_tool_hook(
        self, monkeypatch
    ):
        from src.main import configure_content_inspection_from_config
        from src.tooling.runtime import TOOL_SERVICE

        monkeypatch.setattr(
            "src.main.config.untrusted_content_inspection_enabled", True
        )
        monkeypatch.setattr("src.main.config.untrusted_content_backend", "null")
        monkeypatch.setattr("src.main.config.untrusted_content_inspection_mode", "warn")
        monkeypatch.setattr(
            "src.main.config.untrusted_content_fail_policy", "fail_open"
        )

        TOOL_SERVICE.clear_hooks()
        try:
            configure_content_inspection_from_config()
            hook_types = [type(h).__name__ for h in TOOL_SERVICE.hooks]
            assert "ContentInspectionHook" in hook_types
        finally:
            TOOL_SERVICE.clear_hooks()

    def test_configure_content_inspection_from_config_removes_tool_hook_when_disabled(
        self, monkeypatch
    ):
        from src.main import configure_content_inspection_from_config
        from src.tooling.inspection_hook import ContentInspectionHook
        from src.tooling.runtime import TOOL_SERVICE

        monkeypatch.setattr(
            "src.main.config.untrusted_content_inspection_enabled", False
        )

        TOOL_SERVICE.set_hooks([ContentInspectionHook()])
        try:
            configure_content_inspection_from_config()
            hook_types = [type(h).__name__ for h in TOOL_SERVICE.hooks]
            assert "ContentInspectionHook" not in hook_types
        finally:
            TOOL_SERVICE.clear_hooks()

    def test_configure_content_inspection_from_config_rejects_unimplemented_backend(
        self, monkeypatch
    ):
        from src.main import configure_content_inspection_from_config
        from src.tooling.runtime import TOOL_SERVICE

        monkeypatch.setattr(
            "src.main.config.untrusted_content_inspection_enabled", True
        )
        monkeypatch.setattr("src.main.config.untrusted_content_backend", "http")
        monkeypatch.setattr("src.main.config.untrusted_content_inspection_mode", "warn")
        monkeypatch.setattr(
            "src.main.config.untrusted_content_fail_policy", "fail_open"
        )

        TOOL_SERVICE.clear_hooks()
        try:
            with pytest.raises(ValueError, match="only 'null' is implemented"):
                configure_content_inspection_from_config()
        finally:
            TOOL_SERVICE.clear_hooks()


class TestBaselineCaptureCliHelpers:
    def test_preflight_blocks_dirty_worktree_before_analysis(self):
        from src.main import _run_baseline_capture_preflight

        args = SimpleNamespace(capture_baseline=True, capture_baseline_cleanup=False)

        class DirtyCapture:
            def __init__(self):
                self.cleaned = False

            def cleanup_stale_inflight_runs(self):
                self.cleaned = True
                return SimpleNamespace(
                    scanned=0,
                    moved_to_rejected=0,
                    removed_empty=0,
                    rejected_paths=(),
                )

            def preflight_git_clean(self):
                return False, ["dirty worktree"]

        capture = DirtyCapture()
        ok, messages = _run_baseline_capture_preflight(args, capture)

        assert ok is False
        assert capture.cleaned is True
        assert messages[-1] == "dirty worktree"

    def test_preflight_cleanup_only_mode_skips_git_clean_gate(self):
        from src.main import _run_baseline_capture_preflight

        args = SimpleNamespace(capture_baseline=False, capture_baseline_cleanup=True)

        class CleanupOnlyCapture:
            def cleanup_stale_inflight_runs(self):
                return SimpleNamespace(
                    scanned=2,
                    moved_to_rejected=1,
                    removed_empty=1,
                    rejected_paths=("a",),
                )

            def preflight_git_clean(self):
                raise AssertionError("should not run git preflight")

        ok, messages = _run_baseline_capture_preflight(args, CleanupOnlyCapture())
        assert ok is True
        assert "Cleaned 1 stale inflight capture(s)" in messages[0]

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


class TestRuntimeOverrides:
    def test_apply_runtime_overrides_updates_config(self, monkeypatch):
        from src.main import _apply_runtime_overrides, config

        monkeypatch.setattr(config, "quick_think_llm", "old-quick")
        monkeypatch.setattr(config, "deep_think_llm", "old-deep")
        monkeypatch.setattr(config, "enable_memory", True)
        monkeypatch.setattr(config, "langfuse_enabled", False)

        args = SimpleNamespace(
            quick_model="new-quick",
            deep_model="new-deep",
            no_memory=True,
            enable_langfuse=False,
            trace_langfuse=True,
        )

        _apply_runtime_overrides(args)

        assert config.quick_think_llm == "new-quick"
        assert config.deep_think_llm == "new-deep"
        assert config.enable_memory is False
        assert config.langfuse_enabled is True

    def test_enable_langfuse_flag_updates_config(self, monkeypatch):
        from src.main import _apply_runtime_overrides, config

        monkeypatch.setattr(config, "langfuse_enabled", False)
        args = SimpleNamespace(
            quick_model=None,
            deep_model=None,
            no_memory=False,
            enable_langfuse=True,
            trace_langfuse=False,
        )

        _apply_runtime_overrides(args)

        assert config.langfuse_enabled is True


class TestValidateCliArgs:
    def test_quick_with_svg_exits_2(self, capsys):
        from src.main import _validate_cli_args

        args = SimpleNamespace(quick=True, transparent=False, svg=True)

        with pytest.raises(SystemExit) as exc_info:
            _validate_cli_args(args)

        assert exc_info.value.code == 2
        assert "--svg" in capsys.readouterr().err

    def test_quick_with_transparent_exits_2(self, capsys):
        from src.main import _validate_cli_args

        args = SimpleNamespace(quick=True, transparent=True, svg=False)

        with pytest.raises(SystemExit) as exc_info:
            _validate_cli_args(args)

        assert exc_info.value.code == 2
        assert "--transparent" in capsys.readouterr().err


class TestResolveOutputTargets:
    def test_stdout_without_imagedir_disables_charts(self):
        from src.main import _resolve_output_targets

        args = SimpleNamespace(output=None, imagedir=None, no_charts=False)

        targets = _resolve_output_targets(args)

        assert targets.output_file is None
        assert targets.image_dir == Path("images")
        assert targets.skip_charts is True

    def test_stdout_with_imagedir_preserves_chart_generation(self):
        from src.main import _resolve_output_targets

        args = SimpleNamespace(output=None, imagedir="assets/charts", no_charts=False)

        targets = _resolve_output_targets(args)

        assert targets.output_file is None
        assert targets.image_dir == Path("assets/charts")
        assert targets.skip_charts is False

    def test_file_output_keeps_charts_enabled_by_default(self):
        from src.main import _resolve_output_targets

        args = SimpleNamespace(
            output="results/report.md", imagedir=None, no_charts=False
        )

        targets = _resolve_output_targets(args)

        assert targets.output_file == Path("results/report.md")
        assert targets.image_dir == Path("results/images")
        assert targets.skip_charts is False


class TestMainOrchestration:
    def test_run_retrospective_only_returns_one_on_failure(self, monkeypatch):
        from src.main import _run_retrospective_only

        async def fake_run_retrospective(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            "src.retrospective.run_retrospective",
            fake_run_retrospective,
        )

        args = SimpleNamespace(quiet=True, brief=False)

        assert asyncio.run(_run_retrospective_only(args)) == 1

    def test_retrospective_only_returns_early(self, monkeypatch):
        from src.main import OutputTargets, main

        calls = []
        args = SimpleNamespace(retrospective_only=True)

        async def fake_retrospective_only(passed_args):
            calls.append(("retrospective_only", passed_args))
            return 0

        monkeypatch.setattr("src.main.parse_arguments", lambda: args)
        monkeypatch.setattr(
            "src.main._apply_runtime_overrides", lambda passed_args: None
        )
        monkeypatch.setattr("src.main._validate_cli_args", lambda passed_args: None)
        monkeypatch.setattr(
            "src.main._resolve_output_targets",
            lambda passed_args: OutputTargets(None, Path("images"), True),
        )
        monkeypatch.setattr("src.main._setup_runtime", lambda passed_args, targets: {})
        monkeypatch.setattr("src.main._run_retrospective_only", fake_retrospective_only)
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        assert asyncio.run(main()) == 0
        assert calls == [("retrospective_only", args)]

    def test_main_success_path_returns_zero(self, monkeypatch):
        from src.main import OutputTargets, main

        call_order = []
        args = SimpleNamespace(
            retrospective_only=False,
            ticker="6083.T",
            quick=True,
            strict=False,
            article=False,
            quiet=False,
            brief=False,
            svg=False,
            transparent=False,
            imagedir=None,
        )

        async def fake_async(label, value=None):
            call_order.append(label)
            return value

        monkeypatch.setattr("src.main.parse_arguments", lambda: args)
        monkeypatch.setattr(
            "src.main._apply_runtime_overrides",
            lambda passed_args: call_order.append("apply"),
        )
        monkeypatch.setattr(
            "src.main._validate_cli_args",
            lambda passed_args: call_order.append("validate"),
        )
        monkeypatch.setattr(
            "src.main._resolve_output_targets",
            lambda passed_args: OutputTargets(
                Path("results/report.md"), Path("results/images"), False
            ),
        )
        monkeypatch.setattr(
            "src.main._setup_runtime",
            lambda passed_args, targets: (
                call_order.append("setup") or {"google": {"dns": "ok"}}
            ),
        )
        monkeypatch.setattr(
            "src.main._maybe_run_ticker_retrospective",
            lambda passed_args: fake_async("retrospective"),
        )
        monkeypatch.setattr(
            "src.main._emit_start_banner",
            lambda passed_args, targets: call_order.append("banner") or "banner",
        )
        monkeypatch.setattr(
            "src.main._execute_analysis",
            lambda passed_args, targets, **kwargs: fake_async(
                "execute", {"analysis_validity": {"publishable": True}}
            ),
        )
        monkeypatch.setattr(
            "src.main._attach_run_summary",
            lambda result, passed_args, preflight: call_order.append("summary"),
        )
        monkeypatch.setattr(
            "src.main._render_primary_output",
            lambda result, passed_args, targets, banner: (
                call_order.append("render") or (None, None, None)
            ),
        )
        monkeypatch.setattr(
            "src.main._persist_analysis_outputs",
            lambda result, passed_args, **kwargs: call_order.append("persist"),
        )
        monkeypatch.setattr(
            "src.main._maybe_save_rejection_record",
            lambda result, passed_args, **kwargs: fake_async("rejection"),
        )
        monkeypatch.setattr(
            "src.main._maybe_generate_article",
            lambda result,
            passed_args,
            targets,
            company_name,
            report,
            reporter,
            **kwargs: (fake_async("article", False)),
        )
        monkeypatch.setattr(
            "src.main._log_final_summary",
            lambda result, passed_args, article_generated: call_order.append("final"),
        )
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        assert asyncio.run(main()) == 0
        assert call_order == [
            "apply",
            "validate",
            "setup",
            "retrospective",
            "banner",
            "execute",
            "summary",
            "render",
            "persist",
            "rejection",
            "article",
            "final",
        ]

    def test_main_returns_two_for_cli_usage_error(self, monkeypatch):
        from src.main import main

        args = SimpleNamespace(retrospective_only=False)

        monkeypatch.setattr("src.main.parse_arguments", lambda: args)
        monkeypatch.setattr(
            "src.main._apply_runtime_overrides", lambda passed_args: None
        )
        monkeypatch.setattr(
            "src.main._validate_cli_args",
            lambda passed_args: (_ for _ in ()).throw(SystemExit(2)),
        )
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        assert asyncio.run(main()) == 2

    def test_main_returns_one_when_analysis_fails(self, monkeypatch):
        from src.main import OutputTargets, main

        args = SimpleNamespace(
            retrospective_only=False,
            ticker="6083.T",
            quick=False,
            strict=False,
            article=False,
            quiet=True,
            brief=False,
            svg=False,
            transparent=False,
            imagedir=None,
        )

        monkeypatch.setattr("src.main.parse_arguments", lambda: args)
        monkeypatch.setattr(
            "src.main._apply_runtime_overrides", lambda passed_args: None
        )
        monkeypatch.setattr("src.main._validate_cli_args", lambda passed_args: None)
        monkeypatch.setattr(
            "src.main._resolve_output_targets",
            lambda passed_args: OutputTargets(None, Path("images"), True),
        )
        monkeypatch.setattr("src.main._setup_runtime", lambda passed_args, targets: {})
        monkeypatch.setattr(
            "src.main._maybe_run_ticker_retrospective",
            lambda passed_args: _async_none(),
        )
        monkeypatch.setattr(
            "src.main._emit_start_banner", lambda passed_args, targets: "banner"
        )
        monkeypatch.setattr(
            "src.main._execute_analysis",
            lambda passed_args, targets, **kwargs: _async_result(None),
        )
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        assert asyncio.run(main()) == 1


class TestSavedDiagnostics:
    def test_attach_run_summary_recomputes_mode_aware_validity(self, monkeypatch):
        from src.main import _attach_run_summary

        class StubTracker:
            def get_total_stats(self):
                return {"failed_attempts": 0, "total_calls": 0}

        monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())
        monkeypatch.setattr("src.main.config.llm_provider", "google")

        result = {
            "pre_screening_result": "PASS",
            "investment_debate_state": {"count": 0},
            "value_trap_report": "",
            "fundamentals_report": (
                "### --- START DATA_BLOCK ---\n"
                "SECTOR: Industrials\n"
                "RAW_HEALTH_SCORE: 5/12\n"
                "ADJUSTED_HEALTH_SCORE: 41.7%\n"
                "RAW_GROWTH_SCORE: 1/6\n"
                "ADJUSTED_GROWTH_SCORE: 16.7%\n"
                "US_REVENUE_PERCENT: Not disclosed\n"
                "### --- END DATA_BLOCK ---"
            ),
            "final_trade_decision": "VERDICT: BUY",
            "analysis_validity": {
                "publishable": False,
                "required_failures": {"value_trap_report": {}},
                "optional_failures": {},
            },
            "artifact_statuses": {
                "market_report": {"complete": True, "ok": True, "content": "market"},
                "sentiment_report": {
                    "complete": True,
                    "ok": True,
                    "content": "sentiment",
                },
                "news_report": {"complete": True, "ok": True, "content": "news"},
                "value_trap_report": {
                    "complete": True,
                    "ok": False,
                    "error_kind": "dns_resolution",
                    "provider": "google",
                },
                "fundamentals_report": {
                    "complete": True,
                    "ok": True,
                    "content": (
                        "### --- START DATA_BLOCK ---\n"
                        "SECTOR: Industrials\n"
                        "RAW_HEALTH_SCORE: 5/12\n"
                        "ADJUSTED_HEALTH_SCORE: 41.7%\n"
                        "RAW_GROWTH_SCORE: 1/6\n"
                        "ADJUSTED_GROWTH_SCORE: 16.7%\n"
                        "US_REVENUE_PERCENT: Not disclosed\n"
                        "### --- END DATA_BLOCK ---"
                    ),
                },
                "final_trade_decision": {
                    "complete": True,
                    "ok": True,
                    "content": "VERDICT: BUY",
                },
            },
        }
        args = SimpleNamespace(article=False, quick=True)

        _attach_run_summary(result, args, provider_preflight={})

        assert result["analysis_validity"]["publishable"] is True
        assert "quick_mode" not in result
        assert result["run_summary"]["quick_mode"] is True
        assert result["run_summary"]["publishable"] is True
        assert result["run_summary"]["required_failures"] == []
        assert result["run_summary"]["optional_failures"] == ["value_trap_report"]

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

    def test_build_run_summary_counts_manual_tool_failures_and_multi_provider_usage(
        self, monkeypatch
    ):
        from langchain_core.messages import ToolMessage

        from src.main import build_run_summary

        class StubTracker:
            def get_total_stats(self):
                return {
                    "failed_attempts": 0,
                    "total_calls": 2,
                }

        monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())
        monkeypatch.setattr("src.main.config.llm_provider", "google")

        result = {
            "analysis_validity": {"publishable": True},
            "consultant_tool_failures": 1,
            "artifact_statuses": {
                "consultant_review": {
                    "complete": True,
                    "ok": False,
                    "provider": "openai",
                }
            },
            "messages": [
                ToolMessage(
                    content='{"error":"invalid key","provider":"fmp","failure_kind":"auth_error"}',
                    tool_call_id="call_1",
                    name="spot_check_metric_alt",
                ),
                ToolMessage(
                    content="TOOL_ERROR: runner exploded",
                    tool_call_id="call_2",
                    name="fetch_reference_content",
                ),
            ],
        }

        summary = build_run_summary(
            result,
            quick_mode=False,
            article_requested=False,
        )

        assert summary["tool_failures"] == 3
        assert summary["llm_provider"] == "multi-provider"
        assert summary["llm_providers_used"] == ["google", "openai"]

    def test_build_run_summary_includes_quick_and_deep_models(self, monkeypatch):
        from src.main import build_run_summary

        class StubTracker:
            def get_total_stats(self):
                return {
                    "failed_attempts": 0,
                    "total_calls": 0,
                }

        monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())
        monkeypatch.setattr("src.main.config.quick_think_llm", "gemini-3-flash-preview")
        monkeypatch.setattr("src.main.config.deep_think_llm", "gemini-3-pro-preview")

        summary = build_run_summary(
            {"analysis_validity": {"publishable": True}},
            quick_mode=True,
            article_requested=False,
        )

        assert summary["quick_model"] == "gemini-3-flash-preview"
        assert summary["deep_model"] == "gemini-3-pro-preview"

    def test_build_run_summary_includes_macro_context_metadata(self, monkeypatch):
        from src.main import build_run_summary

        class StubTracker:
            def get_total_stats(self):
                return {
                    "failed_attempts": 0,
                    "total_calls": 1,
                }

        monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())
        monkeypatch.setattr("src.main.config.llm_provider", "google")

        summary = build_run_summary(
            {
                "analysis_validity": {"publishable": True},
                "macro_context_status": "generated_fallback",
                "macro_context_region": "SEA",
                "macro_context_report": "brief",
                "macro_context_injected_into_news": True,
            },
            quick_mode=True,
            article_requested=False,
        )

        assert summary["macro_context_status"] == "generated_fallback"
        assert summary["macro_context_region"] == "SEA"
        assert summary["macro_context_report_present"] is True
        assert summary["macro_context_injected_into_news"] is True

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
        assert payload["metadata"]["llm_provider"] == "google"

    def test_save_results_includes_macro_context_metadata(self, tmp_path, monkeypatch):
        from src.main import save_results_to_file

        monkeypatch.setattr("src.main.config.results_dir", str(tmp_path))
        monkeypatch.setattr("src.main.config.enable_memory", False)
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
            "pre_screening_result": "PASS",
            "investment_debate_state": {"count": 1},
            "analysis_validity": {"publishable": True},
            "artifact_statuses": {},
            "prompts_used": {
                "macro_context_analyst": {
                    "agent_name": "Macro Context Analyst",
                    "version": "1.0",
                    "execution_path": "pre_graph",
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
        }

        with patch("src.main.logger") as mock_logger:
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
        assert (
            payload["prompts_metadata"]["prompts_used"]["macro_context_analyst"][
                "agent_name"
            ]
            == "Macro Context Analyst"
        )
        snapshot_calls = [
            call
            for call in mock_logger.info.call_args_list
            if call.args and call.args[0] == "analysis_artifact_macro_snapshot"
        ]
        assert len(snapshot_calls) == 1
        assert snapshot_calls[0].kwargs["ticker"] == "7203.T"
        assert snapshot_calls[0].kwargs["has_macro_context_block"] is True
        assert snapshot_calls[0].kwargs["has_run_summary_macro_fields"] is True
        assert snapshot_calls[0].kwargs["has_macro_prompt_metadata"] is True
        assert snapshot_calls[0].kwargs["has_macro_token_row"] is True

    def test_save_results_prefers_direct_macro_context_fields(
        self, tmp_path, monkeypatch
    ):
        from src.main import save_results_to_file

        monkeypatch.setattr("src.main.config.results_dir", str(tmp_path))
        monkeypatch.setattr("src.main.config.enable_memory", False)
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
            "run_summary": {
                "quick_mode": True,
                "tool_calls": 0,
                "publishable": True,
                "macro_context_status": "failed",
                "macro_context_region": "GLOBAL",
                "macro_context_report_present": False,
                "macro_context_injected_into_news": False,
            },
            "macro_context_status": "generated",
            "macro_context_region": "JAPAN",
            "macro_context_report": "brief",
            "macro_context_llm_invoked": True,
            "macro_context_generated_at": "2026-04-18T00:00:00+00:00",
            "macro_context_injected_into_news": True,
        }

        output_path = save_results_to_file(result, "7203.T", quick_mode=True)
        payload = json.loads(output_path.read_text())

        assert payload["macro_context"]["status"] == "generated"
        assert payload["macro_context"]["region"] == "JAPAN"
        assert payload["macro_context"]["report_present"] is True
        assert payload["macro_context"]["injected_into_news"] is True
        assert payload["macro_context"]["llm_invoked"] is True

    def test_save_results_warns_on_macro_artifact_mismatch(self, tmp_path, monkeypatch):
        from src.main import save_results_to_file

        monkeypatch.setattr("src.main.config.results_dir", str(tmp_path))
        monkeypatch.setattr("src.main.config.enable_memory", False)
        monkeypatch.setattr("src.prompts.get_all_prompts", lambda: {})

        class StubTracker:
            def get_total_stats(self):
                return {
                    "failed_attempts": 0,
                    "total_calls": 1,
                    "total_agents": 0,
                    "total_prompt_tokens": 10,
                    "total_completion_tokens": 5,
                    "total_tokens": 15,
                    "total_cost_usd": 0.1,
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
            "run_summary": {
                "quick_mode": True,
                "tool_calls": 0,
                "publishable": True,
                "macro_context_status": "generated",
                "macro_context_region": "JAPAN",
                "macro_context_report_present": True,
                "macro_context_injected_into_news": True,
            },
            "macro_context_status": "generated",
            "macro_context_region": "JAPAN",
            "macro_context_report": "brief",
            "macro_context_llm_invoked": True,
            "macro_context_generated_at": "2026-04-18T00:00:00+00:00",
            "macro_context_injected_into_news": True,
            "prompts_used": {},
        }

        with patch("src.main.logger") as mock_logger:
            save_results_to_file(result, "7203.T", quick_mode=True)

        mock_logger.warning.assert_any_call(
            "analysis_artifact_macro_mismatch",
            ticker="7203.T",
            macro_expected=True,
            macro_llm_invoked=True,
            macro_context_injected_into_news=True,
            has_macro_context_block=True,
            has_run_summary_macro_fields=True,
            has_macro_prompt_metadata=False,
            has_macro_token_row=False,
        )

    def test_save_results_uses_read_only_memory_stats_helper(
        self, tmp_path, monkeypatch
    ):
        from src.main import save_results_to_file

        monkeypatch.setattr("src.main.config.results_dir", str(tmp_path))
        monkeypatch.setattr("src.main.config.enable_memory", True)
        monkeypatch.setattr("src.prompts.get_all_prompts", lambda: {})
        monkeypatch.setattr(
            "src.memory.get_ticker_memory_stats",
            lambda ticker: {
                "bull_researcher": {"available": True, "name": "bull", "count": 1}
            },
        )

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

        with patch(
            "src.memory.create_memory_instances",
            side_effect=AssertionError("save path should not recreate ticker memories"),
        ):
            output_path = save_results_to_file(
                {
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
                    "run_summary": {"llm_provider": "multi-provider"},
                },
                "1308.HK",
                quick_mode=False,
            )

        payload = json.loads(output_path.read_text())
        assert payload["memory_statistics"]["bull_researcher"]["count"] == 1
        assert payload["metadata"]["llm_provider"] == "multi-provider"

    def test_save_results_updates_index_for_next_indexed_load(
        self, tmp_path, monkeypatch
    ):
        from src.ibkr.reconciler import load_latest_analyses
        from src.main import save_results_to_file

        monkeypatch.setattr("src.main.config.results_dir", str(tmp_path))
        monkeypatch.setattr("src.main.config.enable_memory", False)
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

        seed_result = {
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
            "run_summary": {"quick_mode": False, "tool_calls": 0, "publishable": True},
        }
        save_results_to_file(seed_result, "7203.T", quick_mode=False)
        load_latest_analyses(tmp_path)

        second_result = {
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
            "run_summary": {"quick_mode": True, "tool_calls": 0, "publishable": True},
        }
        output_path = save_results_to_file(second_result, "6083.T", quick_mode=True)

        events = []
        analyses = load_latest_analyses(tmp_path, progress=events.append)

        assert "6083.T" in analyses
        assert analyses["6083.T"].file_path == str(output_path)
        assert any(event.phase == "indexed" for event in events)
        assert not any(event.phase == "rebuilding_index" for event in events)

    def test_save_results_uses_incremental_update_when_mtime_is_stale_but_count_matches(
        self, tmp_path, monkeypatch
    ):
        from src.ibkr.reconciler import _analysis_index_path, load_latest_analyses
        from src.main import save_results_to_file

        monkeypatch.setattr("src.main.config.results_dir", str(tmp_path))
        monkeypatch.setattr("src.main.config.enable_memory", False)
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

        seed_result = {
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
            "run_summary": {"quick_mode": False, "tool_calls": 0, "publishable": True},
        }
        save_results_to_file(seed_result, "7203.T", quick_mode=False)
        load_latest_analyses(tmp_path)

        index_path = _analysis_index_path(tmp_path)
        payload = json.loads(index_path.read_text())
        payload["results_dir_mtime_ns"] = int(payload["results_dir_mtime_ns"]) - 1
        index_path.write_text(json.dumps(payload))

        second_result = {
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
            "run_summary": {"quick_mode": True, "tool_calls": 0, "publishable": True},
        }

        with patch("src.main.logger") as mock_logger:
            output_path = save_results_to_file(second_result, "6083.T", quick_mode=True)

        analyses = load_latest_analyses(tmp_path)
        assert "6083.T" in analyses
        assert analyses["6083.T"].file_path == str(output_path)
        accepted_calls = [
            call
            for call in mock_logger.debug.call_args_list
            + mock_logger.info.call_args_list
            if call.args and call.args[0] == "analysis_index_refreshed_after_save"
        ]
        assert not accepted_calls

    def test_save_results_rebuilds_index_when_incremental_update_skips(
        self, tmp_path, monkeypatch
    ):
        from src.ibkr.reconciler import _analysis_index_path, load_latest_analyses
        from src.main import save_results_to_file

        monkeypatch.setattr("src.main.config.results_dir", str(tmp_path))
        monkeypatch.setattr("src.main.config.enable_memory", False)
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

        seed_result = {
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
            "run_summary": {"quick_mode": False, "tool_calls": 0, "publishable": True},
        }
        save_results_to_file(seed_result, "7203.T", quick_mode=False)
        load_latest_analyses(tmp_path)

        index_path = _analysis_index_path(tmp_path)
        payload = json.loads(index_path.read_text())
        payload["results_dir_mtime_ns"] = int(payload["results_dir_mtime_ns"]) - 1
        payload["total_files"] = 0
        index_path.write_text(json.dumps(payload))

        second_result = {
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
            "run_summary": {"quick_mode": True, "tool_calls": 0, "publishable": True},
        }

        with patch("src.main.logger") as mock_logger:
            output_path = save_results_to_file(second_result, "6083.T", quick_mode=True)

        analyses = load_latest_analyses(tmp_path)
        assert "6083.T" in analyses
        assert analyses["6083.T"].file_path == str(output_path)
        mock_logger.info.assert_any_call(
            "analysis_index_refreshed_after_save",
            ticker="6083.T",
            path=str(tmp_path),
            refreshed_count=len(analyses),
        )


class TestSavedFileBannerRemoval:
    """
    Saved .md files must contain only the report body — no startup banner (A1 fix).

    The banner is already emitted to stdout by _emit_start_banner(); prepending it
    to the file was redundant and polluted saved reports.
    """

    BANNER_SENTINEL = "# Multi-Agent Investment Analysis System"

    def _make_report_result(self):
        return {
            "final_trade_decision": "Action: BUY\n\nStrong fundamentals.",
            "market_report": "RSI at 55.",
            "fundamentals_report": "P/E: 14.",
        }

    def _make_output_targets(self, output_file):
        from src.main import OutputTargets

        return OutputTargets(
            output_file=output_file,
            image_dir=output_file.parent / "images",
            skip_charts=True,
        )

    def test_saved_markdown_does_not_start_with_banner(self, tmp_path):
        """Written file must not begin with the startup banner block."""
        from types import SimpleNamespace

        from src.main import _render_primary_output, get_welcome_banner

        output_file = tmp_path / "report.md"
        args = SimpleNamespace(
            ticker="TST",
            brief=False,
            quiet=False,
            quick=False,
            svg=False,
            transparent=False,
        )
        welcome_banner = get_welcome_banner("TST", quick_mode=False)
        _render_primary_output(
            self._make_report_result(),
            args,
            self._make_output_targets(output_file),
            welcome_banner,
        )
        content = output_file.read_text()
        assert not content.startswith(
            self.BANNER_SENTINEL
        ), "Saved file must not start with the startup banner"

    def test_saved_markdown_starts_with_report_title(self, tmp_path):
        """First non-blank line of the saved file must be the report title (# TICKER ...)."""
        from types import SimpleNamespace

        from src.main import _render_primary_output, get_welcome_banner

        output_file = tmp_path / "report.md"
        args = SimpleNamespace(
            ticker="TST",
            brief=False,
            quiet=False,
            quick=False,
            svg=False,
            transparent=False,
        )
        welcome_banner = get_welcome_banner("TST", quick_mode=False)
        _render_primary_output(
            self._make_report_result(),
            args,
            self._make_output_targets(output_file),
            welcome_banner,
        )
        content = output_file.read_text()
        first_non_blank = next(
            (line for line in content.splitlines() if line.strip()), ""
        )
        assert first_non_blank.startswith(
            "# "
        ), f"First non-blank line should be a markdown title, got: {first_non_blank!r}"

    def test_brief_mode_saved_file_no_banner(self, tmp_path):
        """Brief mode with --output also writes report-only (no banner)."""
        from types import SimpleNamespace

        from src.main import _render_primary_output, get_welcome_banner

        output_file = tmp_path / "brief.md"
        args = SimpleNamespace(
            ticker="TST",
            brief=True,
            quiet=False,
            quick=False,
            svg=False,
            transparent=False,
        )
        welcome_banner = get_welcome_banner("TST", quick_mode=False)
        _render_primary_output(
            self._make_report_result(),
            args,
            self._make_output_targets(output_file),
            welcome_banner,
        )
        content = output_file.read_text()
        assert self.BANNER_SENTINEL not in content


# ---------------------------------------------------------------------------
# rich API smoke tests
# These exercise every rich symbol imported by src/main.py.  If a future
# rich major version removes or renames Console, Panel, Table, or box.ROUNDED
# the relevant test will fail with an AttributeError or ImportError before
# any analysis code runs, making the breakage immediately obvious.
# ---------------------------------------------------------------------------


class TestRichApiSurface:
    """Smoke-tests for the rich symbols used in src/main.py."""

    def test_console_importable_and_instantiable(self):
        import io

        from rich.console import Console

        buf = io.StringIO()
        c = Console(file=buf, width=80)
        assert c is not None

    def test_panel_importable_and_renderable(self):
        import io

        from rich.console import Console
        from rich.panel import Panel

        buf = io.StringIO()
        c = Console(file=buf, width=80)
        panel = Panel("content text", title="Test Title", border_style="green")
        c.print(panel)
        output = buf.getvalue()
        assert "content text" in output
        assert "Test Title" in output

    def test_table_add_column_and_row(self):
        import io

        from rich import box
        from rich.console import Console
        from rich.table import Table

        buf = io.StringIO()
        c = Console(file=buf, width=120)
        t = Table(show_header=True, box=box.ROUNDED)
        t.add_column("Agent", style="cyan")
        t.add_column("Value", style="green", justify="right")
        t.add_row("test-agent", "42")
        c.print(t)
        output = buf.getvalue()
        assert "Agent" in output
        assert "test-agent" in output
        assert "42" in output

    def test_box_rounded_attribute_exists(self):
        from rich import box

        assert hasattr(box, "ROUNDED"), "box.ROUNDED removed from rich"

    def test_console_markup_styling(self):
        """console.print with markup strings must not raise."""
        import io

        from rich.console import Console

        buf = io.StringIO()
        c = Console(file=buf, width=80)
        # These markup patterns appear verbatim in main.py
        c.print("[bold cyan]Token Usage Summary:[/bold cyan]")
        c.print("[yellow]Warning: test[/yellow]")
        c.print("[dim]Word count: 123 words[/dim]")
        assert buf.getvalue()  # non-empty output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
