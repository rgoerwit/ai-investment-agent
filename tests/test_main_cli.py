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


@pytest.fixture
def stub_observability(monkeypatch):
    """Keep CLI orchestration unit tests independent of real tracing shutdown."""
    from src.observability import NoopObservabilityRuntime

    monkeypatch.setattr(
        "src.observability.get_observability_runtime",
        lambda *args, **kwargs: NoopObservabilityRuntime(),
    )
    monkeypatch.setattr("src.observability.flush_traces", lambda: None)


class TestStrictModeCLI:
    """Test --strict CLI flag is wired correctly."""

    def test_strict_flag_parsed_from_cli(self):
        """--strict sets args.strict = True."""
        from src.cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--strict"])
        assert args.strict is True

    def test_no_strict_flag_defaults_false(self):
        """Without --strict, args.strict = False."""
        from src.cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK"])
        assert args.strict is False


class TestOutputCompanyNameLookup:
    def test_load_company_name_for_output_retries_normalized_alias(self):
        from src.output import _load_company_name_for_output

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
            assert (
                _load_company_name_for_output(
                    "TRUE.B.ST",
                    thread_pool_executor_cls=lambda **_kwargs: _Executor(),
                )
                == "Truecaller AB"
            )

        assert requested_symbols == ["TRUE.B.ST", "TRUE-B.ST"]

    def test_load_company_name_for_output_returns_none_on_timeout(self):
        from src.output import _load_company_name_for_output

        fake_yfinance = MagicMock()
        fake_yfinance.Ticker.return_value.info = {"longName": "Should Not Return"}

        mock_future = MagicMock()
        mock_future.result.side_effect = FuturesTimeoutError()

        mock_executor = MagicMock()
        mock_executor.submit.return_value = mock_future

        with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
            assert (
                _load_company_name_for_output(
                    "SNTIA.OL",
                    thread_pool_executor_cls=lambda **_kwargs: mock_executor,
                )
                is None
            )
        mock_executor.shutdown.assert_called()

    def test_run_analysis_warns_with_lookup_candidates_after_company_name_exhaustion(
        self,
    ):
        from src.main import run_analysis
        from src.ticker_utils import CompanyNameResult

        fake_tracker = MagicMock()
        fake_graph = MagicMock()
        fake_graph.ainvoke = AsyncMock(return_value={})
        fake_macro_context = {
            "report": "",
            "region": "EUROPE",
            "status": "failed",
            "generated_at": None,
            "llm_invoked": False,
            "prompt_used": None,
        }

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
            # Name is unresolved but data exists — the vacuum gate must not fire.
            patch("src.main._is_total_data_vacuum", new=AsyncMock(return_value=False)),
            patch(
                "src.main._prefetch_macro_context",
                new=AsyncMock(return_value=fake_macro_context),
            ),
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
        assert result["macro_regime_block"] == {}
        assert result["macro_regime_raw"] == ""
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

        macro_result = {
            "report": "### EQUITY REGIME\n- Summary: Risk appetite is mixed.",
            "region": "JAPAN",
            "status": "cached",
            "generated_at": None,
            "llm_invoked": False,
            "prompt_used": None,
            "regime_block_dict": {
                "risk_appetite": "RISK_OFF",
                "shock_type": "ENERGY",
                "shock_phase": "ACUTE",
                "equity_transmission": "EARNINGS_PRESSURE",
                "dip_posture": "WAIT_FOR_CONFIRMATION",
                "confidence": "MEDIUM",
                "present": True,
            },
            "regime_raw": "MACRO_REGIME_BLOCK:\nRISK_APPETITE: RISK_OFF",
        }

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
                "src.main._prefetch_macro_context",
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
        assert context.macro_context_report == macro_result["report"]
        assert context.macro_context_region == "JAPAN"
        assert context.macro_context_status == "cached"
        assert context.macro_regime["risk_appetite"] == "RISK_OFF"
        assert result["macro_regime_block"]["risk_appetite"] == "RISK_OFF"
        assert result["macro_regime_raw"].startswith("MACRO_REGIME_BLOCK:")

    @pytest.mark.parametrize(
        ("quick_mode", "configured_rounds", "expected_rounds"),
        [
            (False, 1, 1),
            (False, 2, 2),
            (True, 2, 1),
        ],
    )
    def test_run_analysis_wires_debate_rounds_behaviorally(
        self,
        monkeypatch,
        quick_mode,
        configured_rounds,
        expected_rounds,
    ):
        """MAX_DEBATE_ROUNDS must reach both graph builder and TradingContext."""
        from src.main import config, run_analysis
        from src.ticker_utils import CompanyNameResult

        fake_tracker = MagicMock()
        captured_context = {}

        async def _capture_ainvoke(_state, *, config):
            captured_context["context"] = config["configurable"]["context"]
            return {}

        fake_graph = MagicMock()
        fake_graph.ainvoke = AsyncMock(side_effect=_capture_ainvoke)
        macro_result = {
            "report": "",
            "region": "GLOBAL",
            "status": "disabled",
            "generated_at": None,
            "llm_invoked": False,
            "prompt_used": None,
            "regime_block_dict": {},
            "regime_raw": "",
        }

        monkeypatch.setattr(config, "max_debate_rounds", configured_rounds)

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
                "src.main._prefetch_macro_context",
                new=AsyncMock(return_value=macro_result),
            ),
            patch(
                "src.graph.create_trading_graph", return_value=fake_graph
            ) as create_graph,
            patch("src.token_tracker.get_tracker", return_value=fake_tracker),
            patch("src.main.build_analysis_validity", return_value={"ok": True}),
        ):
            result = asyncio.run(
                run_analysis(
                    ticker="0005.HK",
                    quick_mode=quick_mode,
                    strict_mode=False,
                    skip_charts=True,
                )
            )

        assert result["analysis_validity"] == {"ok": True}
        assert create_graph.call_args.kwargs["max_debate_rounds"] == expected_rounds
        assert captured_context["context"].max_debate_rounds == expected_rounds

    def test_strict_and_quick_composable(self):
        """--strict --quick can be combined without conflict."""
        from src.cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--strict", "--quick"])
        assert args.strict is True
        assert args.quick is True

    def test_strict_with_quiet_composable(self):
        """--strict --quiet can be combined (batch use case)."""
        from src.cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--strict", "--quiet"])
        assert args.strict is True
        assert args.quiet is True

    def test_strict_with_output_composable(self):
        """--strict with --output is valid."""
        from src.cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            ["--ticker", "0005.HK", "--strict", "--output", "results/test.md"]
        )
        assert args.strict is True
        assert args.output == "results/test.md"

    def test_strict_quick_quiet_all_composable(self):
        """--strict --quick --quiet can all be combined (pipeline batch mode)."""
        from src.cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(
            ["--ticker", "0005.HK", "--strict", "--quick", "--quiet"]
        )
        assert args.strict is True
        assert args.quick is True
        assert args.quiet is True

    def test_capture_baseline_flag_parsed_from_cli(self):
        """--capture-baseline enables baseline capture mode."""
        from src.cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--capture-baseline"])
        assert args.capture_baseline is True

    def test_capture_baseline_cleanup_flag_parsed_from_cli(self):
        from src.cli import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args(["--ticker", "0005.HK", "--capture-baseline-cleanup"])
        assert args.capture_baseline_cleanup is True

    def test_enable_langfuse_flag_parsed_from_cli(self):
        from src.cli import build_arg_parser

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
        from src.cli import parse_arguments

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
        macro_result = {
            "report": "brief",
            "region": "GLOBAL",
            "status": "generated",
            "generated_at": "2026-04-18T00:00:00+00:00",
            "llm_invoked": True,
            "prompt_used": {"agent_name": "Macro Context Analyst", "version": "1.0"},
            "regime_block_dict": {
                "risk_appetite": "UNCERTAIN",
                "shock_type": "NONE",
                "shock_phase": "NONE",
                "equity_transmission": "UNCERTAIN",
                "dip_posture": "WAIT_FOR_CONFIRMATION",
                "confidence": "LOW",
                "present": False,
            },
            "regime_raw": "",
        }

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
                "src.main._prefetch_macro_context",
                new=AsyncMock(return_value=macro_result),
            ) as prefetch_macro_context,
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
        assert prefetch_macro_context.await_count == 1
        assert prefetch_macro_context.await_args.args[0] == "0005.HK"
        assert prefetch_macro_context.await_args.kwargs["callbacks"] == [
            tracing_callback
        ]
        assert result["prompts_used"]["macro_context_analyst"]["agent_name"] == (
            "Macro Context Analyst"
        )

    @pytest.mark.asyncio
    async def test_prefetch_macro_context_logs_completion(self):
        from src.main import _prefetch_macro_context

        macro_result = SimpleNamespace(
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
                "src.macro_context.get_macro_context",
                new=AsyncMock(return_value=macro_result),
            ),
        ):
            result = await _prefetch_macro_context("RIC.AX", "2026-04-19")

        assert result == {
            "report": "brief",
            "region": "AUSTRALIA",
            "status": "generated",
            "generated_at": "2026-04-19T13:49:10.364729+00:00",
            "llm_invoked": True,
            "prompt_used": {"agent_name": "Macro Context Analyst", "version": "1.0"},
            "regime_block_dict": {
                "risk_appetite": "UNCERTAIN",
                "shock_type": "NONE",
                "shock_phase": "NONE",
                "equity_transmission": "UNCERTAIN",
                "dip_posture": "WAIT_FOR_CONFIRMATION",
                "confidence": "LOW",
                "present": False,
            },
            "regime_raw": "",
            # Summarizer-prompt fingerprint, carried so the retrospective can
            # tell a changed macro classifier from a changed world. None here
            # because the stubbed MacroContextResult sets no fingerprint.
            "fingerprint": None,
        }
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


class TestRuntimeServiceHookConfig:
    def test_debug_flag_implies_verbose(self, monkeypatch):
        from src.cli import build_arg_parser, parse_arguments

        parser = build_arg_parser()
        parsed = parser.parse_args(["--ticker", "6083.T", "--debug"])
        assert parsed.debug is True

        monkeypatch.setattr(sys, "argv", ["prog", "--ticker", "6083.T", "--debug"])
        args = parse_arguments()

        assert args.debug is True
        assert args.verbose is True

    def test_build_runtime_services_from_config_audit_hook_is_opt_in(self, monkeypatch):
        from src.main import build_runtime_services_from_config

        monkeypatch.setattr(
            "src.main.config.untrusted_content_inspection_enabled", False
        )
        # Pin MCP off so the assertion only inspects the audit/inspection wiring;
        # MCP_ENABLED in the operator's .env would otherwise add MCPBudgetHook.
        monkeypatch.setattr("src.main.config.mcp_enabled", False)

        services = build_runtime_services_from_config(enable_tool_audit=False)
        assert [type(h).__name__ for h in services.tool_service.hooks] == [
            "EvidenceRecorder"
        ]

        services = build_runtime_services_from_config(enable_tool_audit=True)
        hook_types = [type(h).__name__ for h in services.tool_service.hooks]
        assert hook_types == ["EvidenceRecorder", "LoggingToolAuditHook"]

    def test_build_runtime_services_from_config_audit_hook_coexists_with_inspection(
        self, monkeypatch
    ):
        from src.main import build_runtime_services_from_config

        monkeypatch.setattr(
            "src.main.config.untrusted_content_inspection_enabled", True
        )
        monkeypatch.setattr("src.main.config.untrusted_content_backend", "null")
        monkeypatch.setattr("src.main.config.untrusted_content_inspection_mode", "warn")
        monkeypatch.setattr(
            "src.main.config.untrusted_content_fail_policy", "fail_open"
        )

        services = build_runtime_services_from_config(enable_tool_audit=True)
        hook_types = [type(h).__name__ for h in services.tool_service.hooks]
        assert "LoggingToolAuditHook" in hook_types
        assert "ContentInspectionHook" in hook_types

    def test_build_runtime_services_from_config_builds_mcp_when_inspection_disabled(
        self, monkeypatch, tmp_path: Path
    ):
        from src.main import build_runtime_services_from_config

        registry_path = tmp_path / "mcp_servers.json"
        registry_path.write_text(
            json.dumps(
                {
                    "servers": [
                        {
                            "id": "fmp_remote",
                            "description": "FMP",
                            "transport": "streamable_http",
                            "base_url": "https://example.test/mcp",
                            "enabled": True,
                            "scopes": ["consultant"],
                            "tool_allowlist": ["quote"],
                            "trust_tier": "official_vendor",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "src.main.config.untrusted_content_inspection_enabled", False
        )
        monkeypatch.setattr("src.main.config.mcp_enabled", True)
        monkeypatch.setattr("src.main.config.mcp_servers_path", registry_path)
        monkeypatch.setattr(
            "src.main.config.mcp_usage_db_path",
            tmp_path / "runtime" / "mcp_usage.db",
        )

        services = build_runtime_services_from_config(enable_tool_audit=False)

        assert services.mcp_runtime is not None
        hook_types = [type(h).__name__ for h in services.tool_service.hooks]
        assert "ContentInspectionHook" not in hook_types
        assert "MCPBudgetHook" in hook_types

    def test_build_runtime_services_from_config_degrades_when_mcp_registry_missing(
        self, monkeypatch, tmp_path: Path
    ):
        from src.main import build_runtime_services_from_config

        monkeypatch.setattr("src.main.config.mcp_enabled", True)
        monkeypatch.setattr(
            "src.main.config.mcp_servers_path",
            tmp_path / "missing_mcp_servers.json",
        )
        monkeypatch.setattr(
            "src.main.config.mcp_usage_db_path",
            tmp_path / "runtime" / "mcp_usage.db",
        )

        services = build_runtime_services_from_config(enable_tool_audit=False)

        assert services.mcp_runtime is None
        hook_types = [type(h).__name__ for h in services.tool_service.hooks]
        assert "MCPBudgetHook" not in hook_types

    def test_configure_content_inspection_from_config_installs_tool_hook(
        self, monkeypatch
    ):
        from src.main import configure_content_inspection_from_config

        monkeypatch.setattr(
            "src.main.config.untrusted_content_inspection_enabled", True
        )
        monkeypatch.setattr("src.main.config.untrusted_content_backend", "null")
        monkeypatch.setattr("src.main.config.untrusted_content_inspection_mode", "warn")
        monkeypatch.setattr(
            "src.main.config.untrusted_content_fail_policy", "fail_open"
        )

        services = configure_content_inspection_from_config()
        hook_types = [type(h).__name__ for h in services.tool_service.hooks]
        assert "ContentInspectionHook" in hook_types

    def test_configure_content_inspection_from_config_removes_tool_hook_when_disabled(
        self, monkeypatch
    ):
        from src.main import configure_content_inspection_from_config

        monkeypatch.setattr(
            "src.main.config.untrusted_content_inspection_enabled", False
        )

        services = configure_content_inspection_from_config()
        hook_types = [type(h).__name__ for h in services.tool_service.hooks]
        assert "ContentInspectionHook" not in hook_types

    def test_configure_content_inspection_from_config_rejects_unimplemented_backend(
        self, monkeypatch
    ):
        from src.main import configure_content_inspection_from_config

        monkeypatch.setattr(
            "src.main.config.untrusted_content_inspection_enabled", True
        )
        monkeypatch.setattr("src.main.config.untrusted_content_backend", "http")
        monkeypatch.setattr("src.main.config.untrusted_content_inspection_mode", "warn")
        monkeypatch.setattr(
            "src.main.config.untrusted_content_fail_policy", "fail_open"
        )

        with pytest.raises(ValueError, match="is not implemented"):
            configure_content_inspection_from_config()


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

    @pytest.mark.asyncio
    async def test_execute_analysis_adds_capture_hook_without_mutating_global_service(
        self,
    ):
        from src.main import _execute_analysis
        from src.runtime_services import RuntimeServices, build_provider_runtime
        from src.tooling.inspection_service import InspectionService
        from src.tooling.runtime import TOOL_SERVICE, ToolExecutionService

        base_hook = object()
        global_hooks_before = TOOL_SERVICE.hooks
        runtime_services = RuntimeServices(
            tool_service=ToolExecutionService([base_hook]),
            inspection_service=InspectionService(),
            providers=build_provider_runtime(),
        )

        class Capture:
            def make_tool_hook(self):
                return "capture_hook"

        args = SimpleNamespace(
            ticker="7203.T",
            quick=True,
            strict=False,
            svg=False,
            transparent=False,
        )
        output_targets = SimpleNamespace(image_dir=Path("."), skip_charts=True)

        with patch(
            "src.main.run_analysis", new=AsyncMock(return_value={"ok": True})
        ) as mock_run:
            result = await _execute_analysis(
                args,
                output_targets,
                baseline_capture=Capture(),
                runtime_services=runtime_services,
            )

        assert result == {"ok": True}
        passed_services = mock_run.await_args.kwargs["runtime_services"]
        assert runtime_services.tool_service.hooks == [base_hook]
        assert passed_services.tool_service.hooks == [base_hook, "capture_hook"]
        assert TOOL_SERVICE.hooks == global_hooks_before

    @pytest.mark.asyncio
    async def test_execute_analysis_does_not_leak_capture_hook_on_error(self):
        from src.main import _execute_analysis
        from src.runtime_services import RuntimeServices, build_provider_runtime
        from src.tooling.inspection_service import InspectionService
        from src.tooling.runtime import TOOL_SERVICE, ToolExecutionService

        global_hooks_before = TOOL_SERVICE.hooks
        runtime_services = RuntimeServices(
            tool_service=ToolExecutionService(["base_hook"]),
            inspection_service=InspectionService(),
            providers=build_provider_runtime(),
        )

        class Capture:
            def make_tool_hook(self):
                return "capture_hook"

        args = SimpleNamespace(
            ticker="7203.T",
            quick=True,
            strict=False,
            svg=False,
            transparent=False,
        )
        output_targets = SimpleNamespace(image_dir=Path("."), skip_charts=True)

        with patch(
            "src.main.run_analysis", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await _execute_analysis(
                    args,
                    output_targets,
                    baseline_capture=Capture(),
                    runtime_services=runtime_services,
                )

        assert runtime_services.tool_service.hooks == ["base_hook"]
        assert TOOL_SERVICE.hooks == global_hooks_before

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
        monkeypatch.setattr("src.main.run_provider_preflight", lambda: {"ok": True})

        configure_cli_logging(args)

        assert logging.getLogger("openai").level == logging.DEBUG
        assert logging.getLogger("httpx").level == logging.DEBUG
        assert logging.getLogger("httpcore").level == logging.DEBUG


class TestValidateCliArgs:
    def test_quick_with_svg_exits_2(self, capsys):
        from src.cli import _validate_cli_args

        args = SimpleNamespace(quick=True, transparent=False, svg=True)

        with pytest.raises(SystemExit) as exc_info:
            _validate_cli_args(args)

        assert exc_info.value.code == 2
        assert "--svg" in capsys.readouterr().err

    def test_quick_with_transparent_exits_2(self, capsys):
        from src.cli import _validate_cli_args

        args = SimpleNamespace(quick=True, transparent=True, svg=False)

        with pytest.raises(SystemExit) as exc_info:
            _validate_cli_args(args)

        assert exc_info.value.code == 2
        assert "--transparent" in capsys.readouterr().err


class TestResolveOutputTargets:
    def test_stdout_without_imagedir_disables_charts(self):
        from src.cli import _resolve_output_targets

        args = SimpleNamespace(output=None, imagedir=None, no_charts=False)

        targets = _resolve_output_targets(args)

        assert targets.output_file is None
        assert targets.image_dir == Path("images")
        assert targets.skip_charts is True

    def test_stdout_with_imagedir_preserves_chart_generation(self):
        from src.cli import _resolve_output_targets

        args = SimpleNamespace(output=None, imagedir="assets/charts", no_charts=False)

        targets = _resolve_output_targets(args)

        assert targets.output_file is None
        assert targets.image_dir == Path("assets/charts")
        assert targets.skip_charts is False

    def test_file_output_keeps_charts_enabled_by_default(self):
        from src.cli import _resolve_output_targets

        args = SimpleNamespace(
            output="results/report.md", imagedir=None, no_charts=False
        )

        targets = _resolve_output_targets(args)

        assert targets.output_file == Path("results/report.md")
        assert targets.image_dir == Path("results/images")
        assert targets.skip_charts is False


class TestMainOrchestration:
    @pytest.fixture(autouse=True)
    def _stub_tracing_runtime(self, stub_observability):
        """These orchestration tests exercise CLI control flow, not Langfuse internals."""
        return None

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

    def test_run_retrospective_only_skips_when_no_memory(self, monkeypatch):
        """`--no-memory --retrospective-only` must not write to lessons_learned.

        Mirrors the per-ticker gate at `_maybe_run_ticker_retrospective`.
        Distinct event name `retrospective_batch_skipped_no_memory` so log
        greps can tell the two paths apart.
        """
        from src.main import _run_retrospective_only

        called = {"count": 0}

        async def fake_run_retrospective(*_args, **_kwargs):
            called["count"] += 1
            return []

        monkeypatch.setattr(
            "src.retrospective.run_retrospective", fake_run_retrospective
        )

        args = SimpleNamespace(quiet=True, brief=False, no_memory=True)
        assert asyncio.run(_run_retrospective_only(args)) == 0
        assert called["count"] == 0, (
            "run_retrospective must NOT be called when --no-memory is set"
        )

    def test_retrospective_only_returns_early(self, monkeypatch):
        from src.cli import OutputTargets
        from src.main import main

        calls = []
        args = SimpleNamespace(retrospective_only=True)

        async def fake_retrospective_only(passed_args):
            calls.append(("retrospective_only", passed_args))
            return 0

        monkeypatch.setattr("src.main.cli.parse_arguments", lambda: args)
        monkeypatch.setattr("src.main.cli._validate_cli_args", lambda passed_args: None)
        monkeypatch.setattr(
            "src.main.cli._resolve_output_targets",
            lambda passed_args: OutputTargets(None, Path("images"), True),
        )
        monkeypatch.setattr(
            "src.main._setup_runtime",
            lambda passed_args, targets: ({}, object()),
        )
        monkeypatch.setattr("src.main._run_retrospective_only", fake_retrospective_only)
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        assert asyncio.run(main()) == 0
        assert calls == [("retrospective_only", args)]

    def test_main_success_path_returns_zero(self, monkeypatch):
        from src.cli import OutputTargets
        from src.main import main

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

        monkeypatch.setattr("src.main.cli.parse_arguments", lambda: args)
        monkeypatch.setattr(
            "src.main.cli._validate_cli_args",
            lambda passed_args: call_order.append("validate"),
        )
        monkeypatch.setattr(
            "src.main.cli._resolve_output_targets",
            lambda passed_args: OutputTargets(
                Path("results/report.md"), Path("results/images"), False
            ),
        )
        monkeypatch.setattr(
            "src.main._setup_runtime",
            lambda passed_args, targets: (
                call_order.append("setup") or {"google": {"dns": "ok"}},
                object(),
            ),
        )
        monkeypatch.setattr(
            "src.main._maybe_run_ticker_retrospective",
            lambda passed_args: fake_async("retrospective"),
        )
        monkeypatch.setattr(
            "src.main.output._emit_start_banner",
            lambda passed_args, targets, **kwargs: (
                call_order.append("banner") or "banner"
            ),
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
            "src.main.output._render_primary_output",
            lambda result, passed_args, targets, banner, **kwargs: (
                call_order.append("render") or (None, None, None)
            ),
        )
        monkeypatch.setattr(
            "src.main.persistence._persist_analysis_outputs",
            lambda result, passed_args, **kwargs: call_order.append("persist"),
        )
        monkeypatch.setattr(
            "src.main.persistence._maybe_save_rejection_record",
            lambda result, passed_args, **kwargs: fake_async("rejection"),
        )
        monkeypatch.setattr(
            "src.main.output._maybe_generate_article",
            lambda result, passed_args, targets, company_name, report, reporter, **kwargs: (
                fake_async("article", False)
            ),
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

        monkeypatch.setattr("src.main.cli.parse_arguments", lambda: args)
        monkeypatch.setattr(
            "src.main.cli._validate_cli_args",
            lambda passed_args: (_ for _ in ()).throw(SystemExit(2)),
        )
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        assert asyncio.run(main()) == 2

    def test_main_returns_one_when_analysis_fails(self, monkeypatch):
        from src.cli import OutputTargets
        from src.main import main

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

        monkeypatch.setattr("src.main.cli.parse_arguments", lambda: args)
        monkeypatch.setattr("src.main.cli._validate_cli_args", lambda passed_args: None)
        monkeypatch.setattr(
            "src.main.cli._resolve_output_targets",
            lambda passed_args: OutputTargets(None, Path("images"), True),
        )
        monkeypatch.setattr(
            "src.main._setup_runtime",
            lambda passed_args, targets: ({}, object()),
        )
        monkeypatch.setattr(
            "src.main._maybe_run_ticker_retrospective",
            lambda passed_args: _async_none(),
        )
        monkeypatch.setattr(
            "src.main.output._emit_start_banner",
            lambda passed_args, targets, **kwargs: "banner",
        )
        monkeypatch.setattr(
            "src.main._execute_analysis",
            lambda passed_args, targets, **kwargs: _async_result(None),
        )
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        assert asyncio.run(main()) == 1

    def test_main_keeps_results_dir_persistent_when_output_is_set(self, monkeypatch):
        from src.cli import OutputTargets
        from src.main import config, main

        output_file = Path("scratch/report.md")
        observed_results_dirs: list[Path] = []
        original_results_dir = Path("results")
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

        async def fake_execute(*_args, **_kwargs):
            observed_results_dirs.append(Path(config.results_dir))
            return {"analysis_validity": {"publishable": True}}

        async def fake_article(*_args, **_kwargs):
            observed_results_dirs.append(Path(config.results_dir))
            return False

        def fake_persist(*_args, **_kwargs):
            observed_results_dirs.append(Path(config.results_dir))

        monkeypatch.setattr("src.main.cli.parse_arguments", lambda: args)
        monkeypatch.setattr("src.main.config.results_dir", original_results_dir)
        monkeypatch.setattr("src.main.cli._validate_cli_args", lambda passed_args: None)
        monkeypatch.setattr(
            "src.main.cli._resolve_output_targets",
            lambda passed_args: OutputTargets(
                output_file, Path("scratch/images"), False
            ),
        )
        monkeypatch.setattr(
            "src.main._setup_runtime",
            lambda passed_args, targets: (
                observed_results_dirs.append(Path(config.results_dir))
                or {"google": {"dns": "ok"}},
                object(),
            ),
        )
        monkeypatch.setattr(
            "src.main._maybe_run_ticker_retrospective",
            lambda passed_args: (
                observed_results_dirs.append(Path(config.results_dir)) or _async_none()
            ),
        )
        monkeypatch.setattr(
            "src.main.output._emit_start_banner",
            lambda passed_args, targets, **kwargs: "banner",
        )
        monkeypatch.setattr("src.main._execute_analysis", fake_execute)
        monkeypatch.setattr(
            "src.main._attach_run_summary",
            lambda result, passed_args, preflight: None,
        )
        monkeypatch.setattr(
            "src.main.output._render_primary_output",
            lambda result, passed_args, targets, banner, **kwargs: (None, None, None),
        )
        monkeypatch.setattr(
            "src.main.persistence._persist_analysis_outputs", fake_persist
        )
        monkeypatch.setattr(
            "src.main.persistence._maybe_save_rejection_record",
            lambda result, passed_args, **kwargs: _async_none(),
        )
        monkeypatch.setattr("src.main.output._maybe_generate_article", fake_article)
        monkeypatch.setattr(
            "src.main._log_final_summary",
            lambda result, passed_args, article_generated: None,
        )
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        assert asyncio.run(main()) == 0
        assert observed_results_dirs
        assert all(path == original_results_dir for path in observed_results_dirs)
        assert Path(config.results_dir) == original_results_dir

    def test_main_restores_results_dir_after_failed_run_with_output(self, monkeypatch):
        from src.cli import OutputTargets
        from src.main import config, main

        original_results_dir = Path("results")
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

        monkeypatch.setattr("src.main.cli.parse_arguments", lambda: args)
        monkeypatch.setattr("src.main.config.results_dir", original_results_dir)
        monkeypatch.setattr("src.main.cli._validate_cli_args", lambda passed_args: None)
        monkeypatch.setattr(
            "src.main.cli._resolve_output_targets",
            lambda passed_args: OutputTargets(
                Path("scratch/report.md"), Path("scratch/images"), False
            ),
        )
        monkeypatch.setattr(
            "src.main._setup_runtime",
            lambda passed_args, targets: ({}, object()),
        )
        monkeypatch.setattr(
            "src.main._maybe_run_ticker_retrospective",
            lambda passed_args: _async_none(),
        )
        monkeypatch.setattr(
            "src.main.output._emit_start_banner",
            lambda passed_args, targets, **kwargs: "banner",
        )
        monkeypatch.setattr(
            "src.main._execute_analysis",
            lambda passed_args, targets, **kwargs: _async_result(None),
        )
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        assert asyncio.run(main()) == 1
        assert Path(config.results_dir) == original_results_dir

    def test_main_preserves_explicit_imagedir_without_retargeting_results_dir(
        self, monkeypatch
    ):
        from src.cli import OutputTargets
        from src.main import config, main

        output_file = Path("scratch/report.md")
        explicit_image_dir = Path("custom-images/charts")
        observed: dict[str, Path] = {}
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
            imagedir=str(explicit_image_dir),
        )

        async def fake_execute(_args, targets, **_kwargs):
            observed["results_dir"] = Path(config.results_dir)
            observed["image_dir"] = targets.image_dir
            return {"analysis_validity": {"publishable": True}}

        monkeypatch.setattr("src.main.cli.parse_arguments", lambda: args)
        monkeypatch.setattr("src.main.config.results_dir", Path("results"))
        monkeypatch.setattr("src.main.cli._validate_cli_args", lambda passed_args: None)
        monkeypatch.setattr(
            "src.main.cli._resolve_output_targets",
            lambda passed_args: OutputTargets(output_file, explicit_image_dir, False),
        )
        monkeypatch.setattr(
            "src.main._setup_runtime",
            lambda passed_args, targets: ({}, object()),
        )
        monkeypatch.setattr(
            "src.main._maybe_run_ticker_retrospective",
            lambda passed_args: _async_none(),
        )
        monkeypatch.setattr(
            "src.main.output._emit_start_banner",
            lambda passed_args, targets, **kwargs: "banner",
        )
        monkeypatch.setattr("src.main._execute_analysis", fake_execute)
        monkeypatch.setattr(
            "src.main._attach_run_summary",
            lambda result, passed_args, preflight: None,
        )
        monkeypatch.setattr(
            "src.main.output._render_primary_output",
            lambda result, passed_args, targets, banner, **kwargs: (None, None, None),
        )
        monkeypatch.setattr(
            "src.main.persistence._persist_analysis_outputs",
            lambda result, passed_args, **kwargs: None,
        )
        monkeypatch.setattr(
            "src.main.persistence._maybe_save_rejection_record",
            lambda result, passed_args, **kwargs: _async_none(),
        )
        monkeypatch.setattr(
            "src.main.output._maybe_generate_article",
            lambda result, passed_args, targets, company_name, report, reporter, **kwargs: (
                _async_result(False)
            ),
        )
        monkeypatch.setattr(
            "src.main._log_final_summary",
            lambda result, passed_args, article_generated: None,
        )
        monkeypatch.setattr(
            "src.cleanup.cleanup_async_resources", lambda: _async_none()
        )

        assert asyncio.run(main()) == 0
        assert observed["results_dir"] == Path("results")
        assert observed["image_dir"] == explicit_image_dir


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
            # A real live run carries a VALID canonical snapshot + decision trace;
            # _attach_run_summary now stamps the provenance contract, so these are
            # required for publishability (fail-closed).
            "analysis_snapshot": {"contract_status": "VALID", "claims": {}},
            "decision_trace": {"status": "VALID", "verdict": "BUY"},
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
        from src.persistence import build_run_summary

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
                # ok=True + a parseable non-caveated STATUS → genuine successful audit.
                "auditor_report": {
                    "complete": True,
                    "ok": True,
                    "content": "STATUS: CLEAN\nNo anomalies detected.",
                },
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

        from src.persistence import build_run_summary

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
        from src.persistence import build_run_summary

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
        from src.persistence import build_run_summary

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

    def test_build_run_summary_includes_compact_quick_consultant(self, monkeypatch):
        from src.persistence import build_run_summary

        class StubTracker:
            def get_total_stats(self):
                return {
                    "failed_attempts": 1,
                    "total_calls": 1,
                    "agents": {"Consultant": {"total_tokens": 1234}},
                    "call_attempts": [
                        {
                            "agent_name": "External Consultant",
                            "elapsed_seconds": 12.5,
                            "failure_kind": None,
                        },
                        {
                            "agent_name": "External Consultant",
                            "elapsed_seconds": 3.0,
                            "failure_kind": "timeout",
                        },
                    ],
                }

        monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())
        summary = build_run_summary(
            {
                "analysis_validity": {"publishable": True},
                "consultant_tool_failures": 2,
                "artifact_statuses": {
                    "consultant_review": {"complete": True, "ok": False}
                },
            },
            quick_mode=True,
            article_requested=False,
        )

        assert summary["quick_consultant"] == {
            "status": "failed",
            "elapsed_seconds": 15.5,
            "tokens": 1234,
            "attempts": 2,
            "timeout": True,
            "tool_failures": 2,
            "profile": "unknown",
        }

    def test_log_final_summary_emits_one_quick_slow_tail_warning(self, monkeypatch):
        from src import main

        class StubTracker:
            def get_total_stats(self):
                return {
                    "call_diagnostics": {
                        "timeout_seconds_lost": 61.0,
                        "consultant_timeout": True,
                        "slowest_call": {
                            "agent_name": "External Consultant",
                            "provider": "openai",
                            "model_name": "gpt-5.4-mini",
                            "status": "failure",
                            "failure_kind": "timeout",
                            "elapsed_seconds": 60.1,
                        },
                    }
                }

        logger = MagicMock()
        monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())
        monkeypatch.setattr(main, "logger", logger)

        main._log_final_summary(
            {"run_summary": {"quick_mode": True}},
            SimpleNamespace(ticker="TEST", quick=True),
            article_generated=False,
        )

        logger.info.assert_called_once()
        logger.warning.assert_called_once()
        warning = logger.warning.call_args
        assert warning.args == ("quick_run_slow_tail_warning",)
        assert warning.kwargs["ticker"] == "TEST"
        assert warning.kwargs["slowest_agent"] == "External Consultant"
        assert (
            warning.kwargs["suggested_knob"] == "CONSULTANT_QUICK_TOTAL_TIMEOUT_SECONDS"
        )

    def test_log_final_summary_skips_slow_tail_warning_for_normal_quick_run(
        self, monkeypatch
    ):
        from src import main

        class StubTracker:
            def get_total_stats(self):
                return {
                    "call_diagnostics": {
                        "timeout_seconds_lost": 0.0,
                        "consultant_timeout": False,
                        "slowest_call": {
                            "agent_name": "Market Analyst",
                            "elapsed_seconds": 5.0,
                        },
                    }
                }

        logger = MagicMock()
        monkeypatch.setattr("src.token_tracker.get_tracker", lambda: StubTracker())
        monkeypatch.setattr(main, "logger", logger)

        main._log_final_summary(
            {"run_summary": {"quick_mode": True}},
            SimpleNamespace(ticker="TEST", quick=True),
            article_generated=False,
        )

        logger.info.assert_called_once()
        logger.warning.assert_not_called()

    def test_warn_quick_timeout_config_drift_logs_once(self, monkeypatch):
        from src import main

        logger = MagicMock()
        monkeypatch.setattr(main, "logger", logger)
        monkeypatch.setattr(main, "_QUICK_TIMEOUT_CONFIG_WARNED", False)
        monkeypatch.setattr(main.config, "quick_llm_api_timeout_seconds", 300)
        monkeypatch.setattr(main.config, "llm_call_hard_timeout_seconds", 600.0)
        monkeypatch.setattr(main.config, "api_retry_attempts", 3)
        monkeypatch.setattr(main.config, "gemini_rpm_limit", 1000)

        args = SimpleNamespace(ticker="TEST", quick=True)
        main._warn_quick_timeout_config_drift(args)
        main._warn_quick_timeout_config_drift(args)

        logger.warning.assert_called_once()
        warning = logger.warning.call_args
        assert warning.args == ("quick_timeout_config_drift",)
        assert warning.kwargs["config"]["QUICK_LLM_API_TIMEOUT_SECONDS"] == 300
        assert warning.kwargs["config"]["LLM_CALL_HARD_TIMEOUT_SECONDS"] == 600.0
        assert warning.kwargs["config"]["API_RETRY_ATTEMPTS"] == 3
        assert warning.kwargs["config"]["GEMINI_RPM_LIMIT"] == 1000

    def test_warn_quick_timeout_config_drift_skips_full_mode(self, monkeypatch):
        from src import main

        logger = MagicMock()
        monkeypatch.setattr(main, "logger", logger)
        monkeypatch.setattr(main, "_QUICK_TIMEOUT_CONFIG_WARNED", False)
        monkeypatch.setattr(main.config, "quick_llm_api_timeout_seconds", 300)

        main._warn_quick_timeout_config_drift(
            SimpleNamespace(ticker="TEST", quick=False)
        )

        logger.warning.assert_not_called()

    def test_save_results_includes_pre_screening_and_run_summary(
        self, tmp_path, monkeypatch
    ):
        from langchain_core.messages import ToolMessage

        from src.persistence import save_results_to_file

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
            "red_flags": [
                {
                    "type": "PFIC_UNCERTAIN",
                    "severity": "WARNING",
                    "detail": "PFIC status unclear.",
                    "risk_penalty": 0.5,
                }
            ],
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
        assert payload["red_flags"] == result["red_flags"]
        assert payload["metadata"]["llm_provider"] == "google"

    def test_save_results_includes_macro_context_metadata(self, tmp_path, monkeypatch):
        from src.persistence import save_results_to_file

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

        mock_logger = MagicMock()
        output_path = save_results_to_file(
            result, "7203.T", quick_mode=True, logger_obj=mock_logger
        )
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
        from src.persistence import save_results_to_file

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
        from src.persistence import save_results_to_file

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

        mock_logger = MagicMock()
        save_results_to_file(result, "7203.T", quick_mode=True, logger_obj=mock_logger)

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
        from src.persistence import save_results_to_file

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
        from src.ibkr.analysis_index import load_latest_analyses
        from src.persistence import save_results_to_file

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
        from src.ibkr.analysis_index import _analysis_index_path, load_latest_analyses
        from src.persistence import save_results_to_file

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

        mock_logger = MagicMock()
        output_path = save_results_to_file(
            second_result, "6083.T", quick_mode=True, logger_obj=mock_logger
        )

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
        from src.ibkr.analysis_index import _analysis_index_path, load_latest_analyses
        from src.persistence import save_results_to_file

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

        mock_logger = MagicMock()
        output_path = save_results_to_file(
            second_result, "6083.T", quick_mode=True, logger_obj=mock_logger
        )

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
        from src.cli import OutputTargets

        return OutputTargets(
            output_file=output_file,
            image_dir=output_file.parent / "images",
            skip_charts=True,
        )

    def test_saved_markdown_does_not_start_with_banner(self, tmp_path):
        """Written file must not begin with the startup banner block."""
        from types import SimpleNamespace

        from src.output import _render_primary_output, get_welcome_banner

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
        assert not content.startswith(self.BANNER_SENTINEL), (
            "Saved file must not start with the startup banner"
        )

    def test_saved_markdown_starts_with_report_title(self, tmp_path):
        """First non-blank line of the saved file must be the report title (# TICKER ...)."""
        from types import SimpleNamespace

        from src.output import _render_primary_output, get_welcome_banner

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
        assert first_non_blank.startswith("# "), (
            f"First non-blank line should be a markdown title, got: {first_non_blank!r}"
        )

    def test_brief_mode_saved_file_no_banner(self, tmp_path):
        """Brief mode with --output also writes report-only (no banner)."""
        from types import SimpleNamespace

        from src.output import _render_primary_output, get_welcome_banner

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


class TestDegradationNotice:
    """An optional cross-check seat failing is publishable by design, which on
    2026-08-14 also made it invisible: three full-mode tickers ran against an
    unfunded provider with no cross-check at all and reported success. Losing
    those agents *lowers* the risk tally, so a degraded run reads as a cleaner
    stock unless something says otherwise."""

    @staticmethod
    def _degraded_result():
        return {
            "run_summary": {
                "optional_failures": ["auditor_report", "consultant_review"]
            },
            "artifact_statuses": {
                "auditor_report": {"ok": False, "error_kind": "auth_error"},
                "consultant_review": {"ok": False, "error_kind": "auth_error"},
            },
        }

    def test_names_each_failed_artifact_and_its_error_kind(self, capsys):
        from src.main import _print_degradation_notice

        _print_degradation_notice(self._degraded_result(), quiet=True, brief=False)

        out = capsys.readouterr().out
        assert "auditor_report (auth_error)" in out
        assert "consultant_review (auth_error)" in out

    @pytest.mark.parametrize(
        ("quiet", "brief"), [(True, False), (False, True), (True, True)]
    )
    def test_quiet_and_brief_do_not_suppress_it(self, capsys, quiet, brief):
        # The invariant that makes this worth having: verbosity is about noise,
        # not about hiding a run that produced no cross-check.
        from src.main import _print_degradation_notice

        _print_degradation_notice(self._degraded_result(), quiet=quiet, brief=brief)

        assert "DEGRADED" in capsys.readouterr().out

    def test_verbose_mode_explains_the_consequence(self, capsys):
        from src.main import _print_degradation_notice

        _print_degradation_notice(self._degraded_result(), quiet=False, brief=False)

        out = capsys.readouterr().out
        assert "Degraded run" in out
        assert "risk flags" in out

    def test_clean_run_prints_nothing(self, capsys):
        from src.main import _print_degradation_notice

        _print_degradation_notice(
            {"run_summary": {"optional_failures": []}}, quiet=False, brief=False
        )

        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        "result",
        [
            {},
            {"run_summary": None},
            {"run_summary": {}},
            {"run_summary": {"optional_failures": None}},
        ],
    )
    def test_absent_run_summary_is_silent_not_fatal(self, capsys, result):
        from src.main import _print_degradation_notice

        _print_degradation_notice(result, quiet=True, brief=False)

        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("statuses", ["garbage", None, {"consultant_review": None}])
    def test_malformed_statuses_degrade_to_unknown(self, capsys, statuses):
        # A diagnostic must never be able to break an otherwise successful run.
        from src.main import _print_degradation_notice

        _print_degradation_notice(
            {
                "run_summary": {"optional_failures": ["consultant_review"]},
                "artifact_statuses": statuses,
            },
            quiet=True,
            brief=False,
        )

        assert "consultant_review (unknown)" in capsys.readouterr().out
