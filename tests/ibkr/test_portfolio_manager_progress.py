"""Tests for portfolio-manager progress output while loading saved analyses."""

import json
import logging
from argparse import Namespace
from types import ModuleType
from unittest.mock import AsyncMock, patch

import pytest

from scripts.portfolio_manager import (
    _configure_external_loggers,
    _load_analyses_with_progress,
    _load_ibkr_context,
    _preflight_ibkr_requirements,
    main,
)
from src.ibkr.models import PortfolioSummary
from src.ibkr.recommendation_service import PortfolioRecommendationBundle
from src.ibkr.refresh_service import AnalysisFreshnessSummary, RefreshActivity


def _make_args(**overrides) -> Namespace:
    defaults = {
        "debug": False,
        "test_auth": False,
        "execute": False,
        "results_dir": "results/",
        "read_only": False,
        "recommend": False,
        "report_only": True,
        "refresh_stale": False,
        "refresh_policy": None,
        "refresh_limit": 10,
        "json": False,
        "output": "",
        "max_age": 14,
        "drift_pct": 15.0,
        "sector_limit": 30.0,
        "exchange_limit": 40.0,
        "watchlist_name": None,
        "quick": False,
        "cash_buffer": 0.05,
        "account_id": "",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_load_analyses_with_progress_emits_stderr_updates(tmp_path, capsys):
    """Loading analyses prints immediate and completion progress to stderr."""
    analysis_json = {
        "prediction_snapshot": {
            "ticker": "7203.T",
            "analysis_date": "2026-03-01",
            "verdict": "BUY",
        },
        "investment_analysis": {},
    }
    (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(analysis_json))

    analyses = _load_analyses_with_progress(tmp_path)
    captured = capsys.readouterr()

    assert "Loading analyses from" in captured.err
    assert "Found 1 analysis file" in captured.err
    assert "Progress: 1/1 files scanned; 1 latest analyses loaded" in captured.err
    assert f"Loaded 1 analyses from {tmp_path}/" in captured.err
    assert "7203.T" in analyses


def test_load_analyses_with_progress_reports_index_hit(tmp_path, capsys):
    """A reused latest-analyses index is surfaced explicitly to the user."""
    analysis_json = {
        "prediction_snapshot": {
            "ticker": "7203.T",
            "analysis_date": "2026-03-01",
            "verdict": "BUY",
        },
        "investment_analysis": {},
    }
    (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(analysis_json))

    _load_analyses_with_progress(tmp_path)
    capsys.readouterr()

    analyses = _load_analyses_with_progress(tmp_path)
    captured = capsys.readouterr()

    assert "Loaded 1 analyses from cache index" in captured.err
    assert "7203.T" in analyses


def test_load_analyses_with_progress_reports_index_reconstruction(tmp_path, capsys):
    """Corrupt latest-analyses index is reported before fallback reconstruction."""
    analysis_json = {
        "prediction_snapshot": {
            "ticker": "7203.T",
            "analysis_date": "2026-03-01",
            "verdict": "BUY",
        },
        "investment_analysis": {},
    }
    (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(analysis_json))

    _load_analyses_with_progress(tmp_path)
    from src.ibkr.reconciler import _analysis_index_path

    _analysis_index_path(tmp_path).write_text("{broken json")
    capsys.readouterr()

    analyses = _load_analyses_with_progress(tmp_path)
    captured = capsys.readouterr()

    assert (
        "is invalid (reason: unknown); reconstructing from analysis files"
        in captured.err
    )
    assert "Progress: 1/1 files scanned; 1 latest analyses loaded" in captured.err
    assert "7203.T" in analyses


def test_load_analyses_with_progress_reports_rebuild_reason(tmp_path, capsys):
    """Directory-state invalidation prints the explicit rebuild reason."""
    first = {
        "prediction_snapshot": {
            "ticker": "7203.T",
            "analysis_date": "2026-03-01",
            "verdict": "BUY",
        },
        "investment_analysis": {},
    }
    second = {
        "prediction_snapshot": {
            "ticker": "6758.T",
            "analysis_date": "2026-03-02",
            "verdict": "BUY",
        },
        "investment_analysis": {},
    }
    (tmp_path / "7203_T_2026-03-01_analysis.json").write_text(json.dumps(first))
    _load_analyses_with_progress(tmp_path)
    capsys.readouterr()

    (tmp_path / "6758_T_2026-03-02_analysis.json").write_text(json.dumps(second))

    analyses = _load_analyses_with_progress(tmp_path)
    captured = capsys.readouterr()

    assert "reason: stale_directory_state" in captured.err
    assert "reconstructing from analysis files" in captured.err
    assert "6758.T" in analyses


def test_load_ibkr_context_emits_phase_status_and_returns_live_state(capsys):
    """IBKR loading prints phase status before each external step."""

    class FakeClient:
        def __init__(self, config):
            self.config = config
            self.connected = False
            self.closed = False

        def connect(self, *, brokerage_session: bool) -> None:
            assert brokerage_session is False
            self.connected = True

        def get_live_orders(self, account_id: str | None = None) -> list[dict]:
            assert account_id == "U123456"
            return [{"ticker": "7203", "side": "BUY"}]

        def close(self) -> None:
            self.closed = True

    def fake_read_portfolio(client, account_id, cash_buffer):
        assert client.connected is True
        assert account_id == "U123456"
        assert cash_buffer == 0.05
        return ([], PortfolioSummary(portfolio_value_usd=1000, cash_balance_usd=50))

    def fake_read_watchlist(client, watchlist_name):
        assert client.connected is True
        assert watchlist_name == "watchlist-2026"
        return {"7203.T", "6758.T"}

    class FakeConfig:
        ibkr_account_id = "U123456"

        @staticmethod
        def get_oauth_access_token_secret() -> str:
            return "present"

    args = Namespace(
        account_id=None,
        cash_buffer=0.05,
        watchlist_name="watchlist-2026",
        recommend=True,
    )

    (
        positions,
        portfolio,
        watchlist_tickers,
        loaded_watchlist_name,
        loaded_watchlist_total,
        live_orders,
    ) = _load_ibkr_context(
        args,
        client_cls=FakeClient,
        read_portfolio_fn=fake_read_portfolio,
        read_watchlist_fn=fake_read_watchlist,
        config=FakeConfig(),
    )

    captured = capsys.readouterr()

    assert "Preparing IBKR client..." in captured.err
    assert "Connecting to IBKR..." in captured.err
    assert "Loading portfolio from IBKR..." in captured.err
    assert "Loading watchlist from IBKR..." in captured.err
    assert "Loading live orders from IBKR..." in captured.err
    assert "Loaded 2 watchlist tickers from 'watchlist-2026'" in captured.err
    assert positions == []
    assert portfolio.portfolio_value_usd == 1000
    assert watchlist_tickers == {"7203.T", "6758.T"}
    assert loaded_watchlist_name == "watchlist-2026"
    assert loaded_watchlist_total == 2
    assert live_orders == [{"ticker": "7203", "side": "BUY"}]


def test_preflight_ibkr_requirements_fails_when_ibind_missing(capsys):
    """Missing IBKR runtime should fail before any results scan begins."""
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.ibkr.client":
            raise ImportError("No module named 'ibind'")
        return original_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(SystemExit):
            _preflight_ibkr_requirements()

    captured = capsys.readouterr()
    assert "ibind not installed. Run: poetry install" in captured.err
    assert "Or use --read-only for offline mode." in captured.err


def test_main_non_read_only_preflight_failure_skips_analysis_load():
    """Normal mode should abort on IBKR preflight before scanning results."""
    args = _make_args(read_only=False)

    with (
        patch("scripts.portfolio_manager.parse_args", return_value=args),
        patch("scripts.portfolio_manager._configure_logging"),
        patch(
            "scripts.portfolio_manager._preflight_ibkr_requirements",
            side_effect=SystemExit(1),
        ),
        patch("scripts.portfolio_manager._load_analyses_with_progress") as mock_load,
    ):
        with pytest.raises(SystemExit):
            main()

    mock_load.assert_not_called()


def test_main_read_only_bypasses_ibkr_preflight_and_loads_analyses(capsys):
    """Read-only mode should still scan saved analyses without IBKR preflight."""
    args = _make_args(read_only=True)

    with (
        patch("scripts.portfolio_manager.parse_args", return_value=args),
        patch("scripts.portfolio_manager._configure_logging"),
        patch(
            "scripts.portfolio_manager._preflight_ibkr_requirements"
        ) as mock_preflight,
        patch(
            "scripts.portfolio_manager._load_analyses_with_progress",
            return_value={"7203.T": object()},
        ) as mock_load,
        patch("scripts.portfolio_manager.reconcile", return_value=[]),
        patch("scripts.portfolio_manager.compute_portfolio_health", return_value=[]),
        patch("scripts.portfolio_manager._store_macro_event_if_detected"),
        patch("scripts.portfolio_manager.format_report", return_value="report"),
    ):
        main()

    captured = capsys.readouterr()
    mock_preflight.assert_not_called()
    mock_load.assert_called_once()
    assert "Read-only mode: no IBKR connection" in captured.err
    assert captured.out.strip() == "report"


def test_main_test_auth_bypasses_analysis_loading():
    """Credential test mode should exit before touching saved analyses."""
    args = _make_args(test_auth=True)

    with (
        patch("scripts.portfolio_manager.parse_args", return_value=args),
        patch("scripts.portfolio_manager._configure_logging"),
        patch("scripts.portfolio_manager.cmd_test_auth") as mock_test_auth,
        patch("scripts.portfolio_manager._load_analyses_with_progress") as mock_load,
    ):
        main()

    mock_test_auth.assert_called_once_with(args)
    mock_load.assert_not_called()


def test_configure_logging_keeps_structlog_compatible_with_ibkr_modules():
    """Portfolio logging must keep imported IBKR structlog loggers on stdlib semantics."""
    from scripts.portfolio_manager import _configure_logging
    from src.ibkr.reconciler import logger as reconciler_logger

    _configure_logging(debug=False)

    reconciler_logger.info("analyses_loaded", count=1)


def test_preflight_ibkr_requirements_checks_config_before_scanning():
    """Preflight should validate required IBKR config without connecting."""
    fake_client_module = ModuleType("src.ibkr.client")
    fake_client_module.IbkrClient = object
    fake_config_module = ModuleType("src.ibkr_config")
    fake_config = object()
    fake_config_module.ibkr_config = fake_config
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.ibkr.client":
            return fake_client_module
        if name == "src.ibkr_config":
            return fake_config_module
        return original_import(name, globals, locals, fromlist, level)

    with (
        patch("builtins.__import__", side_effect=fake_import),
        patch(
            "scripts.portfolio_manager._check_config", side_effect=SystemExit(1)
        ) as mock_check,
    ):
        with pytest.raises(SystemExit):
            _preflight_ibkr_requirements()

    mock_check.assert_called_once_with(fake_config)


def test_configure_external_loggers_suppresses_transport_noise_in_normal_mode():
    """Normal mode keeps noisy third-party transport loggers off the console."""
    _configure_external_loggers(debug=False)

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("src.llms").level == logging.WARNING
    assert logging.getLogger("ibind.ibkr_client").level == logging.WARNING
    assert logging.getLogger("ibind_fh").level == logging.WARNING
    assert logging.getLogger("ibind_fh").propagate is False


def test_configure_external_loggers_restores_debug_visibility():
    """Debug mode re-enables noisy loggers for transport-level diagnosis."""
    _configure_external_loggers(debug=True)

    assert logging.getLogger("httpx").level == logging.DEBUG
    assert logging.getLogger("src.llms").level == logging.DEBUG
    assert logging.getLogger("ibind.ibkr_client").level == logging.DEBUG
    assert logging.getLogger("ibind_fh").level == logging.DEBUG
    assert logging.getLogger("ibind_fh").propagate is False


def _make_bundle(**overrides) -> PortfolioRecommendationBundle:
    values = {
        "portfolio": PortfolioSummary(portfolio_value_usd=1000),
        "watchlist_tickers": {"7203.T"},
        "watchlist_name": "watchlist-2026",
        "watchlist_total": 1,
        "watchlist_candidates_blocked_by_cash": 0,
        "live_orders": [{"ticker": "7203", "side": "BUY"}],
        "freshness_summary": AnalysisFreshnessSummary(),
        "refresh_activity": RefreshActivity(policy="off", limit=10),
    }
    values.update(overrides)
    return PortfolioRecommendationBundle(**values)


def test_main_recommend_mode_builds_request_with_blocking_refresh(capsys):
    """CLI recommend mode should delegate to the recommendation service with the resolved policy."""
    args = _make_args(
        recommend=True,
        report_only=False,
        refresh_limit=1,
        watchlist_name="watchlist-2026",
        cash_buffer=0.08,
        quick=True,
    )
    bundle = _make_bundle(
        refresh_activity=RefreshActivity(policy="blocking", limit=1),
    )
    mock_build = AsyncMock(return_value=bundle)

    with (
        patch("scripts.portfolio_manager.parse_args", return_value=args),
        patch("scripts.portfolio_manager._configure_logging"),
        patch("scripts.portfolio_manager._preflight_ibkr_requirements"),
        patch(
            "scripts.portfolio_manager.PortfolioRecommendationService.build_bundle",
            mock_build,
        ),
        patch("scripts.portfolio_manager._store_macro_event_if_detected"),
        patch("scripts.portfolio_manager.format_report", return_value="report"),
    ):
        main()

    captured = capsys.readouterr()
    request = mock_build.await_args.args[0]
    assert request.recommend is True
    assert request.read_only is False
    assert request.refresh_policy == "blocking"
    assert request.refresh_limit == 1
    assert request.watchlist_name == "watchlist-2026"
    assert request.cash_buffer == 0.08
    assert request.quick_mode is True
    assert callable(mock_build.await_args.kwargs["progress"])
    assert captured.out.strip() == "report"


def test_main_report_mode_uses_service_bundle_in_formatter():
    """CLI report mode should render the structured bundle returned by the service."""
    args = _make_args(recommend=False, report_only=True)
    bundle = _make_bundle(
        watchlist_tickers={"7203.T", "6758.T"},
        watchlist_total=2,
        refresh_activity=RefreshActivity(
            policy="off",
            limit=10,
            skipped_due_to_limit=["6758.T"],
        ),
    )
    mock_build = AsyncMock(return_value=bundle)

    with (
        patch("scripts.portfolio_manager.parse_args", return_value=args),
        patch("scripts.portfolio_manager._configure_logging"),
        patch("scripts.portfolio_manager._preflight_ibkr_requirements"),
        patch(
            "scripts.portfolio_manager.PortfolioRecommendationService.build_bundle",
            mock_build,
        ),
        patch("scripts.portfolio_manager._store_macro_event_if_detected"),
        patch(
            "scripts.portfolio_manager.format_report",
            return_value="report",
        ) as mock_report,
    ):
        main()

    mock_report.assert_called_once()
    assert mock_report.call_args.args[0] == bundle.items
    assert mock_report.call_args.args[1] == bundle.portfolio
    assert mock_report.call_args.kwargs["watchlist_name"] == "watchlist-2026"
    assert mock_report.call_args.kwargs["watchlist_total"] == 2
    assert mock_report.call_args.kwargs["watchlist_tickers"] == {"7203.T", "6758.T"}
    assert mock_report.call_args.kwargs["watchlist_candidates_blocked_by_cash"] == 0
    assert mock_report.call_args.kwargs["refresh_activity"].skipped_due_to_limit == [
        "6758.T"
    ]


def test_main_json_mode_uses_service_bundle_in_json_formatter(capsys):
    """JSON mode should serialize the bundle via format_json instead of format_report."""
    args = _make_args(json=True)
    bundle = _make_bundle()
    mock_build = AsyncMock(return_value=bundle)

    with (
        patch("scripts.portfolio_manager.parse_args", return_value=args),
        patch("scripts.portfolio_manager._configure_logging"),
        patch("scripts.portfolio_manager._preflight_ibkr_requirements"),
        patch(
            "scripts.portfolio_manager.PortfolioRecommendationService.build_bundle",
            mock_build,
        ),
        patch("scripts.portfolio_manager._store_macro_event_if_detected"),
        patch(
            "scripts.portfolio_manager.format_json", return_value='{"ok": true}'
        ) as mock_json,
        patch("scripts.portfolio_manager.format_report") as mock_report,
    ):
        main()

    captured = capsys.readouterr()
    mock_json.assert_called_once()
    mock_report.assert_not_called()
    assert captured.out.strip() == '{"ok": true}'
