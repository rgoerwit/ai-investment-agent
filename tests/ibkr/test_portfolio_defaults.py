"""Guard: every layer that defaults a portfolio knob reads it from one source.

These knobs (cash buffer, max analysis age, drift, sector/exchange limits,
refresh limit, over/underweight bands) were previously re-declared as literals
across the CLI argparse defaults, function signatures, ``IbkrSettings``, and the
dashboard settings — and drifted (cash buffer was 0.03 in some signatures while
the operative CLI/dashboard default was 0.05). ``src.ibkr.portfolio_defaults`` is
now the single source of truth; this test fails if any layer re-literalizes a
value instead of referencing the constant.
"""

from __future__ import annotations

import argparse
import inspect

from src.ibkr import portfolio_defaults as pd
from src.ibkr.cli_options import add_common_portfolio_request_args
from src.ibkr.portfolio import build_portfolio_summary, read_portfolio
from src.ibkr.reconciler import reconcile
from src.ibkr_config import IbkrSettings
from src.web.ibkr_dashboard.settings import DashboardSettings


def _cli_defaults() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_portfolio_request_args(
        parser,
        read_only_help="x",
        account_id_help="x",
        results_dir_help="x",
        watchlist_help="x",
    )
    return parser.parse_args([])


def test_ibkr_settings_default_from_constants():
    fields = IbkrSettings.model_fields
    assert fields["ibkr_cash_buffer_pct"].default == pd.DEFAULT_CASH_BUFFER_PCT
    assert fields["ibkr_max_analysis_age_days"].default == pd.DEFAULT_MAX_AGE_DAYS
    assert fields["ibkr_drift_threshold_pct"].default == pd.DEFAULT_DRIFT_PCT


def test_dashboard_settings_default_from_constants():
    fields = DashboardSettings.model_fields
    assert fields["cash_buffer"].default == pd.DEFAULT_CASH_BUFFER_PCT
    assert fields["max_age_days"].default == pd.DEFAULT_MAX_AGE_DAYS
    assert fields["drift_pct"].default == pd.DEFAULT_DRIFT_PCT
    assert fields["sector_limit_pct"].default == pd.DEFAULT_SECTOR_LIMIT_PCT
    assert fields["exchange_limit_pct"].default == pd.DEFAULT_EXCHANGE_LIMIT_PCT


def test_cli_defaults_from_constants():
    args = _cli_defaults()
    assert args.cash_buffer == pd.DEFAULT_CASH_BUFFER_PCT
    assert args.max_age == pd.DEFAULT_MAX_AGE_DAYS
    assert args.drift_pct == pd.DEFAULT_DRIFT_PCT
    assert args.sector_limit == pd.DEFAULT_SECTOR_LIMIT_PCT
    assert args.exchange_limit == pd.DEFAULT_EXCHANGE_LIMIT_PCT
    assert args.refresh_limit == pd.DEFAULT_REFRESH_LIMIT


def test_function_signature_defaults_from_constants():
    assert (
        inspect.signature(build_portfolio_summary).parameters["cash_buffer_pct"].default
        == pd.DEFAULT_CASH_BUFFER_PCT
    )
    assert (
        inspect.signature(read_portfolio).parameters["cash_buffer_pct"].default
        == pd.DEFAULT_CASH_BUFFER_PCT
    )
    rec = inspect.signature(reconcile).parameters
    assert rec["max_age_days"].default == pd.DEFAULT_MAX_AGE_DAYS
    assert rec["drift_threshold_pct"].default == pd.DEFAULT_DRIFT_PCT
    assert rec["sector_limit_pct"].default == pd.DEFAULT_SECTOR_LIMIT_PCT
    assert rec["exchange_limit_pct"].default == pd.DEFAULT_EXCHANGE_LIMIT_PCT
    assert rec["overweight_threshold_pct"].default == pd.DEFAULT_OVERWEIGHT_PCT
    assert rec["underweight_threshold_pct"].default == pd.DEFAULT_UNDERWEIGHT_PCT
