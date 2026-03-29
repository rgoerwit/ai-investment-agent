from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

from scripts.portfolio_manager import parse_args
from src.ibkr.cli_options import (
    add_common_portfolio_request_args,
    dashboard_preferences_overrides_from_args,
    dashboard_settings_overrides_from_args,
    portfolio_request_kwargs_from_args,
)


def test_portfolio_request_kwargs_from_args_normalizes_shared_fields():
    args = Namespace(
        results_dir="results/",
        account_id=" U20958465 ",
        watchlist_name=" default watchlist ",
        cash_buffer=0.05,
        max_age=21,
        drift_pct=12.5,
        sector_limit=28.0,
        exchange_limit=35.0,
        read_only=False,
        refresh_limit=7,
    )

    payload = portfolio_request_kwargs_from_args(args)

    assert payload == {
        "results_dir": Path("results/"),
        "account_id": "U20958465",
        "watchlist_name": "default watchlist",
        "cash_buffer": 0.05,
        "max_age_days": 21,
        "drift_pct": 12.5,
        "sector_limit_pct": 28.0,
        "exchange_limit_pct": 35.0,
        "read_only": False,
        "refresh_limit": 7,
    }


def test_dashboard_overrides_only_include_explicit_values():
    args = Namespace(
        results_dir=None,
        account_id=" U20958465 ",
        watchlist_name=None,
        cash_buffer=None,
        max_age=None,
        drift_pct=None,
        sector_limit=None,
        exchange_limit=None,
        read_only=None,
        refresh_limit=None,
    )

    settings_overrides = dashboard_settings_overrides_from_args(args)
    preferences_overrides = dashboard_preferences_overrides_from_args(args)

    assert settings_overrides == {"account_id": "U20958465"}
    assert preferences_overrides == {"account_id": "U20958465"}


def test_common_portfolio_arg_help_formats_with_percent_literals():
    parser = argparse.ArgumentParser()
    add_common_portfolio_request_args(parser)

    help_text = parser.format_help()

    assert "--drift-pct" in help_text
    assert "--sector-limit" in help_text
    assert "--exchange-limit" in help_text


def test_portfolio_manager_parse_args_supports_shared_portfolio_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "portfolio_manager.py",
            "--watchlist-name",
            "default watchlist",
            "--account-id",
            "U20958465",
            "--max-age",
            "21",
            "--refresh-limit",
            "5",
            "--read-only",
        ],
    )

    args = parse_args()

    assert args.watchlist_name == "default watchlist"
    assert args.account_id == "U20958465"
    assert args.max_age == 21
    assert args.refresh_limit == 5
    assert args.read_only is True
