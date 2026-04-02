from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal


def add_common_portfolio_request_args(
    parser: argparse.ArgumentParser,
    *,
    mode_flag_style: Literal["single", "dual"] = "single",
    results_dir_default: str | None = "results/",
    max_age_default: int | None = 14,
    cash_buffer_default: float | None = 0.05,
    drift_pct_default: float | None = 15.0,
    refresh_limit_default: int | None = 10,
    sector_limit_default: float | None = 30.0,
    exchange_limit_default: float | None = 40.0,
    read_only_default: bool | None = False,
    read_only_help: str | None = None,
    live_help: str | None = None,
    account_id_help: str | None = None,
    results_dir_help: str | None = None,
    watchlist_help: str | None = None,
) -> None:
    parser.add_argument(
        "--max-age",
        type=int,
        default=max_age_default,
        help=_with_default(
            "Max analysis age in days",
            max_age_default,
        ),
    )
    parser.add_argument(
        "--drift-pct",
        type=float,
        default=drift_pct_default,
        help=_with_default(
            "Price drift threshold %",
            drift_pct_default,
        ),
    )
    parser.add_argument(
        "--cash-buffer",
        type=float,
        default=cash_buffer_default,
        help=_with_default(
            "Cash reserve fraction",
            cash_buffer_default,
        ),
    )
    parser.add_argument(
        "--refresh-limit",
        type=int,
        default=refresh_limit_default,
        help=_with_default(
            "Maximum tickers to auto-refresh in one run",
            refresh_limit_default,
        ),
    )
    parser.add_argument(
        "--sector-limit",
        type=float,
        default=sector_limit_default,
        help=_with_default(
            "Warn when a BUY/ADD would push a sector above this %",
            sector_limit_default,
        ),
    )
    parser.add_argument(
        "--exchange-limit",
        type=float,
        default=exchange_limit_default,
        help=_with_default(
            "Warn when a BUY/ADD would push an exchange above this %",
            exchange_limit_default,
        ),
    )

    if mode_flag_style == "single":
        parser.add_argument(
            "--read-only",
            action="store_true",
            default=bool(read_only_default),
            help=read_only_help or "Never create IBKR connection (offline mode)",
        )
    elif mode_flag_style == "dual":
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            "--read-only",
            dest="read_only",
            action="store_true",
            help=read_only_help
            or "Use saved analysis results only and skip live IBKR portfolio data.",
        )
        mode_group.add_argument(
            "--live",
            dest="read_only",
            action="store_false",
            help=live_help
            or "Use live IBKR portfolio data for this dashboard session.",
        )
        parser.set_defaults(read_only=read_only_default)
    else:
        raise ValueError(f"Unknown mode_flag_style: {mode_flag_style}")

    parser.add_argument(
        "--account-id",
        type=str,
        default=None,
        help=account_id_help or "Override IBKR account ID",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=results_dir_default,
        help=results_dir_help
        or _with_default("Override results directory", results_dir_default),
    )
    parser.add_argument(
        "--watchlist-name",
        type=str,
        default=None,
        help=watchlist_help
        or (
            "Name of the IBKR watchlist to evaluate (case-insensitive substring match)."
        ),
    )


def validate_common_portfolio_request_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    refresh_limit = getattr(args, "refresh_limit", None)
    if refresh_limit is not None and refresh_limit < 1:
        parser.error("--refresh-limit must be >= 1")


def portfolio_request_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "results_dir": Path(_required_arg(args, "results_dir")),
        "account_id": _optional_str(getattr(args, "account_id", None)),
        "watchlist_name": _optional_str(getattr(args, "watchlist_name", None)),
        "cash_buffer": _required_arg(args, "cash_buffer"),
        "max_age_days": _required_arg(args, "max_age"),
        "drift_pct": _required_arg(args, "drift_pct"),
        "sector_limit_pct": _required_arg(args, "sector_limit"),
        "exchange_limit_pct": _required_arg(args, "exchange_limit"),
        "read_only": bool(getattr(args, "read_only", False)),
        "refresh_limit": _required_arg(args, "refresh_limit"),
    }


def dashboard_settings_overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    _copy_if_present(overrides, "results_dir", getattr(args, "results_dir", None), Path)
    _copy_if_present(overrides, "cash_buffer", getattr(args, "cash_buffer", None))
    _copy_if_present(overrides, "max_age_days", getattr(args, "max_age", None))
    _copy_if_present(overrides, "drift_pct", getattr(args, "drift_pct", None))
    _copy_if_present(
        overrides,
        "sector_limit_pct",
        getattr(args, "sector_limit", None),
    )
    _copy_if_present(
        overrides,
        "exchange_limit_pct",
        getattr(args, "exchange_limit", None),
    )
    _copy_if_present(
        overrides,
        "default_refresh_limit",
        getattr(args, "refresh_limit", None),
    )
    if hasattr(args, "account_id") and args.account_id is not None:
        overrides["account_id"] = _optional_str(args.account_id)
    if hasattr(args, "watchlist_name") and args.watchlist_name is not None:
        overrides["watchlist_name"] = _optional_str(args.watchlist_name)
    if hasattr(args, "read_only") and args.read_only is not None:
        overrides["read_only"] = args.read_only
    return overrides


def dashboard_preferences_overrides_from_args(
    args: argparse.Namespace,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if hasattr(args, "account_id") and args.account_id is not None:
        overrides["account_id"] = _optional_str(args.account_id)
    if hasattr(args, "watchlist_name") and args.watchlist_name is not None:
        overrides["watchlist_name"] = _optional_str(args.watchlist_name)
    if hasattr(args, "read_only") and args.read_only is not None:
        overrides["read_only"] = args.read_only
    _copy_if_present(overrides, "max_age_days", getattr(args, "max_age", None))
    _copy_if_present(overrides, "refresh_limit", getattr(args, "refresh_limit", None))
    return overrides


def _with_default(label: str, default: Any) -> str:
    label = label.replace("%", "%%")
    if default is None:
        return label
    return f"{label} (default: {default})"


def _required_arg(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name, None)
    if value is None:
        raise ValueError(f"{name} is required for portfolio request construction")
    return value


def _optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _copy_if_present(
    payload: dict[str, Any],
    key: str,
    value: Any,
    transform: Any | None = None,
) -> None:
    if value is None:
        return
    payload[key] = transform(value) if transform else value
