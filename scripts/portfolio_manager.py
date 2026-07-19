#!/usr/bin/env python3
"""
IBKR Portfolio Reconciliation Tool

Compares live IBKR positions against the equity evaluator's latest
analysis recommendations. Produces position-aware BUY/SELL/HOLD/TRIM/REVIEW
actions that account for existing holdings.

Usage (manual repo mode: use Poetry or an activated venv):
    poetry run python scripts/portfolio_manager.py --test-auth          # Verify IBKR credentials work
    poetry run python scripts/portfolio_manager.py                      # Report only
    poetry run python scripts/portfolio_manager.py --recommend          # + order suggestions
    poetry run python scripts/portfolio_manager.py --execute            # + place orders (with confirmation)
    poetry run python scripts/portfolio_manager.py --read-only          # No IBKR connection (offline)

    # Or activate once: source .venv/bin/activate
    # Then plain `python scripts/portfolio_manager.py` works for the session.

    # Local container mode (inside the runtime image, Poetry is not installed):
    python scripts/portfolio_manager.py --read-only

Requires: project dependencies installed (manual repo mode typically uses `poetry install`)
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.error_safety import summarize_exception
from src.ibkr.cli_options import (
    add_common_portfolio_request_args,
    portfolio_request_kwargs_from_args,
    validate_common_portfolio_request_args,
)
from src.ibkr.dip_watch import (
    score_dip_watch_item,
)
from src.ibkr.exceptions import IBKRAuthError, IBKRError
from src.ibkr.models import (
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
)
from src.ibkr.portfolio_action_plan import (
    PortfolioActionPlan,
    build_action_plan_counts,
    build_portfolio_action_plan,
    has_active_macro_event,
)
from src.ibkr.portfolio_defaults import (
    DEFAULT_EXCHANGE_LIMIT_PCT,
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_SECTOR_LIMIT_PCT,
)
from src.ibkr.portfolio_presentation import (
    build_cash_summary,
    retail_safe_action,
)
from src.ibkr.portfolio_report import (
    PortfolioReportContext,
)
from src.ibkr.portfolio_report_execution import (
    render_cash_execution_and_summary_sections,
)
from src.ibkr.portfolio_report_formatting import (
    bar_chart as report_bar_chart,
)
from src.ibkr.portfolio_report_formatting import (
    currency_prefix,
)
from src.ibkr.portfolio_report_formatting import (
    display_ticker as report_display_ticker,
)
from src.ibkr.portfolio_report_formatting import (
    item_currency as report_item_currency,
)
from src.ibkr.portfolio_report_formatting import (
    urgency_prefix as report_urgency_prefix,
)
from src.ibkr.portfolio_report_positions import render_position_and_risk_sections
from src.ibkr.portfolio_report_status import render_header_and_status_sections
from src.ibkr.refresh_service import (
    AnalysisFreshnessSummary,
    AnalysisRefreshService,
    RefreshActivity,
    RefreshPolicy,
)
from src.ibkr.screening_freshness import ScreeningFreshnessSummary
from src.sector_normalization import (
    aggregate_sector_weights as shared_aggregate_sector_weights,
)
from src.sector_normalization import (
    normalize_sector_label as shared_normalize_sector_label,
)

if TYPE_CHECKING:
    from src.ibkr.analysis_index import AnalysisLoadProgress
    from src.memory import MacroEvent

_IBKR_OAUTH_PORTAL = (
    "https://ndcdyn.interactivebrokers.com/sso/Login?action=OAUTH&RL=1&ip2loc=US"
)

_refresh_service = AnalysisRefreshService()


def search_tavily_sync_inspected(*args: Any, **kwargs: Any) -> Any:
    from src.tavily_utils import (
        search_tavily_sync_inspected as _search_tavily_sync_inspected,
    )

    return _search_tavily_sync_inspected(*args, **kwargs)


def IbkrAccountService(*args: Any, **kwargs: Any) -> Any:
    from src.ibkr.account_service import IbkrAccountService as _IbkrAccountService

    return _IbkrAccountService(*args, **kwargs)


def load_latest_analyses(*args: Any, **kwargs: Any) -> Any:
    from src.ibkr.analysis_index import load_latest_analyses as _load_latest_analyses

    return _load_latest_analyses(*args, **kwargs)


def IbkrPortfolioDataService(*args: Any, **kwargs: Any) -> Any:
    from src.ibkr.portfolio_data_service import (
        IbkrPortfolioDataService as _IbkrPortfolioDataService,
    )

    return _IbkrPortfolioDataService(*args, **kwargs)


def PortfolioRecommendationRequest(*args: Any, **kwargs: Any) -> Any:
    from src.ibkr.recommendation_service import (
        PortfolioRecommendationRequest as _PortfolioRecommendationRequest,
    )

    return _PortfolioRecommendationRequest(*args, **kwargs)


class PortfolioRecommendationService:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        from src.ibkr.recommendation_service import (
            PortfolioRecommendationService as _PortfolioRecommendationService,
        )

        self._inner = _PortfolioRecommendationService(*args, **kwargs)

    async def build_bundle(self, *args: Any, **kwargs: Any) -> Any:
        return await self._inner.build_bundle(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def reconcile(*args: Any, **kwargs: Any) -> Any:
    from src.ibkr.reconciler import reconcile as _reconcile

    return _reconcile(*args, **kwargs)


def compute_portfolio_health(*args: Any, **kwargs: Any) -> Any:
    from src.ibkr.portfolio_health import (
        compute_portfolio_health as _compute_portfolio_health,
    )

    return _compute_portfolio_health(*args, **kwargs)


def _python_command_prefix() -> str:
    """Return the user-facing Python launcher for the current runtime."""
    if (
        os.getenv("INVESTMENT_AGENT_CONTAINER")
        or Path("/.dockerenv").exists()
        or Path("/run/.containerenv").exists()
    ):
        return "python"
    return "poetry run python"


def _analysis_command(ticker: str) -> str:
    return f"{_python_command_prefix()} -m src.main --ticker {ticker}"


def _portfolio_manager_command(*args: str) -> str:
    suffix = " ".join(args).strip()
    base = f"{_python_command_prefix()} scripts/portfolio_manager.py"
    return f"{base} {suffix}".strip()


def _portfolio_manager_recommend_command(*, watchlist_name: str | None = None) -> str:
    args = ["--recommend"]
    if watchlist_name:
        args.extend(["--watchlist-name", f'"{watchlist_name}"'])
    return _portfolio_manager_command(*args)


def _prompt_for_missing_secret(config) -> None:
    """Prompt for OAuth token secret if absent. Held in memory only — never written to disk."""
    if not config.get_oauth_access_token_secret():
        from pydantic import SecretStr

        print(
            f"\nIBKR OAuth Access Token Secret not set.\n"
            f"The secret is shown ONCE when you generate the token at the IBKR portal:\n"
            f"  {_IBKR_OAUTH_PORTAL}\n"
            f"Copy it immediately and save it as IBKR_OAUTH_ACCESS_TOKEN_SECRET in your .env file.\n"
            f"It does NOT expire — only the 24-hour brokerage session does (needed for --execute).\n"
            f"For read-only portfolio reconciliation, the access token is all you need.",  # lgtm[py/clear-text-logging-sensitive-data]
            file=sys.stderr,
        )
        secret = getpass.getpass("Access Token Secret: ")
        if not secret:
            print("No secret provided. Cannot connect to IBKR.", file=sys.stderr)
            sys.exit(1)
        config.ibkr_oauth_access_token_secret = SecretStr(secret)


# Required credentials checked before any IBKR connection attempt.
# Each tuple is (ENV_VAR_NAME, getter_callable).
# IBKR_OAUTH_ACCESS_TOKEN_SECRET is excluded — handled by _prompt_for_missing_secret.
# IBKR_OAUTH_DH_PRIME is required by ibind (no built-in default).
_REQUIRED_CREDENTIALS: list[tuple[str, Callable[[Any], object]]] = [
    ("IBKR_ACCOUNT_ID", lambda c: c.ibkr_account_id),
    ("IBKR_OAUTH_CONSUMER_KEY", lambda c: c.get_oauth_consumer_key()),
    ("IBKR_OAUTH_ACCESS_TOKEN", lambda c: c.get_oauth_access_token()),
    ("IBKR_OAUTH_ENCRYPTION_KEY_FP", lambda c: c.ibkr_oauth_encryption_key_fp),
    ("IBKR_OAUTH_SIGNATURE_KEY_FP", lambda c: c.ibkr_oauth_signature_key_fp),
    (
        "IBKR_OAUTH_DH_PRIME (or _FP)",
        lambda c: c.ibkr_oauth_dh_prime or c.ibkr_oauth_dh_prime_fp,
    ),
]


def _validate_key_files(config) -> dict[str, str]:
    """
    Load and locally test the RSA signing and encryption key files.

    Runs a sign→verify round-trip on the signature key and an
    encrypt→decrypt round-trip on the encryption key.  Both tests
    are purely local — no network calls, no writes, no side effects.

    Returns a dict with human-readable key info on success
    (e.g. {"signature_key": "2048-bit RSA", "encryption_key": "2048-bit RSA"}).
    Prints errors to stderr and exits on any failure.
    """
    from cryptography.hazmat.primitives.asymmetric.padding import PKCS1v15
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    errors: list[str] = []

    # Flag same-file misconfiguration before touching either key.
    sig_fp = Path(config.ibkr_oauth_signature_key_fp)
    enc_fp = Path(config.ibkr_oauth_encryption_key_fp)
    if sig_fp.exists() and enc_fp.exists() and sig_fp.resolve() == enc_fp.resolve():
        errors.append(
            "IBKR_OAUTH_SIGNATURE_KEY_FP and IBKR_OAUTH_ENCRYPTION_KEY_FP both point "
            "to the same file. IBKR requires two separate RSA private key files:\n"  # gitleaks:allow
            "  - Signature key:   your RSA signing key in PEM format (PKCS8)\n"
            "  - Encryption key:  your RSA encryption key in PEM format (separate key pair)\n"
            "  Upload the matching public keys to the IBKR portal — not the PEM files here."
        )

    # PEM headers that indicate a public key or certificate (not a private key).
    _PUBLIC_HEADERS = (
        b"-----BEGIN PUBLIC KEY-----",  # gitleaks:allow
        b"-----BEGIN RSA PUBLIC KEY-----",  # gitleaks:allow
        b"-----BEGIN CERTIFICATE-----",
        b"-----BEGIN CERTIFICATE REQUEST-----",
    )

    def _load_rsa(fp: str, label: str):
        """Try to load an RSA private key from a PEM file; record any error."""
        path = Path(fp)
        if not path.exists():
            errors.append(f"{label}: file not found: {fp}")
            return None
        if not path.is_file():
            errors.append(f"{label}: path is not a regular file: {fp}")
            return None
        try:
            data = path.read_bytes()
        except OSError as exc:
            errors.append(f"{label}: cannot read file: {exc}")
            return None

        # Detect public-key / certificate headers before trying to load.
        first_line = data.lstrip().split(b"\n", 1)[0].strip()
        if any(first_line.startswith(h) for h in _PUBLIC_HEADERS):
            errors.append(
                f"{label}: {fp}\n"
                f"  contains a public key or certificate (header: {first_line.decode()})\n"
                f"  but IBKR requires the *private* key here — the one you kept locally.\n"
                f"  (The public key is what you uploaded to the IBKR portal.)\n"
                f"  Expected: RSA or PKCS8 PEM private key format"
            )
            return None

        try:
            key = load_pem_private_key(data, password=None)
        except Exception as exc:
            errors.append(f"{label}: not a valid PEM private key ({fp}): {exc}")
            return None
        if not isinstance(key, RSAPrivateKey):
            errors.append(f"{label}: key type is not RSA ({fp})")
            return None
        return key

    info: dict[str, str] = {}

    # --- Signature key: sign → verify round-trip ---
    sig_key = _load_rsa(config.ibkr_oauth_signature_key_fp, "Signature key")
    if sig_key is not None:
        try:
            payload = b"ibkr-auth-selftest-sign"
            sig = sig_key.sign(payload, PKCS1v15(), SHA256())
            sig_key.public_key().verify(sig, payload, PKCS1v15(), SHA256())
            info["signature_key"] = f"{sig_key.key_size}-bit RSA (sign/verify passed)"
        except Exception as exc:
            errors.append(f"Signature key: sign/verify self-test failed: {exc}")

    # --- Encryption key: encrypt → decrypt round-trip ---
    enc_key = _load_rsa(config.ibkr_oauth_encryption_key_fp, "Encryption key")
    if enc_key is not None:
        try:
            plaintext = b"ibkr-auth-selftest-enc"
            ciphertext = enc_key.public_key().encrypt(plaintext, PKCS1v15())
            recovered = enc_key.decrypt(ciphertext, PKCS1v15())
            if recovered != plaintext:
                raise ValueError("Decrypted value does not match original")
            info["encryption_key"] = (
                f"{enc_key.key_size}-bit RSA (encrypt/decrypt passed)"
            )
        except Exception as exc:
            errors.append(f"Encryption key: encrypt/decrypt self-test failed: {exc}")

    if errors:
        print("\nKey file validation failed:", file=sys.stderr)
        for err in errors:
            print(
                f"  {err}", file=sys.stderr
            )  # lgtm[py/clear-text-logging-sensitive-data]
        sys.exit(1)

    return info


def _check_config(config) -> None:
    """
    Validate that all required IBKR credentials are present.

    Prints a clear list of missing environment variable names and exits
    if anything is absent. IBKR_OAUTH_ACCESS_TOKEN_SECRET is not checked
    here; use _prompt_for_missing_secret() for that field.
    """
    missing = [var for var, getter in _REQUIRED_CREDENTIALS if not getter(config)]
    if missing:
        print("Missing required IBKR credentials:", file=sys.stderr)
        for var in missing:
            print(f"  {var}", file=sys.stderr)
        print(
            "\nSet these in your .env file or as environment variables, then retry.",
            file=sys.stderr,
        )
        sys.exit(1)


def _preflight_ibkr_requirements() -> None:
    """Fail fast on missing IBKR runtime/config before scanning analysis files."""
    try:
        from src.ibkr.client import IbkrClient  # noqa: F401
        from src.ibkr_config import ibkr_config
    except ImportError:
        print(
            "ibind not installed. Run: poetry install\n"
            "Or use --read-only for offline mode.",
            file=sys.stderr,
        )
        sys.exit(1)

    _check_config(ibkr_config)


def parse_args(
    argv: list[str] | None = None,
    *,
    analyzer_config: Any | None = None,
    ibkr_settings: Any | None = None,
) -> argparse.Namespace:
    from src.config import config as default_analyzer_config
    from src.ibkr_config import ibkr_config as default_ibkr_settings

    analyzer_config = analyzer_config or default_analyzer_config
    ibkr_settings = ibkr_settings or default_ibkr_settings

    parser = argparse.ArgumentParser(
        description="Reconcile IBKR portfolio against evaluator recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--report-only",
        action="store_true",
        help="Show reconciliation report (default)",
    )
    mode_group.add_argument(
        "--recommend",
        action="store_true",
        help="Report + concrete order suggestions with sizes/prices",
    )
    mode_group.add_argument(
        "--execute",
        action="store_true",
        help="[DISABLED] Order execution coming soon — use --recommend for now",
    )
    mode_group.add_argument(
        "--test-auth",
        action="store_true",
        help="Verify IBKR credentials and connection, then exit",
    )

    # Options. Sector/exchange/refresh defaults come from src.ibkr.portfolio_defaults
    # via add_common_portfolio_request_args. The three knobs that ALSO have an env
    # override (IbkrSettings) are sourced from ibkr_config here so that
    # IBKR_CASH_BUFFER_PCT / IBKR_MAX_ANALYSIS_AGE_DAYS / IBKR_DRIFT_THRESHOLD_PCT
    # actually take effect, with the matching CLI flag still overriding the env value.
    add_common_portfolio_request_args(
        parser,
        mode_flag_style="single",
        max_age_default=ibkr_settings.ibkr_max_analysis_age_days,
        cash_buffer_default=ibkr_settings.ibkr_cash_buffer_pct,
        drift_pct_default=ibkr_settings.ibkr_drift_threshold_pct,
        # Follow the analyzer's RESULTS_DIR so portfolio_manager reads from the same
        # directory the analyzer writes analyses to (the dashboard keeps its own
        # IBKR_DASHBOARD_RESULTS_DIR surface for process isolation).
        results_dir_default=str(analyzer_config.results_dir),
        read_only_default=False,
        read_only_help="Never create IBKR connection (offline mode)",
        account_id_help="Override IBKR account ID",
        results_dir_help="Override results directory (default: RESULTS_DIR or ./results)",
        watchlist_help=(
            "Name of the IBKR watchlist to evaluate "
            "(case-insensitive substring match). "
            'If omitted, tries "default watchlist" and silently skips if not found. '
            "If explicitly provided and not found, aborts."
        ),
    )
    parser.add_argument(
        "--refresh-stale",
        action="store_true",
        help="Backward-compatible alias for --refresh-policy blocking",
    )
    parser.add_argument(
        "--refresh-policy",
        choices=("off", "blocking", "proactive"),
        default=None,
        help=(
            "Auto-refresh policy: off (default for reports), blocking "
            "(default for --recommend), or proactive (also refresh due-soon holds)"
        ),
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use quick mode for re-analysis"
    )
    parser.add_argument(
        "--output", type=str, default="", help="Write report to file (default: stdout)"
    )
    parser.add_argument("--json", action="store_true", help="Structured JSON output")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args(argv)

    # --recommend and --execute override --report-only
    if args.recommend or args.execute:
        args.report_only = False

    validate_common_portfolio_request_args(parser, args)

    return args


def _resolve_refresh_policy(args: argparse.Namespace) -> RefreshPolicy:
    """Resolve refresh policy from explicit flags and execution mode."""
    return _refresh_service.resolve_policy(
        explicit_policy=getattr(args, "refresh_policy", None),
        refresh_stale=getattr(args, "refresh_stale", False),
        recommend=getattr(args, "recommend", False),
        read_only=getattr(args, "read_only", False),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Macro Event Detection
# ══════════════════════════════════════════════════════════════════════════════

import re as _re  # noqa: E402 — used by _store_macro_event_if_detected

import structlog as _structlog

logger = _structlog.get_logger(__name__)

# Keyword → (event_type, impact, opportunity_prior)
# First match wins (checked in order); UNKNOWN is the fallback.
_EVENT_TYPE_RULES: list[tuple[frozenset, str, str, str]] = [
    (
        frozenset(
            [
                "tariff",
                "trade war",
                "import duty",
                "export ban",
                "trade spat",
                "customs duty",
                "trade deal",
                "trade negotiation",
                "trade tension",
                "trade restriction",
                "section 301",
                "section 232",
            ]
        ),
        "TARIFF_TRADE",
        "TRANSIENT",
        "HIGH",
    ),
    (
        frozenset(
            [
                "margin call",
                "forced selling",
                "redemption",
                "deleveraging",
                "liquidity crunch",
                "market maker",
                "flash crash",
                "circuit breaker",
                "panic selling",
                "fire sale",
                "repo stress",
                "collateral call",
            ]
        ),
        "LIQUIDITY_PANIC",
        "TRANSIENT",
        "HIGH",
    ),
    (
        frozenset(
            [
                "contagion",
                "spillover",
                "sentiment contagion",
                "risk-off",
                "flight to safety",
                "flight to quality",
                "sell everything",
                "risk aversion",
                "global risk-off",
            ]
        ),
        "CONTAGION_SPREAD",
        "TRANSIENT",
        "MEDIUM",
    ),
    (
        frozenset(
            [
                "election",
                "election result",
                "cabinet",
                "government collapse",
                "political uncertainty",
                "referendum",
                "coup",
                "prime minister",
                "president",
                "parliament dissolved",
                "snap election",
                "policy statement",
            ]
        ),
        "POLITICAL_EVENT",
        "TRANSIENT",
        "MEDIUM",
    ),
    (
        frozenset(
            [
                "rate hike",
                "rate cut",
                "interest rate",
                "federal reserve",
                "fed",
                "ecb",
                "boj",
                "bank of japan",
                "yield curve control",
                "ycc",
                "quantitative tightening",
                "qt",
                "quantitative easing",
                "qe",
                "monetary policy",
                "rate decision",
                "central bank",
                "hawkish",
                "dovish",
                "inflation target",
            ]
        ),
        "MONETARY_PIVOT",
        "MEDIUM",
        "LOW",
    ),
    (
        frozenset(
            [
                "oil price",
                "crude oil",
                "brent",
                "commodity",
                "metal price",
                "copper",
                "iron ore",
                "supply disruption",
                "opec",
                "oil production",
                "energy crisis",
                "resource nationalism",
                "natural gas",
                "lng",
                "wheat",
                "food prices",
                "supply chain",
            ]
        ),
        "COMMODITY_SHOCK",
        "MEDIUM",
        "LOW",
    ),
    (
        frozenset(
            [
                "war",
                "conflict",
                "military",
                "invasion",
                "sanctions",
                "taiwan",
                "ukraine",
                "russia",
                "north korea",
                "missile",
                "geopolitical",
                "territorial dispute",
                "strait",
                "blockade",
                "nato",
                "arms",
                "escalation",
                "ceasefire",
            ]
        ),
        "GEOPOLITICAL",
        "STRUCTURAL",
        "NEGATIVE",
    ),
    (
        frozenset(
            [
                "regulation",
                "ban",
                "law",
                "legislation",
                "overhaul",
                "reform",
                "compliance",
                "regulator",
                "antitrust",
                "competition law",
                "data protection",
                "gdpr",
                "pharmaceutical regulation",
                "drug approval",
                "fda",
                "cfius",
                "foreign investment review",
                "carbon tax",
                "environmental regulation",
                "decoupling",
                "framework",
                "prohibited",
                "compulsory",
                "mandate",
            ]
        ),
        "REGULATORY_SHIFT",
        "STRUCTURAL",
        "NEGATIVE",
    ),
    (
        frozenset(
            [
                "bank failure",
                "bank run",
                "credit crunch",
                "sovereign default",
                "debt crisis",
                "credit rating downgrade",
                "svb",
                "lehman",
                "systemic risk",
                "financial crisis",
                "bailout",
                "insolvency",
                "refinancing risk",
                "credit spread",
                "bond yield spike",
                "debt ceiling",
                "high yield stress",
            ]
        ),
        "CREDIT_CONTAGION",
        "STRUCTURAL",
        "NEGATIVE",
    ),
    (
        frozenset(
            [
                "recession",
                "gdp contraction",
                "economic slowdown",
                "stagflation",
                "earnings downturn",
                "profit warning",
                "guidance cut",
                "downgrade",
                "unemployment",
                "layoffs",
                "industrial output",
                "pmi contraction",
                "consumer confidence",
                "deflation",
                "debt deflation",
            ]
        ),
        "MACRO_RECESSION",
        "STRUCTURAL",
        "NEGATIVE",
    ),
    (
        frozenset(
            [
                "pandemic",
                "epidemic",
                "covid",
                "lockdown",
                "earthquake",
                "tsunami",
                "hurricane",
                "natural disaster",
                "cyber attack",
                "infrastructure attack",
                "black swan",
                "force majeure",
                "act of god",
                "biosecurity",
            ]
        ),
        "EXOGENOUS_SHOCK",
        "UNCERTAIN",
        "UNCERTAIN",
    ),
]

# Per-event-type expiry window in days.
_EXPIRY_DAYS: dict[str, int] = {
    "TARIFF_TRADE": 28,
    "LIQUIDITY_PANIC": 14,
    "CONTAGION_SPREAD": 21,
    "POLITICAL_EVENT": 30,
    "MONETARY_PIVOT": 90,
    "COMMODITY_SHOCK": 90,
    "GEOPOLITICAL": 180,
    "REGULATORY_SHIFT": 180,
    "CREDIT_CONTAGION": 120,
    "MACRO_RECESSION": 180,
    "EXOGENOUS_SHOCK": 60,
    "UNKNOWN": 60,
}


def _characterize_macro_event(
    event_date: str,
    sell_items: list,
    correlation_pct: float,
    peak_count: int = 0,
) -> tuple[str, str, str, str, str, str, str]:
    """
    Derive event metadata from sell items + Tavily news + Gemini Flash classification.

    Returns: (scope, primary_region, primary_sector, impact, event_type, headline, detail)
    Fails gracefully — returns ("GLOBAL", "GLOBAL", "", "UNCERTAIN", "UNKNOWN", "unknown", "")
    """
    from collections import Counter

    region_counts: Counter = Counter()
    sector_counts: Counter = Counter()
    for item in sell_items:
        _yf = item.ticker.yf
        dot = _yf.rfind(".")
        region_counts[_yf[dot:] if dot >= 0 else ".US"] += 1
        if item.analysis:
            s = (getattr(item.analysis, "sector", None) or "").strip()
            if s:
                sector_counts[s] += 1

    total = len(sell_items) or 1
    top_region, top_region_n = (
        region_counts.most_common(1)[0] if region_counts else (".US", 0)
    )
    top_sector, top_sector_n = (
        sector_counts.most_common(1)[0] if sector_counts else ("", 0)
    )
    top_region_pct = top_region_n / total
    top_sector_pct = top_sector_n / total

    scope = (
        "SECTOR"
        if top_sector_pct >= 0.60 and top_region_pct < 0.60
        else "REGIONAL"
        if top_region_pct >= 0.60
        else "GLOBAL"
    )
    primary_region = top_region if scope == "REGIONAL" else "GLOBAL"
    primary_sector = top_sector if scope == "SECTOR" else ""

    # News search
    headline, detail = "unknown", ""
    try:
        from src.macro_regions import display_region_for_suffix

        region_hint = (
            display_region_for_suffix(top_region) if scope == "REGIONAL" else ""
        )
        query = f"stock market shock {event_date} {region_hint} cause reason".strip()
        result = search_tavily_sync_inspected(query, profile="news_basic")
        if isinstance(result, dict):
            top = (result.get("results", [{}]) if isinstance(result, dict) else [{}])[0]
            headline = (top.get("title") or "")[:120] or "unknown"
            raw_detail = top.get("content") or ""
            sentences = [s.strip() for s in raw_detail.split(".") if s.strip()]
            detail = ". ".join(sentences[:2]) + ("." if sentences else "")
    except Exception as e:
        logger.warning("macro_news_search_failed", error=str(e))

    # Primary: Deep LLM classification (reads DEEP_MODEL from env)
    impact, event_type = "UNCERTAIN", "UNKNOWN"
    if headline != "unknown":
        try:
            import json as _json

            from langchain_core.messages import HumanMessage as _HM

            from src.agents import extract_string_content
            from src.llms import create_deep_thinking_llm

            _llm = create_deep_thinking_llm()
            _valid_types = (
                "TARIFF_TRADE|LIQUIDITY_PANIC|CONTAGION_SPREAD|POLITICAL_EVENT|"
                "MONETARY_PIVOT|COMMODITY_SHOCK|GEOPOLITICAL|REGULATORY_SHIFT|"
                "CREDIT_CONTAGION|MACRO_RECESSION|EXOGENOUS_SHOCK|UNKNOWN"
            )
            _prompt = (
                f"A correlated sell-off across {peak_count} portfolio positions occurred "
                f"on {event_date}. Scope: {scope} ({primary_region or 'mixed regions'}).\n\n"
                f"Headline: {headline}\nDetail: {detail}\n\n"
                f"Classify the macro event. Respond with JSON only, no explanation outside JSON:\n"
                f'{{"event_type": "{_valid_types}", '
                f'"impact": "TRANSIENT|MEDIUM|STRUCTURAL|UNCERTAIN", '
                f'"opportunity_prior": "HIGH|MEDIUM|LOW|NEGATIVE|UNCERTAIN", '
                f'"reasoning": "one sentence max"}}'
            )
            _resp = _llm.invoke([_HM(content=_prompt)])
            _text = extract_string_content(
                _resp.content if hasattr(_resp, "content") else str(_resp)
            )
            _text = _text.strip().strip("`").lstrip("json").strip()
            _classified = _json.loads(_text)
            event_type = _classified.get("event_type", "UNKNOWN")
            impact = _classified.get("impact", "UNCERTAIN")
            logger.info(
                "macro_event_llm_classified",
                event_type=event_type,
                impact=impact,
                reasoning=_classified.get("reasoning", ""),
            )
        except Exception as e:
            logger.warning("macro_event_llm_classification_failed", error=str(e))
            # Fallback: keyword rules
            text = (headline + " " + detail).lower()
            for keywords, etype, eimp, _opp in _EVENT_TYPE_RULES:
                if any(kw in text for kw in keywords):
                    event_type = etype
                    impact = eimp
                    break

    return scope, primary_region, primary_sector, impact, event_type, headline, detail


def _store_macro_event_if_detected(
    health_flags: list[str],
    reconciliation_items: list,
) -> MacroEvent | None:
    """Parse CORRELATED_SELL_EVENT, characterize it, store in ChromaDB. Fail-safe.

    Returns the characterized event (or None when no event was detected). The
    event is returned even when ChromaDB storage is unavailable or fails, so the
    caller's alert banner reflects the current detection regardless of Chroma.
    """
    from src.ibkr.portfolio_health import is_macro_event_evidence
    from src.memory import MacroEvent, create_macro_events_store

    correlated_flag = next(
        (f for f in health_flags if "CORRELATED_SELL_EVENT" in f), None
    )
    if not correlated_flag:
        return None

    # Tolerant of the three trigger phrasings: "within Nd of DATE" (window)
    # and "as of DATE" (cumulative / drawdown_breadth).
    m = _re.search(
        r"CORRELATED_SELL_EVENT:\s*(\d+) positions"
        r".*?(?:within (\d+)d of|as of) (\d{4}-\d{2}-\d{2})"
        r".*?\((\d+\.?\d*)%",
        correlated_flag,
    )
    if not m:
        logger.warning("macro_event_flag_parse_failed", flag=correlated_flag)
        return None

    peak_count = int(m.group(1))
    event_date = m.group(3)
    correlation_pct = float(m.group(4)) / 100.0
    severity = "HIGH" if correlation_pct >= 0.40 else "MEDIUM"

    total_held = sum(
        1 for item in reconciliation_items if item.ibkr_position is not None
    )
    # Event evidence for region/sector characterization — the canonical
    # basis-aware predicate from portfolio_health (do not re-declare the
    # action/sell_type tuple here), plus macro-demoted items (demotion has
    # ALREADY run by the time this executes, and a demoted item's basis no
    # longer matches the evidence predicate).
    sell_items = [
        item
        for item in reconciliation_items
        if item.ibkr_position is not None
        and (is_macro_event_evidence(item) or "[MACRO_" in (item.reason or ""))
    ]

    scope, primary_region, primary_sector, impact, event_type, headline, detail = (
        _characterize_macro_event(event_date, sell_items, correlation_pct, peak_count)
    )

    # Anchor expiry on the LATER of event date and detection date: for an
    # ongoing situation (e.g. a months-long strait closure) each re-detection
    # rolls the override window forward instead of letting it expire while
    # the event is still live.
    anchor = max(date.fromisoformat(event_date), date.today())
    expiry = (anchor + timedelta(days=_EXPIRY_DAYS.get(event_type, 60))).isoformat()

    event = MacroEvent(
        event_date=event_date,
        detected_date=date.today().isoformat(),
        expiry=expiry,
        impact=impact,
        event_type=event_type,
        scope=scope,
        primary_region=primary_region,
        primary_sector=primary_sector,
        severity=severity,
        correlation_pct=correlation_pct,
        peak_count=peak_count,
        total_held=total_held,
        news_headline=headline,
        news_detail=detail,
        forced_reanalysis=(impact == "STRUCTURAL" and correlation_pct >= 0.40),
    )

    # Storage is best-effort: the returned event drives the caller's banner even
    # when Chroma is unavailable or store_event raises.
    try:
        store = create_macro_events_store()
        if store.available:
            store.store_event(event)
    except Exception as e:
        logger.warning("macro_event_storage_failed", error=str(e))

    return event


# ══════════════════════════════════════════════════════════════════════════════
# Report Formatting
# ══════════════════════════════════════════════════════════════════════════════

_MAX_DIP_CANDIDATES = 7


def _ccy(currency: str | None) -> str:
    """Return ISO-code-first local currency display prefix."""
    return currency_prefix(currency)


def _normalize_sector_label(sector: str) -> str:
    """Backward-compatible wrapper for the shared sector normalizer."""
    return shared_normalize_sector_label(sector)


def _aggregate_sector_weights(
    sector_weights: dict[str, float] | None,
) -> dict[str, float]:
    """Backward-compatible wrapper for the shared sector aggregation helper."""
    return shared_aggregate_sector_weights(sector_weights)


def _item_currency(item: ReconciliationItem) -> str | None:
    return report_item_currency(item)


def _urgency_prefix(item: ReconciliationItem) -> str:
    return report_urgency_prefix(item)


def _bar_chart(pct: float, limit: float, width: int = 14) -> str:
    """ASCII bar scaled so 'limit' fills the full bar width."""
    return report_bar_chart(pct, limit, width)


def _compute_dip_score(item: ReconciliationItem) -> float:
    """Backward-compatible alias for the shared dip-watch scorer."""
    return score_dip_watch_item(item)


def _display_ticker(item: ReconciliationItem) -> str:
    """Return IBKR-format symbol for all user-visible tickers.

    IBKR format (no exchange suffix, e.g. "WDO", "7203", "MEGP") is what the
    user sees and types in the IBKR UI.  Run commands use run_ticker_for()
    which returns yFinance format with exchange suffix (e.g. "WDO.TO").
    """
    return report_display_ticker(item)


def format_report(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
    show_recommendations: bool = False,
    portfolio_health_flags: list[str] | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    live_orders: list[dict] | None = None,
    errors: dict[str, str] | None = None,
    watchlist_name: str | None = None,
    watchlist_total: int | None = None,
    watchlist_tickers: set[str] | None = None,
    watchlist_candidates_blocked_by_cash: int = 0,
    freshness_summary: AnalysisFreshnessSummary | None = None,
    refresh_activity: RefreshActivity | None = None,
    screening_freshness: ScreeningFreshnessSummary | None = None,
    portfolio_data_loaded: bool = True,
    current_macro_event: MacroEvent | None = None,
    exchange_limit_pct: float = DEFAULT_EXCHANGE_LIMIT_PCT,
    sector_limit_pct: float = DEFAULT_SECTOR_LIMIT_PCT,
) -> str:
    """Format reconciliation results as sectioned human-readable text.

    ``portfolio_data_loaded`` is False in read-only/offline runs where no IBKR
    connection was made, so holdings, watchlist, and cash are unknown rather
    than zero. The report avoids asserting own/watchlist status in that case.
    """
    generated_at = datetime.now()
    items = [retail_safe_action(item) for item in items]
    error_map = errors or {}
    watchlist_unavailable = bool(error_map.get("watchlist")) and portfolio_data_loaded
    action_plan = build_portfolio_action_plan(
        items,
        portfolio,
        watchlist_tickers=watchlist_tickers,
        watchlist_supplied=watchlist_total is not None and not watchlist_unavailable,
        watchlist_unavailable=watchlist_unavailable,
        live_orders=live_orders,
        macro_event_active=has_active_macro_event(portfolio_health_flags),
        exchange_limit_pct=exchange_limit_pct,
        sector_limit_pct=sector_limit_pct,
    )
    cash_summary = build_cash_summary(
        items,
        portfolio,
        executable_buy_ids=action_plan.executable_buy_ids,
    )
    freshness_summary = freshness_summary or _refresh_service.classify(
        items,
        max_age_days=max_age_days,
    )
    refresh_activity = refresh_activity or RefreshActivity(policy="off", limit=0)
    screening_freshness = screening_freshness or ScreeningFreshnessSummary(
        status="missing"
    )
    freshness_user_action = _refresh_service.user_action(
        freshness_summary,
        refresh_activity,
        show_recommendations=show_recommendations,
        command_builder=_portfolio_manager_command,
    )
    context = PortfolioReportContext(
        items=tuple(items),
        portfolio=portfolio,
        plan=action_plan,
        cash_summary=cash_summary,
        portfolio_health_flags=tuple(portfolio_health_flags or ()),
        max_age_days=max_age_days,
        live_orders=tuple(live_orders or ()),
        errors=error_map,
        watchlist_name=watchlist_name,
        watchlist_total=watchlist_total,
        watchlist_candidates_blocked_by_cash=watchlist_candidates_blocked_by_cash,
        freshness_summary=freshness_summary,
        refresh_activity=refresh_activity,
        screening_freshness=screening_freshness,
        portfolio_data_loaded=portfolio_data_loaded,
        current_macro_event=current_macro_event,
        exchange_limit_pct=exchange_limit_pct,
        sector_limit_pct=sector_limit_pct,
        show_recommendations=show_recommendations,
        generated_at=generated_at.isoformat(),
        today_iso=date.today().isoformat(),
    )
    lines = [
        *render_header_and_status_sections(
            context,
            analysis_command=_analysis_command,
            freshness_user_action=freshness_user_action,
        ),
        *render_position_and_risk_sections(
            context,
            analysis_command=_analysis_command,
        ),
        *render_cash_execution_and_summary_sections(
            context,
            analysis_command=_analysis_command,
            recommend_command=_portfolio_manager_recommend_command,
        ),
    ]
    return "\n".join(lines)


def _plan_item_ref(item: ReconciliationItem) -> dict[str, object]:
    return {
        "ticker_yf": item.ticker.yf,
        "ticker_ibkr": item.ticker.ibkr,
        "action": item.action,
        "is_watchlist": item.is_watchlist,
    }


def _plan_note_ref(note: Any) -> dict[str, object]:
    return {
        **_plan_item_ref(note.item),
        "breaches": [asdict(breach) for breach in note.breaches],
    }


def _serialize_recommendation_plan(
    plan: PortfolioActionPlan,
    items: list[ReconciliationItem],
) -> dict[str, object]:
    optimization = plan.optimization
    return {
        "macro_event_active": plan.macro_event_active,
        "summary_counts": build_action_plan_counts(plan, items),
        "executable_buy_ids": sorted(plan.executable_buy_ids),
        "in_flight_buys": [_plan_item_ref(item) for item in plan.in_flight_buys],
        "concentration_withheld_dips": [
            _plan_item_ref(item) for item in plan.concentration_withheld_dips
        ],
        "watchlist": {
            "case": optimization.case.value,
            "target_size": optimization.target_size,
            "keep": [_plan_item_ref(item) for item in optimization.keep],
            "add": [_plan_item_ref(item) for item in optimization.add],
            "remove": [
                {
                    **_plan_item_ref(move.item),
                    "reason": move.reason,
                    "concentration": (
                        _plan_note_ref(move.note) if move.note is not None else None
                    ),
                }
                for move in optimization.remove
            ],
            "monitors": [_plan_item_ref(item) for item in optimization.monitors],
            "reviews": [_plan_item_ref(item) for item in optimization.reviews],
            "withheld": [
                _plan_note_ref(note) for note in optimization.withheld_candidates
            ],
            "admitted_over_limit": [
                _plan_note_ref(note) for note in optimization.admitted_over_limit
            ],
            "protected_tickers": list(optimization.protected_tickers),
        },
    }


def format_json(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
    *,
    freshness_summary: AnalysisFreshnessSummary | None = None,
    refresh_activity: RefreshActivity | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    show_recommendations: bool = False,
    screening_freshness: ScreeningFreshnessSummary | None = None,
    portfolio_data_loaded: bool = True,
    errors: dict[str, str] | None = None,
    live_orders: list[dict] | None = None,
    portfolio_health_flags: list[str] | None = None,
    watchlist_total: int | None = None,
    watchlist_tickers: set[str] | None = None,
    watchlist_unavailable: bool | None = None,
    exchange_limit_pct: float = DEFAULT_EXCHANGE_LIMIT_PCT,
    sector_limit_pct: float = DEFAULT_SECTOR_LIMIT_PCT,
) -> str:
    """Format reconciliation results as JSON.

    ``portfolio_data_loaded`` is False in read-only/offline runs (no IBKR
    connection): account/cash/positions reflect an empty default, not live data.
    """
    items = [retail_safe_action(item) for item in items]
    freshness_summary = freshness_summary or _refresh_service.classify(
        items,
        max_age_days=max_age_days,
    )
    refresh_activity = refresh_activity or RefreshActivity(policy="off", limit=0)
    screening_freshness = screening_freshness or ScreeningFreshnessSummary(
        status="missing"
    )
    unavailable = (
        bool((errors or {}).get("watchlist")) and portfolio_data_loaded
        if watchlist_unavailable is None
        else watchlist_unavailable
    )
    action_plan = build_portfolio_action_plan(
        items,
        portfolio,
        watchlist_tickers=watchlist_tickers,
        watchlist_supplied=watchlist_total is not None and not unavailable,
        watchlist_unavailable=unavailable,
        live_orders=live_orders,
        macro_event_active=has_active_macro_event(portfolio_health_flags),
        exchange_limit_pct=exchange_limit_pct,
        sector_limit_pct=sector_limit_pct,
    )
    cash_summary = build_cash_summary(
        items,
        portfolio,
        executable_buy_ids=action_plan.executable_buy_ids,
    )
    data = {
        "timestamp": datetime.now().isoformat(),
        "portfolio_data_loaded": portfolio_data_loaded,
        # Non-fatal data-source failures (e.g. {"live_orders": "..."}) so JSON
        # consumers see degraded sections (order-dedup off) rather than assume "none".
        "errors": dict(errors or {}),
        "portfolio": portfolio.model_dump(),
        "items": [item.model_dump() for item in items],
        "recommendation_plan": _serialize_recommendation_plan(action_plan, items),
        "cash_summary": asdict(cash_summary),
        "screening_freshness": {
            "status": screening_freshness.status,
            "screening_date": screening_freshness.screening_date,
            "completed_at": screening_freshness.completed_at,
            "age_days": screening_freshness.age_days,
            "stale_after_days": screening_freshness.stale_after_days,
            "candidate_count": screening_freshness.candidate_count,
            "buy_count": screening_freshness.buy_count,
        },
        "analysis_freshness_summary": {
            "blocking_now_count": len(freshness_summary.blocking_now),
            "stale_in_queue_count": len(freshness_summary.stale_in_queue),
            "due_soon_count": len(freshness_summary.due_soon),
            "candidate_blocked_count": len(freshness_summary.candidate_blocked),
            "refreshed_this_run": refresh_activity.refreshed,
            "refresh_failed": refresh_activity.failed,
            "refresh_policy": refresh_activity.policy,
            "manual_action_required": _refresh_service.user_action(
                freshness_summary,
                refresh_activity,
                show_recommendations=show_recommendations,
                command_builder=_portfolio_manager_command,
            )
            != "none",
            "user_action": _refresh_service.user_action(
                freshness_summary,
                refresh_activity,
                show_recommendations=show_recommendations,
                command_builder=_portfolio_manager_command,
            ),
        },
    }
    return json.dumps(data, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def cmd_test_auth(args) -> None:
    """
    Verify IBKR credentials and connection.

    Checks that all required settings are present, prompts for the OAuth
    token secret if absent, connects in read-only mode, and prints basic
    account information confirming the session is live.
    """
    try:
        from src.ibkr_config import ibkr_config
    except ImportError:
        print(
            "ibind not installed. Run: poetry install",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Checking IBKR credentials...", file=sys.stderr)
    _check_config(ibkr_config)

    print("Validating key files...", file=sys.stderr)
    key_info = _validate_key_files(ibkr_config)

    _prompt_for_missing_secret(ibkr_config)

    account_id = args.account_id or ibkr_config.ibkr_account_id

    print(f"Connecting to IBKR (account: {account_id})...", file=sys.stderr)
    service = IbkrAccountService(
        config=ibkr_config,
    )
    try:
        status = asyncio.run(
            service.verify_connection(
                account_id=account_id,
                include_key_validation=False,
            )
        )
    except IBKRAuthError as e:
        print(f"\nAuthentication error: {e}", file=sys.stderr)
        sys.exit(1)
    except IBKRError as e:
        print(f"\nConnection error: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=== IBKR Authentication: OK ===")
    print()
    print(f"  Configured account:  {account_id}")
    if status.visible_accounts:
        print(f"  Accounts visible:    {', '.join(status.visible_accounts)}")
    print(f"  Signature key:       {key_info.get('signature_key', 'N/A')}")
    print(f"  Encryption key:      {key_info.get('encryption_key', 'N/A')}")
    print(
        f"  Portfolio value:     ${status.portfolio_summary.portfolio_value_usd:,.2f}"
    )
    print(
        f"  Cash balance:        ${status.portfolio_summary.cash_balance_usd:,.2f}"
        f"  ({status.portfolio_summary.cash_pct:.1f}%)"
    )
    print(f"  Open positions:      {status.raw_position_count}")
    print()


def _configure_logging(debug: bool) -> None:
    """Configure structlog and stdlib logging for the script.

    Default level is INFO (human-readable progress lines only).
    Pass --debug to get per-record DEBUG output.
    """
    import logging

    import structlog

    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        force=True,
    )
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.KeyValueRenderer(
                key_order=["timestamp", "level", "event"]
            ),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configure_external_loggers(debug)


def _configure_external_loggers(debug: bool) -> None:
    """Keep normal runs operator-focused by suppressing noisy transport loggers."""
    import logging

    noisy_info_loggers = [
        "httpx",
        "httpcore",
        "openai",
        "anthropic",
        "google",
        "google_genai",
        "ddgs",
        "primp",
        "src.llms",
        "ibind",
        "ibind.ibkr_client",
    ]
    level = logging.DEBUG if debug else logging.WARNING
    for logger_name in noisy_info_loggers:
        logging.getLogger(logger_name).setLevel(level)

    # ibind file-handler loggers emit per-request GET/POST lines; keep them off the
    # console in normal mode so only the higher-level retry summary remains visible.
    ibind_file_logger = logging.getLogger("ibind_fh")
    ibind_file_logger.propagate = False
    ibind_file_logger.setLevel(logging.DEBUG if debug else logging.WARNING)


def _print_status(message: str) -> None:
    """Emit immediate human-readable progress to stderr."""
    print(message, file=sys.stderr, flush=True)


def _format_index_rebuild_notice(details: str | None) -> str:
    """Convert reconciler rebuild details into a user-facing status line."""
    if not details:
        return "Analysis index is invalid; reconstructing from analysis files..."

    parts = details.split(":", 2)
    index_name = parts[0]
    reason = parts[1] if len(parts) >= 2 else "unknown"
    reason_suffix = f" (reason: {reason})"
    if len(parts) == 3:
        reason_suffix = f"{reason_suffix}; detail={parts[2]}"
    return (
        f"Analysis index {index_name} is invalid{reason_suffix}; "
        "reconstructing from analysis files..."
    )


def _load_analyses_with_progress(
    results_dir: Path,
    *,
    label: str = "Loading analyses",
) -> dict[str, AnalysisRecord]:
    """Load analyses with user-visible progress for large result directories."""

    def report(update: AnalysisLoadProgress) -> None:
        if update.phase == "discovered":
            _print_status(
                f"Found {update.total_files} analysis file"
                f"{'' if update.total_files == 1 else 's'} in {results_dir}/; "
                "loading latest result per ticker..."
            )
            return
        if update.phase == "indexed":
            _print_status(
                f"Loaded {update.loaded_analyses} analyses from cache index for {results_dir}/"
            )
            return
        if update.phase == "rebuilding_index":
            _print_status(_format_index_rebuild_notice(update.current_file))
            return
        if update.phase == "parsing":
            _print_status(
                f"  Progress: {update.processed_files}/{update.total_files} files scanned; "
                f"{update.loaded_analyses} latest analyses loaded"
            )
            return
        if update.phase == "complete":
            _print_status(
                f"Loaded {update.loaded_analyses} analyses from {results_dir}/"
            )

    _print_status(f"{label} from {results_dir}/...")
    return load_latest_analyses(results_dir, progress=report)


def _load_ibkr_context(
    args: argparse.Namespace,
    *,
    client_cls=None,
    read_portfolio_fn=None,
    read_watchlist_fn=None,
    config=None,
) -> tuple[
    list[NormalizedPosition],
    PortfolioSummary,
    set[str],
    str | None,
    int | None,
    list[dict],
]:
    """Load live IBKR state with explicit user-visible phase status."""
    if config is None:
        from src.ibkr_config import ibkr_config

        config = ibkr_config
    from src.ibkr.client import IbkrClient
    from src.ibkr.portfolio import read_portfolio, read_watchlist

    service = IbkrPortfolioDataService(
        config=config,
        client_cls=client_cls or IbkrClient,
        read_portfolio_fn=read_portfolio_fn or read_portfolio,
        read_watchlist_fn=read_watchlist_fn or read_watchlist,
        prompt_for_missing_secret_fn=_prompt_for_missing_secret,
    )

    wl_explicitly_requested = args.watchlist_name is not None
    snapshot = asyncio.run(
        service.fetch_snapshot(
            account_id=args.account_id or config.ibkr_account_id,
            watchlist_name=args.watchlist_name,
            explicitly_requested=wl_explicitly_requested,
            cash_buffer_pct=args.cash_buffer,
            include_live_orders=args.recommend,
            progress=_print_status,
        )
    )

    watchlist = snapshot.watchlist
    if watchlist.unavailable:
        # Tier-2 (brokerage) session down: holdings still loaded, so continue with
        # a warning rather than abort. The report adds a WATCHLIST UNAVAILABLE banner.
        print(
            "Warning: could not read your IBKR watchlist (brokerage session "
            "unavailable) — continuing with holdings only; watchlist filtering is "
            "unavailable, so unheld BUY analyses may be shown as BUY CANDIDATES.",
            file=sys.stderr,
        )
    elif not watchlist.found and wl_explicitly_requested:
        wl_name_hint = args.watchlist_name or ""
        print(
            f"Error: watchlist '{wl_name_hint}' not found in IBKR.\n"
            f"Use --watchlist-name with a substring that matches one of your IBKR watchlist names.",
            file=sys.stderr,
        )
        sys.exit(1)
    if watchlist.found and not watchlist.tickers and wl_explicitly_requested:
        wl_name_hint = args.watchlist_name or ""
        print(
            f"Warning: could not load watchlist '{wl_name_hint}' "
            f"(API error or watchlist is empty — see log above). "
            f"Continuing without watchlist filtering.",
            file=sys.stderr,
        )
    if watchlist.tickers:
        loaded_name = args.watchlist_name or ""
        print(
            f"Loaded {len(watchlist.tickers)} watchlist tickers from '{loaded_name}'",
            file=sys.stderr,
        )

    return (
        snapshot.positions,
        snapshot.portfolio,
        watchlist.tickers,
        watchlist.loaded_name,
        watchlist.total,
        snapshot.live_orders,
    )


def _install_ibkr_session_teardown(args: argparse.Namespace) -> None:
    """Arm the pooled IBKR session for clean teardown.

    Installs main-thread SIGINT/SIGTERM handlers and seeds the pool's config +
    prompt callback so the single shared session is logged out exactly once on
    exit or signal (atexit covers normal exit; the handlers cover ``kill``).
    No-op in read-only mode, which never opens an IBKR connection.
    """
    if getattr(args, "read_only", False):
        return
    from src.ibkr.session_manager import get_ibkr_session_manager
    from src.ibkr_config import ibkr_config

    manager = get_ibkr_session_manager()
    manager.configure(
        config=ibkr_config,
        prompt_for_missing_secret_fn=_prompt_for_missing_secret,
    )
    manager.install_signal_handlers()


def main() -> None:
    args = parse_args()
    _configure_logging(args.debug)
    _install_ibkr_session_teardown(args)
    refresh_policy = _resolve_refresh_policy(args)

    # --test-auth exits immediately after credential check — no analyses needed.
    if args.test_auth:
        cmd_test_auth(args)
        return

    # --execute is disabled for this release; use --recommend for order suggestions.
    if args.execute:
        print(
            "--execute is currently disabled. Use --recommend for order suggestions.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.read_only:
        _preflight_ibkr_requirements()

    request_kwargs = portfolio_request_kwargs_from_args(args)
    results_dir = request_kwargs["results_dir"]
    if args.read_only:
        print("Read-only mode: no IBKR connection", file=sys.stderr)

    portfolio_service = None
    if not args.read_only:
        portfolio_service = IbkrPortfolioDataService(
            prompt_for_missing_secret_fn=_prompt_for_missing_secret,
        )

    async def _run_analysis_for_refresh(
        *,
        ticker: str,
        quick_mode: bool,
        skip_charts: bool,
    ) -> dict | None:
        from src.main import run_analysis

        return await run_analysis(
            ticker=ticker,
            quick_mode=quick_mode,
            skip_charts=skip_charts,
        )

    def _save_refresh_result(result, ticker: str, *, quick_mode: bool) -> Path:
        from src.persistence import attach_run_summary, save_results_to_file

        # Parity with the main analyzer (src.main._attach_run_summary): enrich
        # run_summary (macro provenance, providers, analysis_validity) BEFORE saving
        # so refreshed artifacts match main-path artifacts and the macro self-check
        # in save_results_to_file does not fire a spurious mismatch. Defensive: a
        # failure here must not abort the refresh save.
        try:
            attach_run_summary(result, quick_mode=quick_mode)
        except Exception as exc:
            logger.warning(
                "refresh_run_summary_attach_failed",
                ticker=ticker,
                **summarize_exception(exc, operation="attach_run_summary"),
            )

        return save_results_to_file(
            result,
            ticker,
            quick_mode=quick_mode,
            results_dir=results_dir,
        )

    service = PortfolioRecommendationService(
        portfolio_data_service=portfolio_service,
        refresh_service=_refresh_service,
        load_analyses_fn=_load_analyses_with_progress,
        reconcile_fn=reconcile,
        compute_portfolio_health_fn=compute_portfolio_health,
        run_analysis_fn=_run_analysis_for_refresh,
        save_results_fn=_save_refresh_result,
    )

    request = PortfolioRecommendationRequest(
        **request_kwargs,
        recommend=args.recommend,
        quick_mode=args.quick,
        refresh_policy=refresh_policy,
    )

    try:
        bundle = asyncio.run(service.build_bundle(request, progress=_print_status))
    except ValueError as e:
        if str(e).startswith("No analysis JSONs found in "):
            print(f"No analysis JSONs found in {results_dir}/", file=sys.stderr)
            print(
                "Run some analyses first: "
                f"{_analysis_command('7203.T')} --output "
                f"{Path(results_dir) / '7203.T.md'}",
                file=sys.stderr,
            )
            sys.exit(1)
        if "watchlist '" in str(e) and " not found in IBKR" in str(e):
            print(
                f"Error: {str(e)}.\n"
                "Use --watchlist-name with a substring that matches one of your IBKR watchlist names.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    except IBKRAuthError as e:
        print(f"IBKR auth error: {e}", file=sys.stderr)
        print(
            "Quick fix: close other IBKR logins (especially the IBKR Mobile app), "
            "or run with --read-only for offline mode.",
            file=sys.stderr,
        )
        sys.exit(1)
    except IBKRError as e:
        print(f"IBKR error: {e}", file=sys.stderr)
        sys.exit(1)

    items = bundle.items
    portfolio = bundle.portfolio
    health_flags = bundle.health_flags
    freshness_summary = bundle.freshness_summary
    refresh_activity = bundle.refresh_activity
    screening_freshness = bundle.screening_freshness
    watchlist_tickers = bundle.watchlist_tickers
    _loaded_watchlist_name = bundle.watchlist_name
    _loaded_watchlist_total = bundle.watchlist_total
    _watchlist_candidates_blocked_by_cash = bundle.watchlist_candidates_blocked_by_cash
    _live_orders_data = bundle.live_orders

    # Detect and store macro events (fail-safe — errors caught internally). The
    # returned event drives the report banner so it reflects THIS run's
    # classification rather than a stale/unrelated stored active event.
    _current_macro_event = _store_macro_event_if_detected(health_flags, items)

    # Output
    show_recs = args.recommend

    if args.json:
        output = format_json(
            items,
            portfolio,
            freshness_summary=freshness_summary,
            refresh_activity=refresh_activity,
            max_age_days=args.max_age,
            show_recommendations=show_recs,
            screening_freshness=screening_freshness,
            portfolio_data_loaded=not args.read_only,
            errors=bundle.errors,
            live_orders=_live_orders_data,
            portfolio_health_flags=health_flags,
            watchlist_total=_loaded_watchlist_total,
            watchlist_tickers=watchlist_tickers if watchlist_tickers else None,
            watchlist_unavailable=bundle.watchlist_unavailable,
            exchange_limit_pct=args.exchange_limit,
            sector_limit_pct=args.sector_limit,
        )
    else:
        output = format_report(
            items,
            portfolio,
            show_recommendations=show_recs,
            portfolio_health_flags=health_flags,
            max_age_days=args.max_age,
            live_orders=_live_orders_data,
            errors=bundle.errors,
            watchlist_name=_loaded_watchlist_name,
            watchlist_total=_loaded_watchlist_total,
            watchlist_tickers=watchlist_tickers if watchlist_tickers else None,
            watchlist_candidates_blocked_by_cash=_watchlist_candidates_blocked_by_cash,
            freshness_summary=freshness_summary,
            refresh_activity=refresh_activity,
            screening_freshness=screening_freshness,
            portfolio_data_loaded=not args.read_only,
            current_macro_event=_current_macro_event,
            exchange_limit_pct=args.exchange_limit,
            sector_limit_pct=args.sector_limit,
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
