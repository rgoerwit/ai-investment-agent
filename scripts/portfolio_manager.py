#!/usr/bin/env python3
"""
IBKR Portfolio Reconciliation Tool

Compares live IBKR positions against the equity evaluator's latest
analysis recommendations. Produces position-aware BUY/SELL/HOLD/TRIM/REVIEW
actions that account for existing holdings.

Usage (requires Poetry venv — either activate it or prefix with `poetry run`):
    poetry run python scripts/portfolio_manager.py --test-auth          # Verify IBKR credentials work
    poetry run python scripts/portfolio_manager.py                      # Report only
    poetry run python scripts/portfolio_manager.py --recommend          # + order suggestions
    poetry run python scripts/portfolio_manager.py --execute            # + place orders (with confirmation)
    poetry run python scripts/portfolio_manager.py --read-only          # No IBKR connection (offline)

    # Or activate once: source .venv/bin/activate
    # Then plain `python scripts/portfolio_manager.py` works for the session.

Requires: poetry install -E ibkr
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ibkr.exceptions import IBKRAuthError, IBKRError
from src.ibkr.models import (
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
)
from src.ibkr.reconciler import (
    compute_portfolio_health,
    load_latest_analyses,
    reconcile,
)

_IBKR_OAUTH_PORTAL = (
    "https://ndcdyn.interactivebrokers.com/sso/Login?action=OAUTH&RL=1&ip2loc=US"
)


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
            f"For read-only portfolio reconciliation, the access token is all you need.",
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
_REQUIRED_CREDENTIALS: list[tuple[str, object]] = [
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
            "to the same file. IBKR requires two separate RSA private key files:\n"
            "  - Signature key:   your private signing key (-----BEGIN RSA PRIVATE KEY----- or\n"
            "                     -----BEGIN PRIVATE KEY-----)\n"
            "  - Encryption key:  your private encryption key (same format, different key pair)\n"
            "  The matching PUBLIC keys are what you upload to the IBKR portal — do not use\n"
            "  those here."
        )

    # PEM headers that indicate a public key or certificate (not a private key).
    _PUBLIC_HEADERS = (
        b"-----BEGIN PUBLIC KEY-----",
        b"-----BEGIN RSA PUBLIC KEY-----",
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
                f"  Expected header: -----BEGIN RSA PRIVATE KEY----- or -----BEGIN PRIVATE KEY-----"
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
            print(f"  {err}", file=sys.stderr)
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


def parse_args() -> argparse.Namespace:
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

    # Options
    parser.add_argument(
        "--max-age", type=int, default=14, help="Max analysis age in days (default: 14)"
    )
    parser.add_argument(
        "--drift-pct",
        type=float,
        default=15.0,
        help="Price drift threshold %% (default: 15)",
    )
    parser.add_argument(
        "--cash-buffer",
        type=float,
        default=0.05,
        help="Cash reserve fraction (default: 0.05)",
    )
    parser.add_argument(
        "--refresh-stale",
        action="store_true",
        help="Re-run evaluator on stale tickers and positions with no analysis",
    )
    parser.add_argument(
        "--sector-limit",
        type=float,
        default=30.0,
        help="Warn when a BUY/ADD would push a sector above this %% (default: 30)",
    )
    parser.add_argument(
        "--exchange-limit",
        type=float,
        default=40.0,
        help="Warn when a BUY/ADD would push an exchange above this %% (default: 40)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Use quick mode for re-analysis"
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Never create IBKR connection (offline mode)",
    )
    parser.add_argument(
        "--account-id", type=str, default="", help="Override IBKR account ID"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/", help="Override results directory"
    )
    parser.add_argument(
        "--output", type=str, default="", help="Write report to file (default: stdout)"
    )
    parser.add_argument("--json", action="store_true", help="Structured JSON output")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args()

    # --recommend and --execute override --report-only
    if args.recommend or args.execute:
        args.report_only = False

    return args


# ══════════════════════════════════════════════════════════════════════════════
# Report Formatting
# ══════════════════════════════════════════════════════════════════════════════

ACTION_SYMBOLS = {
    "BUY": "BUY",
    "SELL": "SELL",
    "TRIM": "TRIM",
    "ADD": "ADD",
    "HOLD": " ",
    "REVIEW": "?",
}

_DIVIDER = "═" * 54

_CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",
    "HKD": "HK$",
    "JPY": "¥",
    "TWD": "NT$",
    "KRW": "₩",
    "EUR": "€",
    "GBP": "£",
    "SGD": "S$",
    "AUD": "A$",
    "CAD": "C$",
    "CNY": "¥",
    "CHF": "Fr",
    "SEK": "kr",
    "NOK": "kr",
    "DKK": "kr",
}


def _ccy(currency: str) -> str:
    """Get currency symbol for a currency code."""
    return _CURRENCY_SYMBOLS.get(
        (currency or "").upper(), (currency + " ") if currency else "$"
    )


def _item_currency(item: ReconciliationItem) -> str:
    if item.analysis and item.analysis.currency:
        return item.analysis.currency
    if item.ibkr_position and item.ibkr_position.currency:
        return item.ibkr_position.currency
    return "USD"


def _urgency_prefix(item: ReconciliationItem) -> str:
    return {"HIGH": "  !!", "MEDIUM": "   !", "LOW": "    "}.get(item.urgency, "    ")


def _bar_chart(pct: float, limit: float, width: int = 14) -> str:
    """ASCII bar scaled so 'limit' fills the full bar width."""
    filled = min(width, round(pct / max(limit, 0.1) * width))
    bar = "█" * filled + "░" * (width - filled)
    warn = " ⚠" if pct >= limit * 0.9 else ""
    return f"{bar}{warn}"


def format_report(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
    show_recommendations: bool = False,
    portfolio_health_flags: list[str] | None = None,
    max_age_days: int = 14,
) -> str:
    """Format reconciliation results as sectioned human-readable text."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Header ──────────────────────────────────────────────────────────────
    lines.append(f"=== IBKR Portfolio Reconciliation  {now} ===")
    lines.append("")
    nlv = portfolio.portfolio_value_usd
    cash = portfolio.cash_balance_usd
    settled = portfolio.settled_cash_usd
    available = portfolio.available_cash_usd
    buffer_amt = max(0.0, settled - available)

    lines.append(f"  Account:          {portfolio.account_id or 'N/A'}")
    lines.append(f"  Net liquidation:  ${nlv:>10,.0f}")
    if nlv > 0:
        lines.append(
            f"  Cash (total):     ${cash:>10,.0f}   ({cash / nlv * 100:.1f}%)"
            "  includes unsettled T+2 proceeds"
        )
        lines.append(
            f"  Settled cash:     ${settled:>10,.0f}   ({settled / nlv * 100:.1f}%)"
            "  spendable today"
        )
        lines.append(
            f"  Buffer reserve:   ${buffer_amt:>10,.0f}   ({buffer_amt / nlv * 100:.1f}%)"
            "  held back"
        )
        lines.append(
            f"  Available:        ${available:>10,.0f}"
            "           see cash summary — buys consume this"
        )
    else:
        lines.append(f"  Cash (total):     ${cash:,.0f}")
        lines.append(f"  Settled cash:     ${settled:,.0f}")
        lines.append(f"  Available:        ${available:,.0f}")
    lines.append("")

    # Split by action
    sells = [i for i in items if i.action == "SELL"]
    trims = [i for i in items if i.action == "TRIM"]
    adds = [i for i in items if i.action == "ADD"]
    buys = [i for i in items if i.action == "BUY" and i.ibkr_position is None]
    holds = [i for i in items if i.action == "HOLD"]
    reviews = [i for i in items if i.action == "REVIEW"]

    def _section(title: str, subtitle: str = "") -> None:
        lines.append(_DIVIDER)
        header = f"  {title}"
        if subtitle:
            header += f"  ({subtitle})"
        lines.append(header)
        lines.append(_DIVIDER)
        lines.append("")

    def _order_line(item: ReconciliationItem, ccy: str) -> str:
        """Build the first line: urgency + action + ticker + qty + price + order type."""
        pfx = _urgency_prefix(item)
        sym = ACTION_SYMBOLS.get(item.action, item.action)
        parts = [f"{pfx} [{sym}]", f"  {item.ticker:<12}"]
        if show_recommendations:
            if item.suggested_quantity:
                parts.append(f"  {abs(item.suggested_quantity)} sh")
            if item.suggested_price:
                parts.append(f"  @ {_ccy(ccy)}{item.suggested_price:,.2f}")
            if item.suggested_order_type:
                parts.append(f"  {item.suggested_order_type}")
        return "".join(parts)

    def _proceeds_line(item: ReconciliationItem) -> str | None:
        """Build proceeds/settlement line for SELL and TRIM."""
        if not show_recommendations or not item.cash_impact_usd:
            return None
        amount = item.cash_impact_usd
        settle = (
            f"settles {item.settlement_date}  ·  not spendable until then"
            if item.settlement_date
            else ""
        )
        parts = [f"Proceeds: ~${amount:,.0f} USD"]
        if settle:
            parts.append(settle)
        return "             " + "  ·  ".join(parts)

    def _cost_line(item: ReconciliationItem, label: str = "Cost") -> str | None:
        """Build cost line for ADD and BUY."""
        if not show_recommendations or not item.cash_impact_usd:
            return None
        cost = abs(item.cash_impact_usd)
        funded = (
            f"funded from settled cash (${settled:,.0f})"
            if settled > 0
            else "funded from settled cash"
        )
        return f"             {label}: ~${cost:,.0f} USD  ·  {funded}"

    # ── SELLS ────────────────────────────────────────────────────────────────
    if sells:
        _section("SELLS", "action needed — verdict changed or stop breached")
        for item in sells:
            ccy = _item_currency(item)
            lines.append(_order_line(item, ccy))
            lines.append(f"             {item.reason}")
            pl = _proceeds_line(item)
            if pl:
                lines.append(pl)
            lines.append("")

    # ── TRIMS ────────────────────────────────────────────────────────────────
    if trims:
        _section("TRIMS", "reduce to target weight")
        for item in trims:
            ccy = _item_currency(item)
            lines.append(_order_line(item, ccy))
            lines.append(f"             {item.reason}")
            pl = _proceeds_line(item)
            if pl:
                lines.append(pl)
            lines.append("")

    # ── ADDS ─────────────────────────────────────────────────────────────────
    if adds:
        _section("ADDS", "increase underweight positions")
        for item in adds:
            ccy = _item_currency(item)
            lines.append(_order_line(item, ccy))
            lines.append(f"             {item.reason}")
            cl = _cost_line(item)
            if cl:
                lines.append(cl)
            lines.append("")

    # ── NEW BUYS ──────────────────────────────────────────────────────────────
    if buys:
        _section("NEW BUYS")
        for item in buys:
            ccy = _item_currency(item)
            lines.append(_order_line(item, ccy))
            a = item.analysis
            if a:
                conviction = a.conviction or a.trade_block.conviction or ""
                size_pct = a.trade_block.size_pct or a.position_size or 0
                detail_parts: list[str] = []
                if conviction:
                    detail_parts.append(f"{conviction} conviction")
                if size_pct and nlv > 0:
                    target_usd = nlv * size_pct / 100
                    detail_parts.append(f"target {size_pct:.1f}% (${target_usd:,.0f})")
                if detail_parts:
                    lines.append(f"             {'  ·  '.join(detail_parts)}")
            cl = _cost_line(item)
            if cl:
                lines.append(cl)
            lines.append("")

    # ── HOLDS ────────────────────────────────────────────────────────────────
    if holds:
        _section("HOLDS", "no action")
        for item in holds:
            pos = item.ibkr_position
            a = item.analysis
            ccy = _item_currency(item)
            sym = _ccy(ccy)

            weight_str = ""
            if pos and nlv > 0:
                wt = pos.market_value_usd / nlv * 100
                weight_str = f"{wt:.1f}%"

            price_str = ""
            if a and pos and a.entry_price and pos.current_price_local:
                gain = (pos.current_price_local - a.entry_price) / a.entry_price * 100
                price_str = (
                    f"entry {sym}{a.entry_price:,.2f} → {sym}{pos.current_price_local:,.2f}"
                    f" ({gain:+.1f}%)"
                )

            stop_str = f"stop {sym}{a.stop_price:,.2f}" if a and a.stop_price else ""
            t1_str = (
                f"T1 {sym}{a.target_1_price:,.2f}" if a and a.target_1_price else ""
            )

            row_parts = [p for p in [weight_str, price_str, stop_str, t1_str] if p]
            lines.append(f"     [ ]    {item.ticker:<12}  {'  '.join(row_parts)}")

        lines.append("")

    # ── REVIEWS ───────────────────────────────────────────────────────────────
    if reviews:
        _section("REVIEWS", "stale or price-drifted — re-run analysis")
        for item in reviews:
            reason_short = item.reason.replace("Stale analysis: ", "").replace(
                "Position held but no evaluator analysis found", "no analysis found"
            )
            run_cmd = f"python -m src.main --ticker {item.ticker}"
            lines.append(f"     [?]    {item.ticker:<12}  {reason_short}  →  {run_cmd}")
        lines.append("")

    if not items:
        lines.append("  No reconciliation items.")
        lines.append("")

    # ── CONCENTRATION ──────────────────────────────────────────────────────────
    sector_weights = portfolio.sector_weights
    exchange_weights = portfolio.exchange_weights
    if sector_weights or exchange_weights:
        _section("CONCENTRATION")

        if sector_weights:
            lines.append("  Sector:")
            for sector, pct in sorted(sector_weights.items(), key=lambda x: -x[1]):
                bar = _bar_chart(pct, 30.0)
                lines.append(f"    {sector:<22} {pct:>5.1f}%  {bar}")
            lines.append("")

        if exchange_weights:
            from src.ibkr.reconciler import _EXCHANGE_LONG_NAMES

            lines.append("  Exchange:")
            for exch, pct in sorted(exchange_weights.items(), key=lambda x: -x[1]):
                long_name = _EXCHANGE_LONG_NAMES.get(exch, exch)
                bar = _bar_chart(pct, 40.0)
                lines.append(f"    {exch:<5} ({long_name:<12}) {pct:>5.1f}%  {bar}")
            lines.append("")

    # ── PORTFOLIO HEALTH ───────────────────────────────────────────────────────
    if portfolio_health_flags:
        _section("PORTFOLIO HEALTH", "cross-portfolio signals")
        for flag in portfolio_health_flags:
            lines.append(f"  !! {flag}")
        lines.append("")

    # ── CASH SUMMARY ──────────────────────────────────────────────────────────
    if show_recommendations:
        _section("CASH SUMMARY")

        buy_cost_items = [
            i for i in items if i.action in ("ADD", "BUY") and i.cash_impact_usd < 0
        ]
        sell_proceed_items = [
            i for i in items if i.action in ("SELL", "TRIM") and i.cash_impact_usd > 0
        ]

        lines.append(
            f"  Settled cash today:                          ${settled:>7,.0f}"
        )

        total_cost = 0.0
        for item in buy_cost_items:
            cost = abs(item.cash_impact_usd)
            total_cost += cost
            qty_str = (
                f"({abs(item.suggested_quantity)} sh)"
                if item.suggested_quantity
                else ""
            )
            label = f"  {item.action}  {item.ticker}  {qty_str}:"
            lines.append(f"{label:<46}- ${cost:>6,.0f}")

        if buy_cost_items:
            remaining = settled - total_cost
            lines.append("  " + "─" * 48)
            lines.append(
                f"  Settled cash after recommended buys:         ${remaining:>7,.0f}"
            )
            lines.append("")

        if sell_proceed_items:
            settle_dates = [
                i.settlement_date for i in sell_proceed_items if i.settlement_date
            ]
            settle_date_str = settle_dates[0] if settle_dates else "T+2"
            lines.append(f"  Pending inflows (T+2, settles {settle_date_str}):")
            total_proceeds = 0.0
            for item in sell_proceed_items:
                proceeds = item.cash_impact_usd
                total_proceeds += proceeds
                qty_str = (
                    f"({abs(item.suggested_quantity)} sh)"
                    if item.suggested_quantity
                    else ""
                )
                label = f"    {item.action}  {item.ticker}  {qty_str}:"
                lines.append(f"{label:<46}+ ${proceeds:>6,.0f}")
            lines.append(f"{'  Total pending:':<46}  ${total_proceeds:>6,.0f}")
            lines.append("")
            lines.append("  ⚠  Sell/trim proceeds are NOT spendable today.")
            lines.append(
                f"     If orders fill by market close, funds available {settle_date_str}."
            )
            lines.append("     Consider additional BUYs only after settlement date.")
            lines.append("")

    # ── DEFERRED ACTIONS ──────────────────────────────────────────────────────
    # Show a sequenced action plan: what to do today vs after T+2 settlement,
    # and which HOLD positions are approaching their staleness deadline.
    from datetime import date, timedelta
    from datetime import datetime as _dt

    today_str = date.today().isoformat()
    action_today = [i for i in items if i.action in ("SELL", "TRIM")]
    funded_today = [
        i for i in items if i.action in ("ADD", "BUY") and i.cash_impact_usd < 0
    ]
    # Sell proceeds grouped by settlement date
    settle_groups: dict[str, float] = {}
    for i in action_today:
        if i.settlement_date and i.cash_impact_usd > 0:
            settle_groups[i.settlement_date] = (
                settle_groups.get(i.settlement_date, 0.0) + i.cash_impact_usd
            )
    # Upcoming review deadlines: HOLDs / items with analysis nearing max_age_days
    upcoming_reviews: list[
        tuple[str, str, int]
    ] = []  # (ticker, expires_date, days_left)
    for item in items:
        if item.action == "HOLD" and item.analysis:
            remaining_days = max_age_days - item.analysis.age_days
            if 0 < remaining_days <= 7:
                try:
                    expires_dt = _dt.strptime(
                        item.analysis.analysis_date, "%Y-%m-%d"
                    ) + timedelta(days=max_age_days)
                    upcoming_reviews.append(
                        (item.ticker, expires_dt.date().isoformat(), remaining_days)
                    )
                except (ValueError, TypeError):
                    pass

    if action_today or funded_today or settle_groups or upcoming_reviews:
        _section("DEFERRED ACTIONS", "pending settlement & upcoming deadlines")

        if action_today or funded_today:
            lines.append(f"  TODAY ({today_str}):")
            for i in action_today:
                qty_str = (
                    f"  {abs(i.suggested_quantity)} sh" if i.suggested_quantity else ""
                )
                price_str = (
                    f"  @ {_ccy(_item_currency(i))}{i.suggested_price:,.2f}"
                    if i.suggested_price
                    else ""
                )
                lines.append(
                    f"    → {i.action}  {i.ticker}{qty_str}{price_str}  {i.suggested_order_type or 'LMT'}"
                )
            for i in funded_today:
                qty_str = (
                    f"  {abs(i.suggested_quantity)} sh" if i.suggested_quantity else ""
                )
                price_str = (
                    f"  @ {_ccy(_item_currency(i))}{i.suggested_price:,.2f}"
                    if i.suggested_price
                    else ""
                )
                cost_str = (
                    f"  (~${abs(i.cash_impact_usd):,.0f})" if i.cash_impact_usd else ""
                )
                lines.append(
                    f"    → {i.action}  {i.ticker}{qty_str}{price_str}{cost_str}  — funded from settled cash"
                )
            lines.append("")

        for settle_date, proceeds in sorted(settle_groups.items()):
            lines.append(f"  {settle_date} (T+2 proceeds from today's sell/trim):")
            lines.append(f"    → ${proceeds:,.0f} clears on this date")
            lines.append("    → Run before placing any additional buys:")
            lines.append(
                "        poetry run python scripts/portfolio_manager.py --recommend"
            )
            lines.append("")

        if upcoming_reviews:
            lines.append("  Upcoming review deadlines:")
            for ticker, expires, days_left in sorted(
                upcoming_reviews, key=lambda x: x[2]
            ):
                lines.append(
                    f"    → {ticker:<14} expires {expires}  ({days_left}d remaining)"
                    f"  →  poetry run python -m src.main --ticker {ticker}"
                )
            lines.append("")

    # ── Summary line ──────────────────────────────────────────────────────────
    action_counts: dict[str, int] = {}
    for item in items:
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
    order = ["SELL", "TRIM", "ADD", "BUY", "HOLD", "REVIEW"]
    summary_parts = [f"{action_counts[a]} {a}" for a in order if a in action_counts]
    lines.append(f"  Summary:  {'  ·  '.join(summary_parts) or 'empty'}")

    return "\n".join(lines)


def format_json(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
) -> str:
    """Format reconciliation results as JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "portfolio": portfolio.model_dump(),
        "items": [item.model_dump() for item in items],
    }
    return json.dumps(data, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


async def refresh_stale_analysis(ticker: str, quick: bool = True) -> bool:
    """Re-run evaluator on a stale ticker."""
    try:
        from src.main import run_analysis, save_results_to_file

        print(f"  Re-analyzing {ticker} (quick={quick})...", file=sys.stderr)
        result = await run_analysis(ticker=ticker, quick_mode=quick, skip_charts=True)
        if result:
            save_results_to_file(result, ticker, quick_mode=quick)
            print(f"  {ticker} re-analysis complete.", file=sys.stderr)
            return True
        print(f"  {ticker} re-analysis returned no result.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  {ticker} re-analysis failed: {e}", file=sys.stderr)
        return False


def cmd_test_auth(args) -> None:
    """
    Verify IBKR credentials and connection.

    Checks that all required settings are present, prompts for the OAuth
    token secret if absent, connects in read-only mode, and prints basic
    account information confirming the session is live.
    """
    try:
        from src.ibkr.client import IbkrClient
        from src.ibkr.portfolio import build_portfolio_summary
        from src.ibkr_config import ibkr_config
    except ImportError:
        print(
            "ibind not installed. Run: poetry install -E ibkr",
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
    try:
        client = IbkrClient(ibkr_config)
        client.connect(brokerage_session=False)
    except IBKRAuthError as e:
        print(f"\nAuthentication error: {e}", file=sys.stderr)
        sys.exit(1)
    except IBKRError as e:
        print(f"\nConnection error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        accounts = client.get_accounts()
        ledger = client.get_ledger(account_id)
        raw_positions = client.get_positions(account_id)
    except IBKRError as e:
        print(f"\nFailed to fetch account data: {e}", file=sys.stderr)
        client.close()
        sys.exit(1)

    client.close()

    summary = build_portfolio_summary(ledger, [], account_id)

    print()
    print("=== IBKR Authentication: OK ===")
    print()
    print(f"  Configured account:  {account_id}")
    if accounts:
        print(f"  Accounts visible:    {', '.join(accounts)}")
    print(f"  Signature key:       {key_info.get('signature_key', 'N/A')}")
    print(f"  Encryption key:      {key_info.get('encryption_key', 'N/A')}")
    print(f"  Portfolio value:     ${summary.portfolio_value_usd:,.2f}")
    print(
        f"  Cash balance:        ${summary.cash_balance_usd:,.2f}"
        f"  ({summary.cash_pct:.1f}%)"
    )
    print(f"  Open positions:      {len(raw_positions)}")
    print()


def main() -> None:
    args = parse_args()

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

    results_dir = Path(args.results_dir)

    # Load analyses from disk (always works, no IBKR needed)
    analyses = load_latest_analyses(results_dir)
    if not analyses:
        print(f"No analysis JSONs found in {results_dir}/", file=sys.stderr)
        print(
            "Run some analyses first: poetry run python -m src.main --ticker 7203.T --output results/7203.T.md",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded {len(analyses)} analyses from {results_dir}/", file=sys.stderr)

    # Read IBKR portfolio (or use empty if --read-only)
    positions: list[NormalizedPosition] = []
    portfolio = PortfolioSummary()

    if args.read_only:
        print("Read-only mode: no IBKR connection", file=sys.stderr)
        # In read-only mode, just show analyses status
        portfolio = PortfolioSummary(
            portfolio_value_usd=0,
            cash_balance_usd=0,
            available_cash_usd=0,
        )
    else:
        try:
            from src.ibkr.client import IbkrClient
            from src.ibkr.portfolio import read_portfolio
            from src.ibkr_config import ibkr_config

            _prompt_for_missing_secret(ibkr_config)
            client = IbkrClient(ibkr_config)
            client.connect(brokerage_session=False)

            account_id = args.account_id or ibkr_config.ibkr_account_id
            positions, portfolio = read_portfolio(
                client,
                account_id,
                args.cash_buffer,
            )
            client.close()

        except ImportError:
            print(
                "ibind not installed. Run: poetry install -E ibkr\n"
                "Or use --read-only for offline mode.",
                file=sys.stderr,
            )
            sys.exit(1)
        except IBKRAuthError as e:
            print(f"IBKR auth error: {e}", file=sys.stderr)
            print("Check IBKR credentials in .env or use --read-only", file=sys.stderr)
            sys.exit(1)
        except IBKRError as e:
            print(f"IBKR error: {e}", file=sys.stderr)
            sys.exit(1)

    # Reconcile
    items = reconcile(
        positions=positions,
        analyses=analyses,
        portfolio=portfolio,
        max_age_days=args.max_age,
        drift_threshold_pct=args.drift_pct,
        sector_limit_pct=args.sector_limit,
        exchange_limit_pct=args.exchange_limit,
    )

    # Compute portfolio-level health flags (uses weights populated by reconcile())
    health_flags = compute_portfolio_health(
        positions=positions,
        analyses=analyses,
        portfolio=portfolio,
        max_age_days=args.max_age,
    )

    # Handle stale refreshes — also picks up positions with no analysis at all
    if args.refresh_stale:
        stale_tickers = [
            item.ticker
            for item in items
            if item.action == "REVIEW"
            and (
                "stale" in (item.reason or "").lower()
                or "no evaluator analysis found" in (item.reason or "").lower()
            )
        ]
        if stale_tickers:
            print(
                f"\nRefreshing {len(stale_tickers)} stale analyses...", file=sys.stderr
            )
            for ticker in stale_tickers:
                asyncio.run(refresh_stale_analysis(ticker, quick=args.quick))

            # Reload and re-reconcile
            analyses = load_latest_analyses(results_dir)
            items = reconcile(
                positions=positions,
                analyses=analyses,
                portfolio=portfolio,
                max_age_days=args.max_age,
                drift_threshold_pct=args.drift_pct,
                sector_limit_pct=args.sector_limit,
                exchange_limit_pct=args.exchange_limit,
            )
            health_flags = compute_portfolio_health(
                positions=positions,
                analyses=analyses,
                portfolio=portfolio,
                max_age_days=args.max_age,
            )

    # Output
    show_recs = args.recommend

    if args.json:
        output = format_json(items, portfolio)
    else:
        output = format_report(
            items,
            portfolio,
            show_recommendations=show_recs,
            portfolio_health_flags=health_flags,
            max_age_days=args.max_age,
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
