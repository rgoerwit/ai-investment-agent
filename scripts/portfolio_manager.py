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
from src.ibkr.order_builder import build_order_dict
from src.ibkr.reconciler import load_latest_analyses, reconcile
from src.ibkr.ticker_mapper import resolve_conid

_IBKR_OAUTH_PORTAL = (
    "https://ndcdyn.interactivebrokers.com/sso/Login?action=OAUTH&RL=1&ip2loc=US"
)


def _prompt_for_missing_secret(config) -> None:
    """Prompt for OAuth token secret if absent. Held in memory only — never written to disk."""
    if not config.get_oauth_access_token_secret():
        from pydantic import SecretStr

        print(
            f"\nIBKR OAuth Access Token Secret not set.\n"
            f"Generate a fresh token at: {_IBKR_OAUTH_PORTAL}\n"
            f"(Token and secret disappear when you leave that page — copy them immediately.)",
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
        help="Report + recommendations + place orders (with per-order confirmation)",
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
        "--refresh-stale", action="store_true", help="Re-run evaluator on stale tickers"
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
    "BUY": "+",
    "SELL": "!",
    "TRIM": "~",
    "HOLD": " ",
    "REVIEW": "?",
}

URGENCY_COLORS = {
    "HIGH": "!!",
    "MEDIUM": " !",
    "LOW": "  ",
}


def format_report(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
    show_recommendations: bool = False,
) -> str:
    """Format reconciliation results as human-readable text."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"=== IBKR Portfolio Reconciliation ({now}) ===")
    lines.append("")
    lines.append(
        f"Account: {portfolio.account_id} | "
        f"Portfolio: ${portfolio.portfolio_value_usd:,.0f} | "
        f"Cash: ${portfolio.cash_balance_usd:,.0f} ({portfolio.cash_pct:.1f}%) | "
        f"Available: ${portfolio.available_cash_usd:,.0f}"
    )
    lines.append("")

    # Held positions
    held_items = [i for i in items if i.ibkr_position is not None]
    if held_items:
        lines.append("--- POSITIONS vs EVALUATOR ---")
        lines.append("")
        for item in held_items:
            pos = item.ibkr_position
            sym = ACTION_SYMBOLS.get(item.action, " ")
            urg = URGENCY_COLORS.get(item.urgency, "  ")

            price_info = ""
            if pos and pos.current_price_local > 0 and pos.avg_cost_local > 0:
                gain = (
                    (pos.current_price_local - pos.avg_cost_local) / pos.avg_cost_local
                ) * 100
                price_info = f" {int(pos.quantity)} @ {pos.avg_cost_local:.2f} → {pos.current_price_local:.2f} ({gain:+.1f}%)"

            verdict_info = ""
            if item.analysis:
                a = item.analysis
                verdict_info = f"  {a.verdict} ({a.analysis_date})"
                if a.health_adj or a.growth_adj:
                    verdict_info += f"  H:{a.health_adj or '?'} G:{a.growth_adj or '?'}"

            lines.append(f"{urg} [{sym}] {item.ticker}{price_info}{verdict_info}")
            lines.append(f"       {item.action} — {item.reason}")

            if show_recommendations and item.suggested_quantity:
                order_type = item.suggested_order_type or "LMT"
                price_str = (
                    f" @ {item.suggested_price:.2f}" if item.suggested_price else ""
                )
                lines.append(
                    f"       → {item.action} {item.suggested_quantity} shares ({order_type}){price_str}"
                )

            lines.append("")

    # New buy recommendations (not held)
    buy_items = [i for i in items if i.ibkr_position is None and i.action == "BUY"]
    if buy_items:
        lines.append("--- NEW BUY RECOMMENDATIONS (not held) ---")
        lines.append("")
        for item in buy_items:
            a = item.analysis
            if a:
                size_pct = a.trade_block.size_pct or a.position_size or 0
                lines.append(
                    f"  {item.ticker}  BUY ({a.analysis_date})  Size:{size_pct:.1f}%  Entry:{a.entry_price or '?'}"
                )
                if show_recommendations and item.suggested_price:
                    lines.append(f"       → BUY @ {item.suggested_price:.2f} (LMT)")
            else:
                lines.append(f"  {item.ticker}  BUY — {item.reason}")
            lines.append("")

    if not items:
        lines.append(
            "No reconciliation items. Portfolio is empty and no BUY recommendations found."
        )
        lines.append("")

    # Summary
    action_counts = {}
    for item in items:
        action_counts[item.action] = action_counts.get(item.action, 0) + 1
    summary_parts = [
        f"{action}: {count}" for action, count in sorted(action_counts.items())
    ]
    lines.append(f"--- Summary: {', '.join(summary_parts) or 'empty'} ---")

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
            save_results_to_file(result, ticker)
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
            client.connect(brokerage_session=args.execute)

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
    )

    # Handle stale refreshes
    if args.refresh_stale:
        stale_tickers = [
            item.ticker
            for item in items
            if item.action == "REVIEW" and "stale" in (item.reason or "").lower()
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
            )

    # Output
    show_recs = args.recommend or args.execute

    if args.json:
        output = format_json(items, portfolio)
    else:
        output = format_report(items, portfolio, show_recommendations=show_recs)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)

    # Order execution (Phase 5 — with per-order confirmation)
    if args.execute:
        actionable = [
            i
            for i in items
            if i.action in ("BUY", "SELL", "TRIM") and i.suggested_quantity
        ]
        if not actionable:
            print("\nNo actionable orders to execute.", file=sys.stderr)
            return

        print(f"\n{len(actionable)} orders ready for execution:", file=sys.stderr)
        for item in actionable:
            print(
                f"  {item.action} {item.ticker}: {item.suggested_quantity} shares",
                file=sys.stderr,
            )

        from src.ibkr.client import IbkrClient
        from src.ibkr_config import ibkr_config

        _prompt_for_missing_secret(ibkr_config)
        exec_client = IbkrClient(ibkr_config)
        exec_client.connect(brokerage_session=True)

        try:
            for item in actionable:
                prompt = (
                    f"\nExecute {item.action} {item.suggested_quantity} {item.ticker}"
                    f" @ {item.suggested_price or 'MKT'}? [y/N]: "
                )
                confirm = input(prompt).strip().lower()
                if confirm != "y":
                    print(f"  Skipped {item.ticker}", file=sys.stderr)
                    continue

                try:
                    conid = resolve_conid(item.ticker, exec_client)
                    if not conid:
                        print(
                            f"  Failed to resolve conid for {item.ticker}",
                            file=sys.stderr,
                        )
                        continue

                    order = build_order_dict(
                        conid=conid,
                        action=item.action if item.action != "TRIM" else "SELL",
                        quantity=item.suggested_quantity,
                        price=item.suggested_price,
                        order_type=item.suggested_order_type,
                        account_id=portfolio.account_id,
                    )
                    result = exec_client.place_order(portfolio.account_id, order)
                    print(
                        f"  Order placed for {item.ticker}: {result}", file=sys.stderr
                    )

                except Exception as e:
                    print(f"  Order failed for {item.ticker}: {e}", file=sys.stderr)
        finally:
            exec_client.close()


if __name__ == "__main__":
    main()
