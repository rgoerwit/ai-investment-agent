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
    AnalysisRecord,
    NormalizedPosition,
    PortfolioSummary,
    ReconciliationItem,
)
from src.ibkr.reconciler import (
    AnalysisLoadProgress,
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
            "  - Signature key:   your RSA signing key in PEM format (PKCS8)\n"
            "  - Encryption key:  your RSA encryption key in PEM format (separate key pair)\n"
            "  Upload the matching public keys to the IBKR portal — not the PEM files here."
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
    parser.add_argument(
        "--watchlist-name",
        type=str,
        default=None,
        help="Name of the IBKR watchlist to evaluate (case-insensitive substring match). "
        'If omitted, tries "default watchlist" and silently skips if not found. '
        "If explicitly provided and not found, aborts.",
    )

    args = parser.parse_args()

    # --recommend and --execute override --report-only
    if args.recommend or args.execute:
        args.report_only = False

    return args


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
        from src.config import config as _cfg

        api_key = _cfg.get_tavily_api_key()
        if api_key:
            from tavily import TavilyClient

            _region_map = {
                "T": "Japan",
                "HK": "Hong Kong",
                "KS": "Korea",
                "TW": "Taiwan",
                "L": "UK",
                "DE": "Germany",
                "AS": "Netherlands",
                "PA": "France",
            }
            region_hint = (
                _region_map.get(top_region.lstrip("."), "")
                if scope == "REGIONAL"
                else ""
            )
            query = (
                f"stock market shock {event_date} {region_hint} cause reason".strip()
            )
            result = TavilyClient(api_key=api_key).search(
                query=query, max_results=3, search_depth="basic"
            )
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
) -> None:
    """Parse CORRELATED_SELL_EVENT, characterize it, store in ChromaDB. Fail-safe."""
    from datetime import date as _date
    from datetime import timedelta as _td

    correlated_flag = next(
        (f for f in health_flags if "CORRELATED_SELL_EVENT" in f), None
    )
    if not correlated_flag:
        return

    m = _re.search(
        r"CORRELATED_SELL_EVENT:\s*(\d+) positions.*?(\d+)d of (\d{4}-\d{2}-\d{2})"
        r".*?\((\d+\.?\d*)%",
        correlated_flag,
    )
    if not m:
        logger.warning("macro_event_flag_parse_failed", flag=correlated_flag)
        return

    peak_count = int(m.group(1))
    event_date = m.group(3)
    correlation_pct = float(m.group(4)) / 100.0
    severity = "HIGH" if correlation_pct >= 0.40 else "MEDIUM"

    total_held = sum(
        1 for item in reconciliation_items if item.ibkr_position is not None
    )
    sell_items = [
        item
        for item in reconciliation_items
        if item.action == "SELL" and item.ibkr_position is not None
    ]

    scope, primary_region, primary_sector, impact, event_type, headline, detail = (
        _characterize_macro_event(event_date, sell_items, correlation_pct, peak_count)
    )

    anchor = _date.fromisoformat(event_date)
    expiry = (anchor + _td(days=_EXPIRY_DAYS.get(event_type, 60))).isoformat()

    try:
        from src.memory import MacroEvent, create_macro_events_store

        store = create_macro_events_store()
        if not store.available:
            return
        store.store_event(
            MacroEvent(
                event_date=event_date,
                detected_date=_date.today().isoformat(),
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
        )
    except Exception as e:
        logger.warning("macro_event_storage_failed", error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Report Formatting
# ══════════════════════════════════════════════════════════════════════════════

ACTION_SYMBOLS = {
    "BUY": "BUY",
    "SELL": "SELL",
    "TRIM": "TRIM",
    "ADD": "ADD",
    "HOLD": "HOLD",
    "REVIEW": "REVIEW",
    "REMOVE": "REMOVE",
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

_MAX_DIP_CANDIDATES = 7


def _ccy(currency: str) -> str:
    """Get currency symbol for a currency code."""
    return _CURRENCY_SYMBOLS.get(
        (currency or "").upper(), (currency + " ") if currency else "$"
    )


def _normalize_sector_label(sector: str) -> str:
    """Normalize sector labels so equivalent names aggregate cleanly."""
    normalized = " ".join((sector or "").split()).strip()
    if not normalized:
        return "Unknown"
    if normalized.casefold() in {"healthcare", "health care"}:
        return "Health Care"
    return normalized


def _aggregate_sector_weights(
    sector_weights: dict[str, float] | None,
) -> dict[str, float]:
    """Merge sector buckets that differ only by benign label variants."""
    aggregated: dict[str, float] = {}
    for sector, pct in (sector_weights or {}).items():
        label = _normalize_sector_label(sector)
        aggregated[label] = aggregated.get(label, 0.0) + pct
    return aggregated


def _item_currency(item: ReconciliationItem) -> str:
    if item.analysis and item.analysis.currency:
        return item.analysis.currency
    if item.ibkr_position and item.ibkr_position.currency:
        return item.ibkr_position.currency
    return "USD"


def _urgency_prefix(item: ReconciliationItem) -> str:
    return {"HIGH": "⚠ ", "MEDIUM": "△ ", "LOW": "  "}.get(item.urgency, "  ")


def _bar_chart(pct: float, limit: float, width: int = 14) -> str:
    """ASCII bar scaled so 'limit' fills the full bar width."""
    filled = min(width, round(pct / max(limit, 0.1) * width))
    bar = "█" * filled + "░" * (width - filled)
    warn = " ⚠" if pct >= limit * 0.9 else ""
    return f"{bar}{warn}"


def _compute_dip_score(item: ReconciliationItem) -> float:
    """Composite dip quality score [0–100]. Higher = better dip opportunity."""
    a = item.analysis
    pos = item.ibkr_position
    if not a:
        return 0.0

    health = a.health_adj or 0.0
    growth = a.growth_adj or 0.0
    base = health * 0.4 + growth * 0.4  # 0–80 pts from fundamentals

    # Dip-below-entry bonus: more attractive if current price < analyst's entry recommendation
    price_bonus = 0.0
    if a.entry_price and pos and pos.current_price_local and a.entry_price > 0:
        dip_pct = (a.entry_price - pos.current_price_local) / a.entry_price * 100
        if dip_pct > 0:
            price_bonus = min(dip_pct * 1.5, 12.0)  # max 12 pts

    # R/R bonus: high upside-to-downside ratio improves ranking
    rr_bonus = 0.0
    if a.target_1_price and a.stop_price and pos and pos.current_price_local:
        current = pos.current_price_local
        if current > 0 and current > a.stop_price:
            upside = (a.target_1_price - current) / current
            downside = max((current - a.stop_price) / current, 0.001)
            rr_bonus = min((upside / downside) * 2.5, 8.0)  # max 8 pts

    return base + price_bonus + rr_bonus


def _display_ticker(item: ReconciliationItem) -> str:
    """Return IBKR-format symbol for all user-visible tickers.

    IBKR format (no exchange suffix, e.g. "WDO", "7203", "MEGP") is what the
    user sees and types in the IBKR UI.  Run commands use _run_ticker_for()
    which returns yFinance format with exchange suffix (e.g. "WDO.TO").
    """
    return item.ticker.ibkr


def _run_ticker_for(item: ReconciliationItem) -> str:
    """Return yfinance ticker for --ticker run commands.

    item.ticker.yf is canonical when exchange is known (the common case).
    Fall back to the analysis record ticker when the position's exchange was
    unresolvable (e.g. SMART for a XETRA stock at ingestion time).
    """
    if item.ticker.has_suffix:
        return item.ticker.yf
    # Fallback: analysis record may carry the canonical suffixed form
    if item.analysis and "." in item.analysis.ticker:
        return item.analysis.ticker
    return item.ticker.yf  # US/ADR or genuinely unresolvable


def format_report(
    items: list[ReconciliationItem],
    portfolio: PortfolioSummary,
    show_recommendations: bool = False,
    portfolio_health_flags: list[str] | None = None,
    max_age_days: int = 14,
    live_orders: list[dict] | None = None,
    watchlist_name: str | None = None,
    watchlist_total: int | None = None,
    watchlist_tickers: set[str] | None = None,
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
            "  includes unsettled sale proceeds (not yet spendable)"
        )
        lines.append(
            f"  Settled cash:     ${settled:>10,.0f}   ({settled / nlv * 100:.1f}%)"
            "  spend this today"
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

    # Split by action — sells are further categorised by sell_type
    stop_sells = [
        i for i in items if i.action == "SELL" and i.sell_type == "STOP_BREACH"
    ]
    hard_sells = [
        i for i in items if i.action == "SELL" and i.sell_type in (None, "HARD_REJECT")
    ]
    soft_sells = [
        i for i in items if i.action == "SELL" and i.sell_type == "SOFT_REJECT"
    ]
    # SOFT_REJECT items demoted to REVIEW by compute_portfolio_health on correlated days
    macro_reviews = [
        i for i in items if i.action == "REVIEW" and i.sell_type == "SOFT_REJECT"
    ]
    trims = [i for i in items if i.action == "TRIM"]
    removes = [i for i in items if i.action == "REMOVE"]
    adds = [i for i in items if i.action == "ADD"]
    buys = [
        i
        for i in items
        if i.action == "BUY" and i.ibkr_position is None and i.is_watchlist
    ]
    buys_offwatch = [
        i
        for i in items
        if i.action == "BUY" and i.ibkr_position is None and not i.is_watchlist
    ]
    holds_real = [i for i in items if i.action == "HOLD" and not i.is_watchlist]
    holds_watch = [i for i in items if i.action == "HOLD" and i.is_watchlist]
    # Exclude macro_reviews from regular REVIEW section (they get their own block below)
    reviews = [
        i for i in items if i.action == "REVIEW" and i.sell_type != "SOFT_REJECT"
    ]

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
        parts = [f"{pfx}{sym:<6}  {_display_ticker(item):<12}"]
        if show_recommendations:
            if item.suggested_quantity:
                parts.append(f"  {abs(item.suggested_quantity)} shares")
            if item.suggested_price:
                parts.append(f"  @ {_ccy(ccy)}{item.suggested_price:,.2f}")
            if item.suggested_order_type:
                parts.append(f"  {item.suggested_order_type}")
            if item.action == "BUY" and not item.suggested_quantity:
                parts.append("  (quantity unavailable — inspect before placing order)")
            if not item.suggested_price:
                parts.append("  (no entry price — re-run analysis)")
        return "".join(parts)

    def _proceeds_line(item: ReconciliationItem) -> str | None:
        """Build proceeds/settlement line for SELL and TRIM."""
        if not show_recommendations or not item.cash_impact_usd:
            return None
        amount = item.cash_impact_usd
        settle = f"spendable on {item.settlement_date}" if item.settlement_date else ""
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
            f"use already-settled cash (${settled:,.0f} available)"
            if settled > 0
            else "use already-settled cash"
        )
        return f"             {label}: ~${cost:,.0f} USD  ·  {funded}"

    def _display_data_line(item: ReconciliationItem) -> None:
        """Append compact fundamentals + dip line for a MACRO_WATCH item."""
        a = item.analysis
        pos = item.ibkr_position
        if not a:
            return
        ccy = _item_currency(item)
        sym = _ccy(ccy)
        h = f"{a.health_adj:.0f}" if a.health_adj is not None else "?"
        g = f"{a.growth_adj:.0f}" if a.growth_adj is not None else "?"
        parts: list[str] = [f"Health:{h}%  Growth:{g}%"]
        if a.zone:
            parts.append(f"Risk:{a.zone}")
        if a.entry_price and pos and pos.current_price_local and a.entry_price > 0:
            chg = (pos.current_price_local - a.entry_price) / a.entry_price * 100
            if abs(chg) >= 90.0:
                parts.append(
                    f"analysis entry {sym}{a.entry_price:,.2f}  now {sym}{pos.current_price_local:,.2f}"
                    f"  (⚠ unit mismatch?)"
                )
            else:
                parts.append(
                    f"analysis entry {sym}{a.entry_price:,.2f}  now {sym}{pos.current_price_local:,.2f}"
                    f"  ({chg:+.1f}%)"
                )
        lines.append("             " + "  |  ".join(parts))

    def _render_dip_watch_section(candidates: list[ReconciliationItem]) -> None:
        """Render the DIP WATCH section for macro-dip candidates."""
        dw = "═" * 54
        lines.append(dw)
        lines.append("  DIP WATCH  (existing positions — consider adding)")
        lines.append(dw)
        lines.append("")
        lines.append("  Ranked by fundamental quality × dip depth × risk/reward:")
        lines.append("")
        for item in candidates:
            a = item.analysis
            pos = item.ibkr_position
            score = _compute_dip_score(item)
            if score >= 75:
                stars = "★★★"
            elif score >= 60:
                stars = "★★ "
            else:
                stars = "★  "
            ccy = _item_currency(item)
            sym = _ccy(ccy)
            h = f"{a.health_adj:.0f}" if a and a.health_adj is not None else "?"
            g = f"{a.growth_adj:.0f}" if a and a.growth_adj is not None else "?"
            hg_str = f"Health:{h}%  Growth:{g}%"
            # Prefer analysis entry price; fall back to IBKR avg cost basis.
            # Both pos.avg_cost_local and pos.current_price_local come from IBKR
            # (same currency unit), so their comparison is reliable.
            _dip_entry = (a.entry_price if a and a.entry_price else None) or (
                pos.avg_cost_local if pos and pos.avg_cost_local else None
            )
            _dip_entry_label = (
                "analysis entry" if (a and a.entry_price) else "IBKR cost basis"
            )
            if _dip_entry and pos and pos.current_price_local and _dip_entry > 0:
                chg = (pos.current_price_local - _dip_entry) / _dip_entry * 100
                if abs(chg) >= 90.0:
                    entry_str = (
                        f"{_dip_entry_label} {sym}{_dip_entry:,.2f}  now {sym}{pos.current_price_local:,.2f}"
                        f"  (⚠ unit mismatch?)"
                    )
                else:
                    entry_str = (
                        f"{_dip_entry_label} {sym}{_dip_entry:,.2f}  now {sym}{pos.current_price_local:,.2f}"
                        f"  ({chg:+.1f}%)"
                    )
            else:
                entry_str = "(no entry price recorded)"
            rr_str = "—"
            if (
                a
                and a.target_1_price
                and a.stop_price
                and pos
                and pos.current_price_local
            ):
                cur = pos.current_price_local
                if cur > 0 and cur > a.stop_price:
                    up = (a.target_1_price - cur) / cur * 100
                    dn = max((cur - a.stop_price) / cur * 100, 0.001)
                    rr = up / dn
                    rr_str = f"R/R {rr:.1f}×  (target +{up:.0f}% / stop -{dn:.0f}%)"
            lines.append(
                f"  {stars}  {_display_ticker(item):<12}  {hg_str}  |  {entry_str}  |  {rr_str}"
            )
        lines.append("")
        lines.append("  → Re-run before acting:")
        for _c in candidates:
            _run_t = _run_ticker_for(_c)
            _sfx_warn = (
                "  ← ⚠ verify exchange suffix (no '.' in ticker)"
                if "." not in _run_t
                else ""
            )
            lines.append(
                f"      poetry run python -m src.main --ticker {_run_t}{_sfx_warn}"
            )
        lines.append("")

    def _norm_reason(r: str) -> str:
        """Translate internal verdict labels to user-facing display text."""
        return r.replace("DO_NOT_INITIATE", "REJECT").replace("Verdict → ", "Verdict: ")

    def _as_of_date(r: str) -> str:
        """Extract just the analysis date from a reason string, e.g. 'analyzed 2026-03-05'.
        Used in sections where the section header already explains the action — the date
        tells the user when this recommendation was generated so they can judge staleness."""
        normed = _norm_reason(r)
        paren = normed.rfind("(")
        if paren != -1 and normed.endswith(")"):
            return f"analyzed {normed[paren + 1:-1]}"
        return normed  # fallback: show full reason

    def _score_line(item: ReconciliationItem) -> str | None:
        """Return an indented fundamentals line (health/growth/zone/verdict) for a SELL item.

        Gives the operator enough context to judge whether a stop breach or
        fundamental failure warrants execution or a macro-event hold.
        """
        a = item.analysis
        if not a:
            return None
        if a.health_adj is None and a.growth_adj is None:
            return None

        date_label = (
            f"Last analysis ({a.analysis_date}):"
            if a.analysis_date and a.age_days < 9999
            else "Last analysis:"
        )

        parts: list[str] = []
        if a.health_adj is not None and a.growth_adj is not None:
            parts.append(f"Health:{a.health_adj:.0f}  Growth:{a.growth_adj:.0f}")
        elif a.health_adj is not None:
            parts.append(f"Health:{a.health_adj:.0f}")
        else:
            parts.append(f"Growth:{a.growth_adj:.0f}")

        if a.zone:
            parts.append(f"Risk zone:{a.zone}")

        verdict_str = a.verdict or ""
        if a.conviction:
            verdict_str += f" ({a.conviction})" if verdict_str else a.conviction
        if verdict_str:
            parts.append(verdict_str)

        return f"             {date_label}  " + "  ·  ".join(parts)

    def _pnl_line(item: ReconciliationItem) -> str | None:
        """Return an indented gain/loss estimate line for a SELL/TRIM item, or None if unavailable."""
        pos = item.ibkr_position
        if not pos or pos.avg_cost_local <= 0 or pos.current_price_local <= 0:
            return None
        if pos.quantity == 0:
            return None

        # Gain/loss percentage (currency-neutral — both values in local currency)
        pct = (pos.current_price_local - pos.avg_cost_local) / pos.avg_cost_local * 100

        # Guard: ≥90% swing likely a MEGP-style currency-unit mismatch in cost basis
        if abs(pct) >= 90.0:
            return (
                "             est. P&L: (⚠ cost basis may have currency-unit mismatch)"
            )

        # Gain/loss in LOCAL currency — avoids FX conversion errors from unrealized_pnl_usd
        sell_qty = abs(item.suggested_quantity or pos.quantity)
        pnl_local = (pos.current_price_local - pos.avg_cost_local) * sell_qty
        ccy_sym = _ccy(pos.currency)

        sign = "+" if pnl_local >= 0 else "-"
        gain_or_loss = "est. gain:" if pnl_local >= 0 else "est. loss:"
        tax_note = "  ·  verify holding period in IBKR" if pnl_local > 0 else ""
        return (
            f"             {gain_or_loss}  {sign}{ccy_sym}{abs(pnl_local):,.0f}"
            f"  ({pct:+.1f}% vs IBKR cost basis {ccy_sym}{pos.avg_cost_local:,.2f})"
            f"{tax_note}"
        )

    def _append_pnl_proceeds(item: ReconciliationItem, ccy: str) -> None:
        """Append combined gain/loss + proceeds on one line, or each separately if only one exists."""
        pnl = _pnl_line(item)
        pl = _proceeds_line(item)
        if pnl and pl:
            lines.append(pnl + "  ·  " + pl.lstrip())
        elif pnl:
            lines.append(pnl)
        elif pl:
            lines.append(pl)

    # ── Live-order lookup ─────────────────────────────────────────────────────
    _live_orders: list[dict] = live_orders or []

    def _candidate_conviction(item: ReconciliationItem) -> str:
        """Return normalised conviction string ('high', 'medium', 'low', '')."""
        a = item.analysis
        if not a:
            return ""
        raw = a.conviction or (a.trade_block.conviction if a.trade_block else "") or ""
        return raw.strip().lower()

    def _candidate_score(item: ReconciliationItem) -> float:
        """Composite score for ranking watchlist candidates: health + growth (0–200)."""
        a = item.analysis
        if not a:
            return 0.0
        return (a.health_adj or 0.0) + (a.growth_adj or 0.0)

    def _buy_pos_tag(item: ReconciliationItem) -> str:
        """Short inline tag for BUY/ADD lines in the action plan."""
        if item.action == "ADD":
            pos = item.ibkr_position
            qty_str = f"{pos.quantity:,.0f} sh" if (pos and pos.quantity) else "held"
            return f"[up position — {qty_str}]"
        if item.is_watchlist:
            return "[watchlist — new position]"
        return "[untracked — new position]"

    def _find_live_order(item: ReconciliationItem) -> tuple[dict, str] | None:
        """Return (order_dict, order_side) for the first live order matching this item, or None.

        Matching strategy (first match wins):

        1. conid — most reliable; used whenever a held position exists and the
           order dict contains a conid field.

        2. Symbol — best-effort fallback for new BUY items with no position.
           IBKR's order symbol format can differ from the yfinance ticker in two ways:
             a. No exchange suffix: "7203" not "7203.T" — handled by splitting on "."
             b. No zero-padding for HK codes: "5" not "0005" — handled below
             c. (Rare) Different base entirely: e.g. IBKR "BRK B" vs yf "BRK-B".
                When a position exists, pos.symbol (the IBKR-native code) is used
                as the authoritative candidate, avoiding the guessing entirely.
                For new BUYs with no position, symbol matching is best-effort only;
                if it misses, the report simply won't annotate the duplicate order.
        """
        if not _live_orders:
            return None
        pos = item.ibkr_position
        conid = pos.conid if pos else None

        # Build the set of symbol strings we'd expect IBKR to use for this ticker.
        # • pos.symbol is the authoritative IBKR-native symbol (available for held positions).
        # • yf_base is the suffix-stripped yfinance base (e.g. "0005" from "0005.HK").
        # • ibkr_base strips HK zero-padding from yf_base (e.g. "5" from "0005"),
        #   since IBKR does not zero-pad HK 4-digit codes.
        yf_base = (
            item.ticker.ibkr.upper()
        )  # IBKR bare symbol (no suffix, no zero-padding)
        hk_padded = item.ticker.yf.split(".")[0].upper()  # e.g. "0005" for HK
        symbol_candidates: set[str] = {yf_base, hk_padded}
        if pos and pos.symbol:
            symbol_candidates.add(pos.symbol.upper())  # authoritative IBKR symbol

        for order in _live_orders:
            matched = False
            order_conid = order.get("conid")
            order_symbol = (order.get("ticker") or order.get("symbol") or "").upper()
            # Priority 1: conid match (exact, reliable for held positions)
            if conid and order_conid is not None:
                try:
                    if int(order_conid) == int(conid):
                        matched = True
                except (TypeError, ValueError):
                    pass
            # Priority 2: symbol match (best-effort; may miss if IBKR base differs)
            if not matched and order_symbol in symbol_candidates:
                matched = True
            if not matched:
                continue
            order_side = (
                "SELL" if order.get("side", "").upper() in ("S", "SELL") else "BUY"
            )
            return (order, order_side)
        return None

    def _order_note(item: ReconciliationItem) -> str | None:
        """Return a compact order-status note if an open order exists for this ticker.

        When the live order is the SAME side as the recommendation (SELL+SELL or BUY+BUY),
        warns explicitly that the order is already submitted. When the sides differ,
        flags as a conflict.
        """
        result = _find_live_order(item)
        if result is None:
            return None
        order, order_side = result
        qty = order.get("remainingSize") or order.get("totalSize") or "?"
        price = order.get("price") or order.get("auxPrice")
        otype = order.get("orderType", "LMT")
        status = order.get("status", "")
        try:
            price_str = f" @ {float(price):.2f}" if price else ""
        except (TypeError, ValueError):
            price_str = f" @ {price}" if price else ""

        rec_side = "SELL" if item.action in ("SELL", "TRIM") else "BUY"
        if order_side == rec_side:
            live_qty: int | None = None
            try:
                if qty != "?":
                    live_qty = int(qty)
            except (TypeError, ValueError):
                pass
            rec_qty = item.suggested_quantity
            if live_qty is not None and rec_qty is not None and live_qty < rec_qty:
                need = rec_qty - live_qty
                return (
                    f"             [PARTIAL ORDER: {live_qty} of {rec_qty} shares already submitted"
                    f" — enter {need} more]"
                )
            return (
                f"             [ORDER ALREADY SUBMITTED: {order_side} {qty}{price_str}"
                f" {otype} ({status}) — do not re-enter]"
            )
        # Opposite-side conflict (e.g., recommending SELL but live BUY pending)
        return (
            f"             [CONFLICT: live {order_side} order {qty}{price_str}"
            f" {otype} ({status}) while recommending {rec_side}]"
        )

    # ── MACRO ALERT BANNER (if correlated sell event detected) ───────────────
    _correlated_flag = next(
        (f for f in (portfolio_health_flags or []) if "CORRELATED_SELL_EVENT" in f),
        None,
    )
    if _correlated_flag:
        # Parse "CORRELATED_SELL_EVENT: N positions changed verdict within Xd of DATE (P% …"
        # Flag format (from compute_portfolio_health):
        #   "CORRELATED_SELL_EVENT: 30 positions changed verdict within 7d of 2026-02-28 (61% …"
        import re as _re

        _bm = _re.search(
            r"(\d+) positions.*?of (\d{4}-\d{2}-\d{2}) \((\d+)%", _correlated_flag
        )
        if _bm:
            _cnt, _dt, _pct = _bm.group(1), _bm.group(2), f"{_bm.group(3)}%"
        else:
            _cnt, _dt, _pct = "?", "?", "?%"
        _W = 52  # inner text width (54 inner box chars minus 2-space left indent)
        _banner_lines = [
            "╔" + "═" * 54 + "╗",
            f"║  {'!! MACRO ALERT':<{_W}}║",
            f"║  {f'{_cnt} positions changed verdict on {_dt}':<{_W}}║",
            f"║  {f'({_pct} of held positions) — probable macro event':<{_W}}║",
            f"║  {'Likely macro event, not individual thesis failure.':<{_W}}║",
            f"║  {'Execute STOP-BREACH SELLs; review others first.':<{_W}}║",
            "╚" + "═" * 54 + "╝",
        ]
        try:
            from src.memory import create_macro_events_store as _cms

            _mstore = _cms()
            if _mstore.available:
                _ev = (_mstore.get_active_events() or [None])[0]
                if _ev and _ev.news_headline != "unknown":
                    import re as _re
                    import textwrap as _tw

                    _prefix = "Characterized: "
                    _headline = _ev.news_headline
                    _wrap_kwargs = {
                        "width": _W,
                        "initial_indent": _prefix,
                        "subsequent_indent": " " * len(_prefix),
                    }
                    _wrapped_all = _tw.wrap(_headline, **_wrap_kwargs)
                    if len(_wrapped_all) <= 2:
                        # Fits cleanly — show as-is
                        _shown = _wrapped_all
                    else:
                        # Too long — find the last sentence boundary that fits in ≤ 2 lines
                        _boundaries = [
                            _m.end() for _m in _re.finditer(r"[.!?](?:\s|$)", _headline)
                        ]
                        _chosen = None
                        for _pos in reversed(_boundaries):
                            _candidate = _headline[:_pos].rstrip()
                            if len(_tw.wrap(_candidate, **_wrap_kwargs)) <= 2:
                                _chosen = _candidate
                                break
                        if _chosen:
                            _shown = _tw.wrap(_chosen, **_wrap_kwargs)
                        else:
                            # No complete sentence fits in 2 lines — show first line only
                            _shown = _wrapped_all[:1]
                    for _wline in _shown:
                        _banner_lines.insert(-1, f"║  {_wline:<{_W}}║")
        except Exception:
            pass
        for bl in _banner_lines:
            lines.append(bl)
        lines.append("")

    # ── SELLS — STOP BREACHED (mechanical — execute) ─────────────────────────
    if stop_sells:
        _section("SELLS — STOP BREACHED", "mechanical — execute these")
        for item in stop_sells:
            ccy = _item_currency(item)
            lines.append(f"{_order_line(item, ccy)}  {_norm_reason(item.reason)}")
            sl = _score_line(item)
            if sl:
                lines.append(sl)
            _append_pnl_proceeds(item, ccy)
            note = _order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    # ── SELLS — FUNDAMENTAL FAILURE (hard reject — execute) ──────────────────
    if hard_sells:
        _section("SELLS — FUNDAMENTAL FAILURE", "hard checks failed — execute")
        for item in hard_sells:
            ccy = _item_currency(item)
            lines.append(f"{_order_line(item, ccy)}  {_as_of_date(item.reason)}")
            sl = _score_line(item)
            if sl:
                lines.append(sl)
            _append_pnl_proceeds(item, ccy)
            note = _order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    # ── SELLS — SOFT REJECTION / MACRO REVIEWS ───────────────────────────────
    # soft_sells: on normal days (no correlated event) — show as SELL
    # macro_reviews: demoted from SELL on correlated days — show as REVIEW
    if soft_sells or macro_reviews:
        if macro_reviews and not soft_sells:
            subtitle = (
                "passed hard checks — review before acting (macro event detected)"
            )
        else:
            subtitle = "passed hard checks — review before acting"
        _section("SELLS — SOFT REJECTION", subtitle)
        for item in soft_sells + macro_reviews:
            ccy = _item_currency(item)
            # Strip internal MACRO_WATCH annotation — section header already contextualises it
            _display_reason = _norm_reason(item.reason.split("  [MACRO_WATCH:")[0])
            lines.append(f"{_order_line(item, ccy)}  {_display_reason}")
            if item in macro_reviews:
                _display_data_line(
                    item
                )  # already shows health/growth + entry/current price
            else:
                sl = _score_line(item)
                if sl:
                    lines.append(sl)
            _append_pnl_proceeds(item, ccy)
            note = _order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    # ── DIP WATCH ────────────────────────────────────────────────────────────
    dip_candidates: list[ReconciliationItem] = []
    if _correlated_flag and macro_reviews:
        dip_candidates = sorted(
            [
                i
                for i in macro_reviews
                if i.analysis
                and (i.analysis.health_adj or 0) >= 55
                and (i.analysis.growth_adj or 0) >= 55
                and _compute_dip_score(i) >= 50
            ],
            key=_compute_dip_score,
            reverse=True,
        )[:_MAX_DIP_CANDIDATES]
        if dip_candidates:
            _render_dip_watch_section(dip_candidates)

    # ── TRIMS ────────────────────────────────────────────────────────────────
    if trims:
        _section("TRIMS", "reduce to target weight")
        for item in trims:
            ccy = _item_currency(item)
            lines.append(f"{_order_line(item, ccy)}  {_norm_reason(item.reason)}")
            _append_pnl_proceeds(item, ccy)
            note = _order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    # ── WATCHLIST REMOVE ─────────────────────────────────────────────────────
    if removes:
        _section("WATCHLIST — REMOVE", "verdict changed — remove from IBKR watchlist")
        for item in removes:
            lines.append(f"  {'REMOVE':<6}  {_display_ticker(item):<12}  {item.reason}")
        lines.append("")

    # ── ADDS ─────────────────────────────────────────────────────────────────
    if adds:
        _section("ADDS", "increase underweight positions")
        for item in adds:
            ccy = _item_currency(item)
            lines.append(f"{_order_line(item, ccy)}  {_norm_reason(item.reason)}")
            pos = item.ibkr_position
            if pos and pos.quantity:
                sym = _ccy(ccy)
                avg_str = (
                    f" @ avg {sym}{pos.avg_cost_local:,.2f}"
                    if pos.avg_cost_local
                    else ""
                )
                lines.append(
                    f"             [upping position — currently hold {pos.quantity:,.0f} shares{avg_str}]"
                )
            cl = _cost_line(item)
            note = _order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    # ── NEW BUYS ──────────────────────────────────────────────────────────────
    # Only shown when items originate from the IBKR watchlist (is_watchlist=True).
    # Section is intentionally absent when no watchlist was loaded.
    if buys:
        _count_str = (
            f"{len(buys)}/{watchlist_total} " if watchlist_total is not None else ""
        )
        _name_str = f"watchlist '{watchlist_name}'" if watchlist_name else "watchlist"
        _wl_subtitle = f"{_count_str}from {_name_str} selected for BUY"
        _section("NEW BUYS", _wl_subtitle)
        for item in buys:
            ccy = _item_currency(item)
            lines.append(_order_line(item, ccy))
            a = item.analysis
            if a:
                conviction = a.conviction or a.trade_block.conviction or "Unspecified"
                size_pct = a.trade_block.size_pct or a.position_size or 0
                detail_parts: list[str] = []
                detail_parts.append(f"{conviction} conviction")
                if size_pct and nlv > 0:
                    target_usd = nlv * size_pct / 100
                    detail_parts.append(f"target {size_pct:.1f}% (${target_usd:,.0f})")
                detail_parts.append("[watchlist — new position]")
                if a.is_quick_mode:
                    detail_parts.append(
                        "⚠ quick mode — re-run full analysis before buying"
                    )
                lines.append(f"             {'  ·  '.join(detail_parts)}")
            cl = _cost_line(item)
            if cl:
                lines.append(cl)
            note = _order_note(item)
            if note:
                lines.append(note)
            lines.append("")

    # ── WATCHLIST CANDIDATES ──────────────────────────────────────────────────
    # Phase 2 BUYs from past analysis runs not yet on the watchlist.
    # Must be reviewed and added to watchlist before executing — not actionable orders.
    #
    # Exclude candidates whose base symbol already appears in an actionable
    # context — SELL/REMOVE (contradictory) or any held position (already owned).
    # Held-position check is belt-and-suspenders: the reconciler's Phase 1 should
    # block these in Phase 2, but ticker-format mismatches (bare "5434" position
    # vs "5434.TW" analysis) can still slip through.
    _action_bases: frozenset[str] = frozenset(
        i.ticker.yf.split(".")[0].upper()
        for i in removes + stop_sells + hard_sells + soft_sells
    )
    _held_bases: frozenset[str] = frozenset(
        i.ticker.yf.split(".")[0].upper() for i in items if i.ibkr_position is not None
    )
    # Belt-and-suspenders: exclude candidates whose base symbol is already on
    # the IBKR watchlist.  The reconciler's Phase 1.5 should have converted
    # these to is_watchlist=True items already, but if conid resolution fails
    # (API error on first encounter) the ticker can silently fall through as
    # a Phase 2 BUY.  Filtering here prevents that leakage.
    _watchlist_bases: frozenset[str] = frozenset(
        t.split(".")[0].upper() for t in (watchlist_tickers or set())
    )
    _cands_deduped = [
        i
        for i in buys_offwatch
        if i.ticker.yf.split(".")[0].upper()
        not in (_action_bases | _held_bases | _watchlist_bases)
    ]
    # Split candidates: exclude any that already have a live BUY order so the
    # ":10" slice is filled with real candidates, not orders already placed.
    _cands_in_flight: list[ReconciliationItem] = []
    _cands_actionable: list[ReconciliationItem] = []
    for _c in _cands_deduped:
        _clo = _find_live_order(_c)
        if _clo and _clo[1] == "BUY":
            _cands_in_flight.append(_c)
        else:
            _cands_actionable.append(_c)

    if _cands_actionable or _cands_in_flight:
        _CONVICTION_RANK = {"high": 0, "medium": 1, "low": 2}
        _top_candidates = sorted(
            _cands_actionable,
            key=lambda i: (
                _CONVICTION_RANK.get(_candidate_conviction(i), 3),
                -_candidate_score(i),
            ),
        )[:10]
        _hidden = len(_cands_actionable) - len(_top_candidates)
        _cand_subtitle = (
            "analysis says BUY — inspect and add to watchlist before acting"
        )
        if _hidden:
            _cand_subtitle += f"  (showing top 10 of {len(_cands_actionable)})"
        _section("WATCHLIST CANDIDATES", _cand_subtitle)
        for item in _top_candidates:
            ccy = _item_currency(item)
            # Append "(yf_ticker)" when the exchange suffix disambiguates the IBKR
            # base symbol — e.g. "DLG (DLG.MI)" vs a bare US ticker "AAPL".
            _yf_hint = f" ({item.ticker.yf})" if "." in item.ticker.yf else ""
            _pfx = _urgency_prefix(item)
            _act = ACTION_SYMBOLS.get(item.action, item.action)
            _sym = f"{_display_ticker(item)}{_yf_hint}"
            _cand_parts = [f"{_pfx}{_act:<6}  {_sym}"]
            if show_recommendations:
                if item.suggested_price:
                    _cand_parts.append(f"  @ {_ccy(ccy)}{item.suggested_price:,.2f}")
                if item.suggested_order_type:
                    _cand_parts.append(f"  {item.suggested_order_type}")
                if not item.suggested_quantity:
                    _cand_parts.append(
                        "  (quantity unavailable — inspect before placing order)"
                    )
                if not item.suggested_price:
                    _cand_parts.append("  (no entry price — re-run analysis)")
            lines.append("".join(_cand_parts))
            a = item.analysis
            if a:
                conviction = a.conviction or a.trade_block.conviction or "Unspecified"
                size_pct = a.trade_block.size_pct or a.position_size or 0
                offwatch_parts = [f"{conviction} conviction"]
                if size_pct and nlv > 0:
                    target_usd = nlv * size_pct / 100
                    offwatch_parts.append(
                        f"target {size_pct:.1f}% (${target_usd:,.0f})"
                    )
                offwatch_parts.append("[not on watchlist — new position]")
                if a.is_quick_mode:
                    offwatch_parts.append("⚠ quick mode — re-run full before adding")
                lines.append(f"             {'  ·  '.join(offwatch_parts)}")
            cl = _cost_line(item)
            if cl:
                lines.append(cl)
            lines.append("")
        if _cands_in_flight:
            _if_syms = ", ".join(_display_ticker(i) for i in _cands_in_flight)
            lines.append(
                f"  ✓ {len(_cands_in_flight)} order{'s' if len(_cands_in_flight) > 1 else ''}"
                f" already in flight ({_if_syms}) — verify in IBKR"
            )
            lines.append("")

    # ── HOLDS ────────────────────────────────────────────────────────────────
    if holds_real:
        _section("HOLDS", "no action")
        for item in holds_real:
            pos = item.ibkr_position
            a = item.analysis
            ccy = _item_currency(item)
            sym = _ccy(ccy)

            weight_str = ""
            if pos and nlv > 0:
                wt = pos.market_value_usd / nlv * 100
                weight_str = f"{wt:.1f}%"

            price_str = ""
            if pos and pos.current_price_local:
                # Prefer analysis entry price; fall back to IBKR avg cost basis
                _entry = (a.entry_price if a and a.entry_price else None) or (
                    pos.avg_cost_local if pos.avg_cost_local else None
                )
                _entry_label = (
                    "analysis entry" if (a and a.entry_price) else "IBKR cost basis"
                )
                if _entry:
                    gain = (pos.current_price_local - _entry) / _entry * 100
                    price_str = (
                        f"{_entry_label} {sym}{_entry:,.2f}  now {sym}{pos.current_price_local:,.2f}"
                        f"  ({gain:+.1f}%)"
                    )

            stop_str = f"stop {sym}{a.stop_price:,.2f}" if a and a.stop_price else ""
            t1_str = (
                f"target {sym}{a.target_1_price:,.2f}" if a and a.target_1_price else ""
            )

            row_parts = [p for p in [weight_str, price_str, stop_str, t1_str] if p]
            lines.append(
                f"  {'HOLD':<6}  {_display_ticker(item):<12}  {'  '.join(row_parts)}"
            )

        lines.append("")

    # ── WATCHLIST MONITORING ──────────────────────────────────────────────────
    if holds_watch:
        _section("WATCHLIST — MONITORING", "on watchlist, not yet a buy")
        for item in holds_watch:
            a = item.analysis
            # Use "Last analysis:" to make clear this is the model's verdict, not a
            # positional hold instruction (the ticker is not held).
            verdict_str = (
                f"Last analysis ({a.analysis_date}): {a.verdict} — not initiated"
                if a
                else "no analysis"
            )
            lines.append(f"  {'WATCH':<6}  {_display_ticker(item):<12}  {verdict_str}")
        lines.append("")

    # ── REVIEWS ───────────────────────────────────────────────────────────────
    if reviews:
        _section("REVIEWS", "stale or price-drifted — re-run analysis")
        for item in reviews:
            reason_short = item.reason.replace("Stale analysis: ", "").replace(
                "Position held but no evaluator analysis found", "no analysis found"
            )
            _run_t = _run_ticker_for(item)
            _sfx_warn = (
                "  ← ⚠ exchange unknown, verify suffix" if "." not in _run_t else ""
            )
            run_cmd = f"python -m src.main --ticker {_run_t}{_sfx_warn}"
            lines.append(
                f"  {'REVIEW':<6}  {_display_ticker(item):<12}  {reason_short}  →  {run_cmd}"
            )
        lines.append("")

    if not items:
        lines.append("  No reconciliation items.")
        lines.append("")

    # ── CONCENTRATION ──────────────────────────────────────────────────────────
    sector_weights = _aggregate_sector_weights(portfolio.sector_weights)
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
                lines.append(f"    {exch:<5} ({long_name:<13}) {pct:>5.1f}%  {bar}")
            lines.append("")

    # ── PORTFOLIO HEALTH ───────────────────────────────────────────────────────
    if portfolio_health_flags:
        _section("PORTFOLIO HEALTH", "cross-portfolio signals")
        for flag in portfolio_health_flags:
            flag_lines = flag.split("\n")
            lines.append(f"  !! {flag_lines[0]}")
            for continuation in flag_lines[1:]:
                lines.append(f"  {continuation}")
        lines.append("")

    # ── CASH SUMMARY ──────────────────────────────────────────────────────────
    if show_recommendations:
        _section("CASH SUMMARY")

        buy_cost_items = [
            i
            for i in items
            if i.action in ("ADD", "BUY")
            and i.cash_impact_usd < 0
            and (i.action != "BUY" or i.is_watchlist)  # exclude unvetted candidates
        ]
        sell_proceed_items = [
            i for i in items if i.action in ("SELL", "TRIM") and i.cash_impact_usd > 0
        ]

        lines.append(
            f"  Already-settled cash (spend now):            ${settled:>7,.0f}"
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
            label = f"  {item.action}  {_display_ticker(item)}  {qty_str}:"
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
            settle_date_str = settle_dates[0] if settle_dates else "in 2 business days"
            lines.append(
                f"  Pending inflows (sale proceeds, clears {settle_date_str}):"
            )
            total_proceeds = 0.0
            for item in sell_proceed_items:
                proceeds = item.cash_impact_usd
                total_proceeds += proceeds
                qty_str = (
                    f"({abs(item.suggested_quantity)} sh)"
                    if item.suggested_quantity
                    else ""
                )
                label = f"    {item.action}  {_display_ticker(item)}  {qty_str}:"
                lines.append(f"{label:<46}+ ${proceeds:>6,.0f}")
            lines.append(f"{'  Total pending:':<46}  ${total_proceeds:>6,.0f}")
            lines.append("")
            lines.append(
                "  ⚠  Do NOT spend sale proceeds today — they have not settled yet."
            )
            lines.append(
                f"     If orders fill by market close, funds clear {settle_date_str}."
            )
            lines.append("     Place additional BUYs only after that settlement date.")
            lines.append("")

    # ── ACTION PLAN ───────────────────────────────────────────────────────────
    # Show a sequenced action plan: what to do today vs after 2-day settlement,
    # and which HOLD positions are approaching their staleness deadline.
    from datetime import date, timedelta
    from datetime import datetime as _dt

    today_str = date.today().isoformat()
    action_today = [i for i in items if i.action in ("SELL", "TRIM")]
    funded_today = [
        i
        for i in items
        if i.action in ("ADD", "BUY")
        and i.cash_impact_usd < 0
        and (
            i.action != "BUY" or i.is_watchlist
        )  # unvetted BUYs go to WATCHLIST CANDIDATES, not here
    ]
    # Sell proceeds grouped by settlement date
    settle_groups: dict[str, float] = {}
    for i in action_today:
        if i.settlement_date and i.cash_impact_usd > 0:
            settle_groups[i.settlement_date] = (
                settle_groups.get(i.settlement_date, 0.0) + i.cash_impact_usd
            )
    # Upcoming review deadlines: real HOLD positions (not watchlist) nearing max_age_days
    upcoming_reviews: list[
        tuple[str, str, str, int]
    ] = []  # (disp_sym, run_ticker, expires_date, days_left)
    for item in items:
        if item.action == "HOLD" and not item.is_watchlist and item.analysis:
            remaining_days = max_age_days - item.analysis.age_days
            if 0 < remaining_days <= 7:
                try:
                    expires_dt = _dt.strptime(
                        item.analysis.analysis_date, "%Y-%m-%d"
                    ) + timedelta(days=max_age_days)
                    upcoming_reviews.append(
                        (
                            _display_ticker(item),
                            _run_ticker_for(item),
                            expires_dt.date().isoformat(),
                            remaining_days,
                        )
                    )
                except (ValueError, TypeError):
                    pass

    if (
        action_today
        or funded_today
        or dip_candidates
        or settle_groups
        or upcoming_reviews
        or _cands_deduped
        or removes
    ):
        _section(
            "ACTION PLAN",
            "execution orders · watchlist moves · when proceeds clear · re-analysis deadlines",
        )

        if action_today or funded_today:
            lines.append(f"  TODAY ({today_str}):")
            for i in action_today:
                qty_str = (
                    f"  {abs(i.suggested_quantity)} shares"
                    if i.suggested_quantity
                    else ""
                )
                price_str = (
                    f"  @ {_ccy(_item_currency(i))}{i.suggested_price:,.2f}"
                    if i.suggested_price
                    else ""
                )
                existing = _find_live_order(i)
                _rec_side = "SELL" if i.action in ("SELL", "TRIM") else "BUY"
                if existing and existing[1] == _rec_side:
                    _order_qty_raw = existing[0].get("remainingSize") or existing[
                        0
                    ].get("totalSize")
                    try:
                        _order_qty: int | None = int(_order_qty_raw)
                    except (TypeError, ValueError):
                        _order_qty = None
                    if (
                        _order_qty is not None
                        and i.suggested_quantity is not None
                        and _order_qty < i.suggested_quantity
                    ):
                        _need = i.suggested_quantity - _order_qty
                        lines.append(
                            f"    → {i.action}  {_display_ticker(i)}  {_need} more shares{price_str}  {i.suggested_order_type or 'LMT'}"
                            f"  ({_order_qty} of {i.suggested_quantity} already submitted)"
                        )
                    else:
                        lines.append(
                            f"    ✓ {i.action}  {_display_ticker(i)}{qty_str}{price_str}  {i.suggested_order_type or 'LMT'}"
                            f"  (order already submitted — verify in IBKR)"
                        )
                else:
                    lines.append(
                        f"    → {i.action}  {_display_ticker(i)}{qty_str}{price_str}  {i.suggested_order_type or 'LMT'}"
                    )
            _buys_in_flight: list[str] = []  # display tickers already placed
            for i in funded_today:
                existing = _find_live_order(i)
                if existing and existing[1] == "BUY":
                    _buys_in_flight.append(_display_ticker(i))
                    continue  # already placed — exclude from action list
                qty_str = (
                    f"  {abs(i.suggested_quantity)} shares"
                    if i.suggested_quantity
                    else ""
                )
                price_str = (
                    f"  @ {_ccy(_item_currency(i))}{i.suggested_price:,.2f}"
                    if i.suggested_price
                    else ""
                )
                cost_str = (
                    f"  (~${abs(i.cash_impact_usd):,.0f})" if i.cash_impact_usd else ""
                )
                qty_note = (
                    ""
                    if i.suggested_quantity
                    else "  [quantity unavailable — inspect before placing order]"
                )
                lines.append(
                    f"    → {i.action}  {_display_ticker(i)}{qty_str}{price_str}{cost_str}"
                    f"{qty_note}  {_buy_pos_tag(i)}  — use already-settled cash"
                )
            if _buys_in_flight:
                lines.append(
                    f"    ✓ {len(_buys_in_flight)} already in flight"
                    f" ({', '.join(_buys_in_flight)}) — verify in IBKR"
                )
            lines.append("")

        if dip_candidates:
            lines.append(f"  DIP OPPORTUNITIES ({today_str}):")
            _dips_in_flight: list[str] = []
            for _di in dip_candidates:
                _dexisting = _find_live_order(_di)
                if _dexisting and _dexisting[1] == "BUY":
                    _dips_in_flight.append(_display_ticker(_di))
                    continue  # already placed — exclude from action list
                _da = _di.analysis
                _dp = _di.ibkr_position
                _dscore = _compute_dip_score(_di)
                _dstars = (
                    "★★★" if _dscore >= 75 else ("★★ " if _dscore >= 60 else "★  ")
                )
                _dqty = (
                    f"{_dp.quantity:,.0f} sh held" if (_dp and _dp.quantity) else "held"
                )
                _dh = (
                    f"{_da.health_adj:.0f}"
                    if (_da and _da.health_adj is not None)
                    else "?"
                )
                _dg = (
                    f"{_da.growth_adj:.0f}"
                    if (_da and _da.growth_adj is not None)
                    else "?"
                )
                lines.append(
                    f"    → DIP ADD  {_display_ticker(_di)}"
                    f"  {_dstars}  score {_dscore:.0f}"
                    f"  H:{_dh}% G:{_dg}%"
                    f"  [{_dqty}]"
                    f"  →  poetry run python -m src.main --ticker {_run_ticker_for(_di)}"
                )
            if _dips_in_flight:
                lines.append(
                    f"    ✓ {len(_dips_in_flight)} already in flight"
                    f" ({', '.join(_dips_in_flight)}) — verify in IBKR"
                )
            lines.append("")

        strong_candidates = sorted(
            [i for i in _cands_actionable if _candidate_conviction(i) == "high"],
            key=_candidate_score,
            reverse=True,
        )[:5]
        if strong_candidates or removes:
            lines.append(f"  WATCHLIST MOVES ({today_str}):")
            for i in strong_candidates:
                _quick_note = (
                    "  ⚠ quick — run full first"
                    if (i.analysis and i.analysis.is_quick_mode)
                    else ""
                )
                lines.append(
                    f"    → ADD TO WATCHLIST  {_display_ticker(i)}"
                    f"  — analysis {i.analysis.analysis_date if i.analysis else '?'} says BUY{_quick_note}"
                    f"  →  poetry run python -m src.main --ticker {_run_ticker_for(i)}"
                )
            skipped = len(_cands_actionable) - len(strong_candidates)
            if skipped > 0:
                lines.append(
                    f"    ({skipped} lower-conviction candidate{'s' if skipped > 1 else ''}"
                    f" in WATCHLIST CANDIDATES above — review before adding)"
                )
            for i in removes:
                verdict = i.analysis.verdict if i.analysis else "DO_NOT_INITIATE"
                lines.append(
                    f"    → REMOVE FROM WATCHLIST  {_display_ticker(i)}"
                    f"  — verdict: {verdict}"
                )
            lines.append("")

        for settle_date, proceeds in sorted(settle_groups.items()):
            lines.append(
                f"  {settle_date} — sale proceeds from today's sells/trims clear:"
            )
            lines.append(f"    → ${proceeds:,.0f} available on this date")
            if dip_candidates:
                top_tickers = "  ".join(_display_ticker(i) for i in dip_candidates[:3])
                lines.append(
                    f"    → Top dip candidates for deployment: {top_tickers}"
                    "  (see DIP OPPORTUNITIES above)"
                )
            lines.append("    → Run before placing any additional buys:")
            lines.append(
                "        poetry run python scripts/portfolio_manager.py --recommend"
            )
            lines.append("")

        if upcoming_reviews:
            lines.append("  Upcoming review deadlines:")
            for disp_sym, yf_t, expires, days_left in sorted(
                upcoming_reviews, key=lambda x: x[3]
            ):
                _sfx_warn = (
                    "  ← ⚠ exchange unknown, verify suffix" if "." not in yf_t else ""
                )
                lines.append(
                    f"    → {disp_sym:<14} expires {expires}  ({days_left}d remaining)"
                    f"  →  poetry run python -m src.main --ticker {yf_t}{_sfx_warn}"
                )
            lines.append("")

    # ── Summary line ──────────────────────────────────────────────────────────
    action_counts: dict[str, int] = {}
    macro_watch_count = 0
    for item in items:
        if item.action == "REVIEW" and item.sell_type == "SOFT_REJECT":
            macro_watch_count += 1
        elif (
            item.action == "BUY"
            and not item.is_watchlist
            and item.ibkr_position is None
        ):
            # Phase 2 non-watchlist BUYs: counted only if not suppressed by a
            # same-base-symbol REMOVE or SELL (those are excluded from _cands_deduped).
            pass
        else:
            action_counts[item.action] = action_counts.get(item.action, 0) + 1
    if _cands_deduped:
        action_counts["CANDIDATES"] = len(_cands_deduped)
    if macro_watch_count:
        action_counts["MACRO_WATCH"] = macro_watch_count
    order = [
        "SELL",
        "REMOVE",
        "TRIM",
        "ADD",
        "BUY",
        "CANDIDATES",
        "HOLD",
        "REVIEW",
        "MACRO_WATCH",
    ]
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
    )
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
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
        "src.llms",
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
    if client_cls is None or read_portfolio_fn is None or read_watchlist_fn is None:
        from src.ibkr.client import IbkrClient
        from src.ibkr.portfolio import read_portfolio, read_watchlist

        client_cls = IbkrClient
        read_portfolio_fn = read_portfolio
        read_watchlist_fn = read_watchlist

    if config is None:
        from src.ibkr_config import ibkr_config

        config = ibkr_config

    _print_status("Preparing IBKR client...")
    _prompt_for_missing_secret(config)
    client = client_cls(config)

    _print_status("Connecting to IBKR...")
    client.connect(brokerage_session=False)

    account_id = args.account_id or config.ibkr_account_id

    _print_status("Loading portfolio from IBKR...")
    positions, portfolio = read_portfolio_fn(
        client,
        account_id,
        args.cash_buffer,
    )

    watchlist_tickers: set[str] = set()
    loaded_watchlist_name: str | None = None
    loaded_watchlist_total: int | None = None

    # args.watchlist_name is None  → user didn't pass the flag; try the default
    #                                name but don't abort if missing.
    # args.watchlist_name is a str → explicitly requested; abort if not found.
    wl_name_hint = (
        args.watchlist_name
        if args.watchlist_name is not None
        else ""  # empty → first available watchlist
    )
    wl_explicitly_requested = args.watchlist_name is not None

    _print_status("Loading watchlist from IBKR...")
    wl_result = read_watchlist_fn(client, wl_name_hint)
    if wl_result is None:
        if wl_explicitly_requested:
            print(
                f"Error: watchlist '{wl_name_hint}' not found in IBKR.\n"
                f"Use --watchlist-name with a substring that matches one of your IBKR watchlist names.",
                file=sys.stderr,
            )
            client.close()
            sys.exit(1)
    elif len(wl_result) == 0:
        if wl_explicitly_requested:
            print(
                f"Warning: could not load watchlist '{wl_name_hint}' "
                f"(API error or watchlist is empty — see log above). "
                f"Continuing without watchlist filtering.",
                file=sys.stderr,
            )
    else:
        watchlist_tickers = wl_result
        loaded_watchlist_name = wl_name_hint or None
        loaded_watchlist_total = len(wl_result)
        print(
            f"Loaded {len(watchlist_tickers)} watchlist tickers from '{wl_name_hint}'",
            file=sys.stderr,
        )

    live_orders_data: list[dict] = []
    if args.recommend:
        _print_status("Loading live orders from IBKR...")
        try:
            live_orders_data = client.get_live_orders()
        except Exception:
            pass  # fail-safe: annotation omitted if unavailable

    client.close()
    return (
        positions,
        portfolio,
        watchlist_tickers,
        loaded_watchlist_name,
        loaded_watchlist_total,
        live_orders_data,
    )


def main() -> None:
    args = parse_args()
    _configure_logging(args.debug)

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
    analyses = _load_analyses_with_progress(results_dir)
    if not analyses:
        print(f"No analysis JSONs found in {results_dir}/", file=sys.stderr)
        print(
            "Run some analyses first: poetry run python -m src.main --ticker 7203.T --output results/7203.T.md",
            file=sys.stderr,
        )
        sys.exit(1)

    # Read IBKR portfolio (or use empty if --read-only)
    positions: list[NormalizedPosition] = []
    portfolio = PortfolioSummary()

    watchlist_tickers: set[str] = set()
    _loaded_watchlist_name: str | None = None  # set when watchlist loads successfully
    _loaded_watchlist_total: int | None = None  # total tickers on the watchlist
    _live_orders_data: list[dict] = []

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
            (
                positions,
                portfolio,
                watchlist_tickers,
                _loaded_watchlist_name,
                _loaded_watchlist_total,
                _live_orders_data,
            ) = _load_ibkr_context(args)

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
        watchlist_tickers=watchlist_tickers if watchlist_tickers else None,
    )

    # Compute portfolio-level health flags (uses weights populated by reconcile()).
    # Pass reconciliation_items so CORRELATED_SELL_EVENT can be detected and
    # SOFT_REJECT SELLs demoted to REVIEW in-place.
    health_flags = compute_portfolio_health(
        positions=positions,
        analyses=analyses,
        portfolio=portfolio,
        max_age_days=args.max_age,
        reconciliation_items=items,
    )

    # Handle stale refreshes — also picks up positions with no analysis at all
    if args.refresh_stale:
        stale_tickers = [
            _run_ticker_for(item)
            for item in items
            if item.action == "REVIEW"
            and (
                "stale" in (item.reason or "").lower()
                or "no evaluator analysis found" in (item.reason or "").lower()
                or "no analysis found" in (item.reason or "").lower()
            )
        ]
        if stale_tickers:
            print(
                f"\nRefreshing {len(stale_tickers)} stale analyses...", file=sys.stderr
            )
            for ticker in stale_tickers:
                asyncio.run(refresh_stale_analysis(ticker, quick=args.quick))

            # Reload and re-reconcile
            analyses = _load_analyses_with_progress(
                results_dir,
                label="Reloading analyses after refresh",
            )
            items = reconcile(
                positions=positions,
                analyses=analyses,
                portfolio=portfolio,
                max_age_days=args.max_age,
                drift_threshold_pct=args.drift_pct,
                sector_limit_pct=args.sector_limit,
                exchange_limit_pct=args.exchange_limit,
                watchlist_tickers=watchlist_tickers if watchlist_tickers else None,
            )
            health_flags = compute_portfolio_health(
                positions=positions,
                analyses=analyses,
                portfolio=portfolio,
                max_age_days=args.max_age,
                reconciliation_items=items,
            )

    # Detect and store macro events (fail-safe — errors caught internally).
    _store_macro_event_if_detected(health_flags, items)

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
            live_orders=_live_orders_data,
            watchlist_name=_loaded_watchlist_name,
            watchlist_total=_loaded_watchlist_total,
            watchlist_tickers=watchlist_tickers if watchlist_tickers else None,
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
