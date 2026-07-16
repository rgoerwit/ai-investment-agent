#!/usr/bin/env python3
"""Diagnose the IBKR market-data snapshot behind the optional analysis source.

Answers the two questions that decide whether `IBKR_DATA_SOURCE_ENABLED` can ever
help a given ticker:

  1. Does the identity probe VERIFY (conid + exchange match)? If not, IBKR can't
     supply anything regardless of entitlement.
  2. If VERIFIED, which fundamental field codes actually return values on THIS
     account's market-data entitlements? Blank fundamental fields => `NO_FIELDS`
     (no Korea/Taiwan/etc. real-time subscription) rather than a code bug.

It prints the parsed probe (identity + mapped fundamentals) and, when a conid
resolves, a RAW snapshot dump of every requested field code so you can confirm the
codes/units used by `src/ibkr/security_data_service.py` against live data.

Usage (NOT in the Claude Code sandbox — needs live IBKR creds + network):
    poetry run python scripts/ibkr_snapshot_probe.py 009970.KS 005930.KS 2458.TW
    poetry run python scripts/ibkr_snapshot_probe.py 0005.HK --raw-only
"""

from __future__ import annotations

import argparse
import asyncio

from src.ibkr.security_data_service import (
    _SNAPSHOT_REQUEST_FIELDS,
    IbkrSecurityDataService,
)

# Broad diagnostic field set: the codes the source uses + nearby fundamental codes,
# so a wrong/way-off code shows up as "the value is actually under <other code>".
_DIAGNOSTIC_FIELDS = ",".join(
    sorted(
        set(_SNAPSHOT_REQUEST_FIELDS.split(","))
        | {
            "7281",  # exchange
            "7282",  # avg vol
            "7285",  # P/E (alt?)
            "7286",  # EPS (alt?)
            "7288",  # ratios
            "7295",  # open
            "7296",  # close
            "7308",  # has options
            "7311",  # 52w high (alt?)
            "7607",  # EMA / misc
        },
        key=int,
    )
)

_LABELS = {
    "31": "last",
    "55": "symbol",
    "84": "bid",
    "86": "ask",
    "87": "volume",
    "6004": "exchange",
    "6008": "conidEx",
    "6509": "mktDataAvail",
    "7051": "company",
    "7289": "marketCap?",
    "7290": "P/E?",
    "7291": "EPS?",
    "7287": "divYield?",
    "7293": "52wHigh?",
    "7294": "52wLow?",
}


def _print_probe(tk, probe) -> None:
    print("=" * 66)
    print(f"TICKER: {tk}")
    print(f"  configured           = {probe.configured}")
    print(
        f"  identity_confidence  = {probe.identity_confidence}   "
        f"(VERIFIED is required for fundamentals)"
    )
    print(f"  error_kind           = {probe.error_kind}")
    print(f"  resolved_conid       = {probe.resolved_conid}")
    print(f"  resolved_yf_ticker   = {probe.resolved_yf_ticker}")
    print(f"  exchange / currency  = {probe.exchange} / {probe.currency}")
    print(f"  last_price           = {probe.last_price}")
    print(f"  mkt_data_availability= {probe.market_data_availability}")
    print("  --- mapped fundamentals (what the source would feed the merge) ---")
    print(f"  fundamentals_status  = {probe.fundamentals_status}")
    print(f"  trailing_pe          = {probe.trailing_pe}")
    print(f"  eps                  = {probe.eps}")
    print(f"  market_cap           = {probe.market_cap}")
    print(f"  dividend_yield       = {probe.dividend_yield}")
    print(
        f"  52w_high / 52w_low   = {probe.fifty_two_week_high} / "
        f"{probe.fifty_two_week_low}"
    )


def _raw_snapshot(service, conid: int) -> None:
    """Low-level dump: every requested field code and its raw value."""
    config = service._resolve_config()
    client = service._client_cls(config)
    try:
        client.connect(brokerage_session=False)
        snap = client.get_marketdata_snapshot(
            conid, fields=_DIAGNOSTIC_FIELDS, compete=False
        )
    finally:
        try:
            client.close()
        except Exception:
            pass
    print("  --- RAW snapshot (requested code -> returned value) ---")
    if not snap:
        print("  (empty snapshot — no brokerage session / no market data returned)")
        return
    for code in sorted((k for k in snap if k.isdigit()), key=int):
        label = _LABELS.get(code, "")
        print(f"    {code:>5} {label:14} = {snap.get(code)!r}")
    extra = [k for k in snap if not k.isdigit()]
    if extra:
        print(f"    (non-numeric keys: {extra})")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="+", help="yfinance-format tickers")
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="only dump the raw snapshot (skip the parsed-probe summary)",
    )
    parser.add_argument(
        "--no-raw", action="store_true", help="skip the raw snapshot dump"
    )
    parser.add_argument(
        "--with-controls",
        action="store_true",
        help="also probe known-liquid controls (0005.HK, AAPL) to distinguish a "
        "per-exchange entitlement gap from a broken integration",
    )
    args = parser.parse_args()

    service = IbkrSecurityDataService()
    if not service._resolve_config().is_configured():
        print("IBKR is NOT configured (.env creds missing). Nothing to probe.")
        return 2

    tickers = list(args.tickers)
    if args.with_controls:
        tickers += [c for c in ("0005.HK", "AAPL") if c not in tickers]

    any_ok = False
    for tk in tickers:
        probe = await service.probe_security(tk)
        if not args.raw_only:
            _print_probe(tk, probe)
        if not args.no_raw and probe.resolved_conid:
            try:
                _raw_snapshot(service, probe.resolved_conid)
            except Exception as exc:  # diagnostic tool: surface, don't swallow
                print(f"  RAW snapshot failed: {type(exc).__name__}: {exc}")
        if probe.fundamentals_status == "OK":
            any_ok = True

    print("=" * 66)
    print(
        "RESULT: IBKR can supply fundamentals for "
        f"{'AT LEAST ONE' if any_ok else 'NONE'} of the probed tickers."
    )
    # Exit codes (consumed by run_ibkr_ab.sh preflight):
    #   0 = IBKR adds value somewhere   1 = configured but inert   2 = not configured
    return 0 if any_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
