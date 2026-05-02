#!/usr/bin/env python3
"""Permanent diagnostic for the MCP integration.

Bypasses the consultant agent's discretion and calls
``spot_check_metric_mcp_fmp`` directly through the full hook chain (audit,
argument-policy, content-inspection, MCPBudgetHook) under the canonical
``mcp__<server>__<tool>`` name. Use this any time the question is "is the
MCP integration actually moving bytes?" — the answer arrives in ~5s and is
deterministic (no LLM in the loop).

Exit codes
----------
  0  success — vendor returned a numeric ``value``
  1  vendor- or hook-level failure — see ``failure_kind`` in the JSON
  2  config / runtime not available — MCP is disabled or registry unreadable
  3  unexpected internal error (architectural failure, not a vendor failure)

Structured JSON is always printed to stdout so it can be piped into ``jq``
or grepped in CI. Diagnostic INFO lines from the SDK / hooks go to stderr
unless the caller explicitly raises log levels.

Examples
--------
    poetry run python scripts/mcp_smoke.py
    poetry run python scripts/mcp_smoke.py --ticker AAPL --metric currentPrice
    poetry run python scripts/mcp_smoke.py --ticker 6914.T --metric trailingPE --quiet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from src.consultant_tools import spot_check_metric_mcp_fmp
from src.main import build_runtime_services_from_config
from src.runtime_services import use_runtime_services


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-call MCP smoke test bypassing the consultant LLM.",
    )
    parser.add_argument(
        "--ticker",
        default="6914.T",
        help="Stock ticker to test (default: 6914.T)",
    )
    parser.add_argument(
        "--metric",
        default="currentPrice",
        help=(
            "Metric to fetch — see _FMP_METRIC_DISPATCH in "
            "src/consultant_tools.py (default: currentPrice)"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Silence SDK/hook INFO chatter on stderr",
    )
    return parser.parse_args()


async def _run(ticker: str, metric: str) -> tuple[int, dict]:
    services = build_runtime_services_from_config(enable_tool_audit=True)
    if services.mcp_runtime is None:
        return 2, {
            "smoke_status": "no_mcp_runtime",
            "hint": "Set MCP_ENABLED=true and ensure config/mcp_servers.json has at least one enabled server.",
        }

    with use_runtime_services(services):
        raw = await spot_check_metric_mcp_fmp.ainvoke(
            {"ticker": ticker, "metric": metric}
        )

    payload = json.loads(raw)

    # The wrapper returns either a success shape (with "value") or a failure
    # shape (with "failure_kind"). Map either onto a meaningful exit code.
    if "value" in payload:
        return 0, payload
    if payload.get("failure_kind") == "config_error":
        return 2, payload
    return 1, payload


def main() -> int:
    args = _parse_args()
    if args.quiet:
        # Hush the SDK + hook chain INFO chatter; keep WARNING+ visible.
        logging.getLogger().setLevel(logging.WARNING)
        for name in ("mcp", "httpx", "src"):
            logging.getLogger(name).setLevel(logging.WARNING)

    try:
        exit_code, payload = asyncio.run(_run(args.ticker, args.metric))
    except Exception as exc:  # pragma: no cover - smoke harness, not library code
        json.dump(
            {
                "smoke_status": "internal_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            sys.stdout,
            indent=2,
        )
        print()
        return 3

    json.dump(payload, sys.stdout, indent=2)
    print()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
