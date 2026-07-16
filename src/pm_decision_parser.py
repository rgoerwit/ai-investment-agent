"""Neutral, dependency-free parser for Portfolio Manager decision text.

This module is the single home for PM verdict canonicalization and best-effort
score extraction from a final-decision string. It deliberately imports only the
standard library so that lightweight callers — the IBKR analysis index, the
off-watchlist BUY-stability gate, chart/report renderers — can reuse it without
dragging in ``src.agents`` (and its heavy LangGraph/LLM surface) or ``src.charts``.

Two distinct parsing behaviors live elsewhere and must not be merged into one
here:
- ``src.agents.pm_verdict_metadata.pm_verdict_metadata_from_text`` uses the
  PM_BLOCK-structured extractor (``src.charts.extractors.pm_block``).
- ``parse_final_decision_scores`` below is the looser regex fallback used by the
  analysis index. It returns the RAW verdict token; callers canonicalize with
  their own normalizer (``canonicalize_pm_verdict`` here, or the IBKR
  ``_normalize_verdict``) — do not canonicalize inside this function or the
  analysis-index verdict string (e.g. "REJECT") would silently change.
"""

from __future__ import annotations

import re
from typing import Literal, cast

PMVerdict = Literal[
    "BUY",
    "HOLD",
    "SELL",
    "DO_NOT_INITIATE",
    "UNPARSEABLE",
]

__all__ = [
    "PMVerdict",
    "canonicalize_pm_verdict",
    "parse_final_decision_scores",
]


def canonicalize_pm_verdict(raw: str | None) -> PMVerdict:
    """Return the canonical PM verdict label for free-text or block values."""
    cleaned = (raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    cleaned = re.sub(r"_+", "_", cleaned)
    if cleaned in {"DO_NOT_INITIATE", "DONOTINITIATE", "DONOTINITATE", "REJECT"}:
        return "DO_NOT_INITIATE"
    if cleaned in {"BUY", "HOLD", "SELL"}:
        return cast(PMVerdict, cleaned)
    return "UNPARSEABLE"


def parse_final_decision_scores(text: str) -> dict:
    """Extract health_adj, growth_adj, verdict, zone, risk_tally from a PM decision.

    Best-effort parse fallback (the prediction_snapshot is preferred when present);
    risk_tally allows a leading sign and is consumed only by the BUY stability gate.

    The verdict is returned RAW (only space->underscore + upper-cased). Callers
    apply their own canonicalization.
    """
    result: dict = {}

    m = re.search(r"\bHEALTH_ADJ[:\s]+([0-9.]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Financial Health[^0-9\n]+([\d.]+)%", text, re.IGNORECASE)
    if m:
        try:
            result["health_adj"] = float(m.group(1))
        except ValueError:
            pass

    m = re.search(r"\bGROWTH_ADJ[:\s]+([0-9.]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"Growth Transition[^0-9\n]+([\d.]+)%", text, re.IGNORECASE)
    if m:
        try:
            result["growth_adj"] = float(m.group(1))
        except ValueError:
            pass

    verdict_token = r"[A-Z_]+(?:[ \t][A-Z_]+)*"
    for pattern in (
        rf"\bVERDICT[:\s]+({verdict_token})",
        rf"PORTFOLIO MANAGER VERDICT:\s*({verdict_token})",
        r"\*\*Action\*\*:\s*\*\*(\w[\w_ ]*)\*\*",
    ):
        m = re.search(pattern, text)
        if m:
            result["verdict"] = m.group(1).strip().replace(" ", "_").upper()
            break

    m = re.search(r"\bZONE[:\s]+(HIGH|MODERATE|LOW)\b", text, re.IGNORECASE)
    if m:
        result["zone"] = m.group(1).upper()

    # Allow a leading sign — bonuses can drive the tally below zero (e.g. -0.5).
    # The fallback gap excludes '-' so the sign is captured, not consumed.
    m = re.search(r"\bRISK_TALLY[:\s]+(-?[0-9.]+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"TOTAL RISK COUNT[^-0-9\n]*(-?[0-9.]+)", text, re.IGNORECASE)
    if m:
        try:
            result["risk_tally"] = float(m.group(1))
        except ValueError:
            pass

    return result
