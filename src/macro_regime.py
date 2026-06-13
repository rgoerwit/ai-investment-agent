"""Parse and normalize macro regime blocks emitted by the macro context brief."""

from __future__ import annotations

import re
from dataclasses import dataclass

_FIELD_ENUMS: dict[str, frozenset[str]] = {
    "risk_appetite": frozenset({"RISK_ON", "RISK_OFF", "MIXED", "UNCERTAIN"}),
    "shock_type": frozenset(
        {
            "NONE",
            "ENERGY",
            "FX",
            "RATES",
            "CREDIT",
            "GEOPOLITICAL",
            "POLICY",
            "LIQUIDITY",
            "OTHER",
        }
    ),
    "shock_phase": frozenset(
        {"NONE", "ACUTE", "STABILIZING", "AFTERSHOCK", "STRUCTURAL"}
    ),
    "equity_transmission": frozenset(
        {
            "MULTIPLE_COMPRESSION",
            "EARNINGS_PRESSURE",
            "FX_TRANSLATION",
            "FUNDING_STRESS",
            "FLOWS_SUPPORT",
            "MIXED",
            "UNCERTAIN",
        }
    ),
    "dip_posture": frozenset(
        {"BUYABLE", "SCALE_SLOWLY", "WAIT_FOR_CONFIRMATION", "AVOID"}
    ),
    "confidence": frozenset({"HIGH", "MEDIUM", "LOW"}),
}

_BLOCK_RE = re.compile(r"(?ims)^MACRO_REGIME_BLOCK:\s*\n?(?P<body>.*?)(?=^###\s+|\Z)")
_LINE_RE = re.compile(r"^\s*([A-Z_]+)\s*:\s*([A-Z_]+)\s*$")


@dataclass(frozen=True, slots=True)
class MacroRegime:
    """Normalized macro regime fields parsed from the free-form macro brief."""

    risk_appetite: str = "UNCERTAIN"
    shock_type: str = "NONE"
    shock_phase: str = "NONE"
    equity_transmission: str = "UNCERTAIN"
    dip_posture: str = "WAIT_FOR_CONFIRMATION"
    confidence: str = "LOW"
    present: bool = False
    raw_block: str = ""

    def to_dict(self) -> dict[str, str | bool]:
        """Return the persisted structured regime payload."""
        return {
            "risk_appetite": self.risk_appetite,
            "shock_type": self.shock_type,
            "shock_phase": self.shock_phase,
            "equity_transmission": self.equity_transmission,
            "dip_posture": self.dip_posture,
            "confidence": self.confidence,
            "present": self.present,
        }


def parse_macro_regime(report: str | None) -> MacroRegime:
    """Parse the first MACRO_REGIME_BLOCK, normalizing unknown enum values."""
    if not report:
        return MacroRegime()

    match = _BLOCK_RE.search(report)
    if not match:
        return MacroRegime()

    body = match.group("body").strip()
    fields: dict[str, str] = {}
    for line in body.splitlines():
        line_match = _LINE_RE.match(line)
        if not line_match:
            continue
        key = line_match.group(1).lower()
        value = line_match.group(2).upper()
        if value in _FIELD_ENUMS.get(key, frozenset()):
            fields[key] = value

    raw = f"MACRO_REGIME_BLOCK:\n{body}" if body else "MACRO_REGIME_BLOCK:"
    return MacroRegime(present=True, raw_block=raw, **fields)
