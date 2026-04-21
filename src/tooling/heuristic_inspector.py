"""Pattern-based prompt-injection detector.

Implements ``ContentInspector`` for the ``"python"`` backend config slot.
Detects common injection signals — override phrases, role-play coercion,
delimiter breakout, hidden markup, and control-character abuse — using
scored pattern families.

Heuristics are a fast, zero-cost first pass.  For semantic coverage
(paraphrasing, multilingual attacks, context-dependent injection) use
``EscalatingInspector`` which chains this with an LLM judge.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

import structlog

from src.tooling.inspector import (
    InspectionDecision,
    InspectionEnvelope,
    SourceKind,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Signal families — each pattern carries a weight and a type tag.
# ---------------------------------------------------------------------------

_ThreatType = Literal[
    "override",
    "role_play",
    "delimiter_breakout",
    "hidden_markup",
    "encoded_payload",
    "control_chars",
]


@dataclass(frozen=True, slots=True)
class _Signal:
    pattern: re.Pattern[str]
    weight: float
    threat_type: _ThreatType


# Compiled once at import time.
_SIGNALS: list[_Signal] = [
    # --- Override phrases ---
    _Signal(
        re.compile(
            r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|directives|rules|prompts)",
            re.I,
        ),
        3.0,
        "override",
    ),
    _Signal(
        re.compile(
            r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|directives|rules|prompts)",
            re.I,
        ),
        3.0,
        "override",
    ),
    _Signal(
        re.compile(
            r"do\s+not\s+follow\s+(the\s+)?(previous|prior|original|above)\s+(instructions|directives|rules)",
            re.I,
        ),
        3.0,
        "override",
    ),
    _Signal(
        re.compile(
            r"forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|context|rules)",
            re.I,
        ),
        3.0,
        "override",
    ),
    _Signal(
        re.compile(
            r"your\s+new\s+(task|instructions?|objective|role|goal)\s+(is|are)\b", re.I
        ),
        2.5,
        "override",
    ),
    _Signal(re.compile(r"you\s+are\s+now\s+(a|an|the)\b", re.I), 2.0, "override"),
    _Signal(re.compile(r"^system\s*:", re.I | re.M), 2.0, "override"),
    # --- Role-play coercion ---
    _Signal(re.compile(r"pretend\s+(that\s+)?you\s+are\b", re.I), 2.0, "role_play"),
    _Signal(
        re.compile(r"act\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\b", re.I),
        1.5,
        "role_play",
    ),
    _Signal(
        re.compile(r"you\s+must\s+now\s+(act|behave|respond|operate)\b", re.I),
        2.0,
        "role_play",
    ),
    _Signal(
        re.compile(r"switch\s+to\s+(a\s+)?new\s+(mode|persona|role)\b", re.I),
        2.0,
        "role_play",
    ),
    _Signal(
        re.compile(r"entering\s+(DAN|developer|jailbreak|unrestricted)\s+mode", re.I),
        3.0,
        "role_play",
    ),
    # --- Delimiter breakout ---
    _Signal(re.compile(r"</search_results>", re.I), 3.0, "delimiter_breakout"),
    _Signal(re.compile(r"</tool_output>", re.I), 3.0, "delimiter_breakout"),
    _Signal(re.compile(r"</function_results>", re.I), 3.0, "delimiter_breakout"),
    _Signal(re.compile(r"<\s*/\s*system\s*>", re.I), 3.0, "delimiter_breakout"),
    _Signal(
        re.compile(r"---\s*END\s+(SYSTEM|INSTRUCTIONS?|CONTEXT)\s*---", re.I),
        2.5,
        "delimiter_breakout",
    ),
    _Signal(
        re.compile(r"\]\]\s*>\s*>", re.I), 1.5, "delimiter_breakout"
    ),  # ]]>> CDATA-style
    # --- Hidden / injected markup ---
    _Signal(re.compile(r"<!--.*?-->", re.S), 1.0, "hidden_markup"),
    _Signal(re.compile(r"display\s*:\s*none", re.I), 1.5, "hidden_markup"),
    _Signal(re.compile(r"font-size\s*:\s*0", re.I), 1.5, "hidden_markup"),
    _Signal(re.compile(r"visibility\s*:\s*hidden", re.I), 1.5, "hidden_markup"),
    _Signal(
        re.compile(r"color\s*:\s*(?:white|transparent|rgba\s*\([^)]*,\s*0\s*\))", re.I),
        1.0,
        "hidden_markup",
    ),
    _Signal(
        re.compile(r"position\s*:\s*absolute[^;]*left\s*:\s*-\d{4,}", re.I),
        1.0,
        "hidden_markup",
    ),
    # --- Encoded payload hints ---
    _Signal(
        re.compile(r"(?:base64|eval|decode)\s*[\(:]", re.I), 1.5, "encoded_payload"
    ),
    _Signal(re.compile(r"(?:atob|btoa)\s*\(", re.I), 1.5, "encoded_payload"),
]

# Delimiter tags that can be safely stripped.
_STRIPPABLE_DELIMITERS: list[re.Pattern[str]] = [
    re.compile(r"</search_results>", re.I),
    re.compile(r"</tool_output>", re.I),
    re.compile(r"</function_results>", re.I),
    re.compile(r"<\s*/\s*system\s*>", re.I),
]

# Threshold for invisible / control character density (fraction of total).
_CONTROL_CHAR_DENSITY_THRESHOLD = 0.03
_CONTROL_CHAR_MIN_LENGTH = 50  # skip short strings

# Source kinds that receive lighter treatment (lower risk, structured data).
_LIGHT_TREATMENT_SOURCES: frozenset[SourceKind] = frozenset(
    {
        SourceKind.official_filing,
        SourceKind.financial_api,
    }
)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Hit:
    signal: _Signal
    match_text: str


def _detect_signals(text: str) -> list[_Hit]:
    """Run all pattern families against *text* and return hits."""
    hits: list[_Hit] = []
    for sig in _SIGNALS:
        m = sig.pattern.search(text)
        if m:
            hits.append(_Hit(signal=sig, match_text=m.group()[:120]))
    return hits


def _control_char_density(text: str) -> float:
    """Return fraction of invisible / control characters in *text*."""
    if len(text) < _CONTROL_CHAR_MIN_LENGTH:
        return 0.0
    count = sum(
        1
        for ch in text
        if unicodedata.category(ch).startswith("C") and ch not in ("\n", "\r", "\t")
    )
    return count / len(text)


def _strip_known_breakouts(text: str) -> str:
    """Remove delimiter-breakout tags that can be safely stripped."""
    result = text
    for pat in _STRIPPABLE_DELIMITERS:
        result = pat.sub("", result)
    return result


def _classify_severity(
    total_weight: float,
    source_kind: SourceKind,
) -> Literal["safe", "low", "medium", "high", "critical"]:
    """Map aggregate weight to a threat level, adjusted by source kind."""
    # Lighter treatment for structured / official sources.
    if source_kind in _LIGHT_TREATMENT_SOURCES:
        total_weight *= 0.5

    if total_weight >= 6.0:
        return "critical"
    if total_weight >= 4.0:
        return "high"
    if total_weight >= 2.0:
        return "medium"
    if total_weight > 0:
        return "low"
    return "safe"


# ---------------------------------------------------------------------------
# Public inspector class
# ---------------------------------------------------------------------------


class HeuristicInspector:
    """Pattern-based prompt-injection detector.

    Implements ``ContentInspector`` protocol for the ``"python"`` backend.
    """

    async def inspect(self, envelope: InspectionEnvelope) -> InspectionDecision:
        text = envelope.content_text

        # --- Control-character density check ---
        cc_density = _control_char_density(text)
        cc_hit = cc_density > _CONTROL_CHAR_DENSITY_THRESHOLD

        # --- Pattern matching ---
        hits = _detect_signals(text)

        if not hits and not cc_hit:
            return InspectionDecision(action="allow", threat_level="safe")

        total_weight = sum(h.signal.weight for h in hits)
        if cc_hit:
            total_weight += 2.0

        threat_types: list[str] = sorted(
            {h.signal.threat_type for h in hits}
            | ({"control_chars"} if cc_hit else set())
        )
        findings: list[str] = [
            f"{h.signal.threat_type}: {h.match_text!r}" for h in hits
        ]
        if cc_hit:
            findings.append(
                f"control_chars: density={cc_density:.3f} "
                f"(threshold={_CONTROL_CHAR_DENSITY_THRESHOLD})"
            )

        severity = _classify_severity(total_weight, envelope.source_kind)

        # Can we safely sanitize?  Only if the only hits are strippable delimiters.
        all_delimiter_breakout = (
            all(h.signal.threat_type == "delimiter_breakout" for h in hits)
            and not cc_hit
        )
        if all_delimiter_breakout:
            sanitized = _strip_known_breakouts(text)
            return InspectionDecision(
                action="sanitize",
                threat_level=severity,
                threat_types=threat_types,
                sanitized_content=sanitized,
                findings=findings,
                reason="stripped delimiter-breakout tags",
            )

        action: Literal["allow", "sanitize", "block", "degrade"]
        if severity in ("critical", "high"):
            action = "block"
        elif severity == "medium":
            action = "degrade"
        else:
            action = "allow"

        return InspectionDecision(
            action=action,
            threat_level=severity,
            threat_types=threat_types,
            confidence=min(total_weight / 6.0, 1.0),
            findings=findings,
            reason="matched prompt-injection heuristics",
        )
