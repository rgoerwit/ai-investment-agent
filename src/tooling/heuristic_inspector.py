"""Pattern-based context-pollution detector.

Implements ``ContentInspector`` for the ``"python"`` backend config slot.
Detects common injection signals — override phrases, role-play coercion,
delimiter breakout, hidden markup, and control-character abuse — using
scored pattern families.

Heuristics are a fast, zero-cost first pass.  For semantic coverage
(paraphrasing, multilingual attacks, context-dependent injection) use
``EscalatingInspector`` which chains this with an LLM judge. Structured
sources such as filings and financial APIs are treated more lightly, but
their free-text fields are still not trusted.
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
    "formatting_chars",
    "encoded_payload",
    "exfiltration",
    "control_chars",
    "context_bomb",
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
    _Signal(
        re.compile(r"(?:system|admin)\s+(?:notification|alert|message)\s*:", re.I),
        2.0,
        "override",
    ),
    _Signal(
        re.compile(r"user\s+has\s+(?:authorized|approved|confirmed)\b", re.I),
        2.0,
        "override",
    ),
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
    # --- Exfiltration / persistence / looping instructions ---
    _Signal(
        re.compile(
            r"(?:send|post|upload|transmit)\s+(?:the\s+)?(?:data|results|output|report)\s+to\b",
            re.I,
        ),
        2.0,
        "exfiltration",
    ),
    _Signal(
        re.compile(
            r"(?:include|append|add)\s+(?:the\s+)?(?:system\s+prompt|api\s+key|credentials?|token)\b",
            re.I,
        ),
        3.0,
        "exfiltration",
    ),
    _Signal(
        re.compile(
            r"(?:reveal|output|print|expose)\s+(?:the\s+)?(?:system\s+prompt|api\s+key|credentials?|token)\b",
            re.I,
        ),
        3.0,
        "exfiltration",
    ),
    _Signal(
        re.compile(
            r"(?:save|store|remember|memorize)\s+(?:this|the\s+following)\s+(?:for\s+)?(?:future|later|next)\b",
            re.I,
        ),
        2.0,
        "override",
    ),
    _Signal(
        re.compile(
            r"(?:keep|continue)\s+(?:calling|searching|fetching|querying)\s+(?:until|for)\b",
            re.I,
        ),
        1.5,
        "override",
    ),
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
_CONTEXT_BOMB_THRESHOLD = 15_000
_FORMATTING_CHAR_MIN_SUSPICIOUS_COUNT = 3
_FORMATTING_CHAR_WEIGHT = 1.5
_FORMATTING_CHARS_PATTERN = re.compile(
    r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff]"
)

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
        if sig.pattern.pattern == r"</search_results>":
            continue
        m = sig.pattern.search(text)
        if m:
            hits.append(_Hit(signal=sig, match_text=m.group()[:120]))
    return hits


def _detect_formatting_char_artifact(text: str) -> _Hit | None:
    """Flag clusters of bidi / zero-width / BOM characters.

    These characters have no semantic value for an LLM but appear naturally in
    CJK web text (bidi marks for mixed-script rendering, NNBSP in numerals,
    BOM at file boundaries). The sanitize path scrubs them rather than block
    so legitimate Foreign Language Analyst output isn't discarded — see the
    May 2026 HK ticker false-positive incident.
    """
    matches = list(_FORMATTING_CHARS_PATTERN.finditer(text))
    if len(matches) < _FORMATTING_CHAR_MIN_SUSPICIOUS_COUNT:
        return None

    preview = (
        "".join(match.group() for match in matches[:8])
        .encode("unicode_escape")
        .decode("ascii")
    )
    return _Hit(
        signal=_Signal(
            _FORMATTING_CHARS_PATTERN, _FORMATTING_CHAR_WEIGHT, "formatting_chars"
        ),
        match_text=f"{preview} (count={len(matches)})",
    )


def _search_results_closer_signal() -> _Signal:
    for sig in _SIGNALS:
        if sig.pattern.pattern == r"</search_results>":
            return sig
    raise RuntimeError("search_results delimiter signal missing")


def _detect_search_results_breakouts(text: str) -> tuple[list[_Hit], bool]:
    """Return unmatched-closer hits and whether the terminal closer is legitimate.

    Multiple stacked ``<search_results>...</search_results>`` blocks are valid:
    callers that merge several Tavily searches (e.g. ``get_news`` general+local,
    ``research.search_foreign_sources`` merged native-language results) emit a
    sequence of well-formed wrappers concatenated with intervening text. A
    closer is only treated as a breakout if no preceding opener is unmatched.

    The second return value is ``True`` iff the *terminal* closer (last
    ``</search_results>`` with only whitespace following it) is matched by an
    opener — used downstream by the sanitize path to preserve that legitimate
    footer when stripping any embedded breakout closers.
    """
    closers = list(re.finditer(r"</search_results>", text, re.I))
    if not closers:
        return [], False

    openers = list(re.finditer(r"<search_results\b[^>]*>", text, re.I))

    # The *terminal* closer is the last one with only whitespace following it.
    # Treated as the wrapper's legitimate footer iff at least one opener
    # precedes it. Earlier closers are evaluated under stricter trust-boundary
    # semantics below — a closer mid-content is suspicious unless it's clearly
    # the seam between two stacked wrapped blocks.
    last_closer = closers[-1]
    terminal_match: re.Match[str] | None = (
        last_closer if text[last_closer.end() :].strip() == "" else None
    )
    terminal_legit = terminal_match is not None and any(
        o.start() < terminal_match.start() for o in openers
    )

    def _is_inner_wrapper_seam(closer: re.Match[str]) -> bool:
        """A non-terminal closer is legitimate only if it's the seam between
        a properly-paired wrapped block and the *next* wrapped block: there
        must be at least one preceding opener available to pair with it, and
        an immediately following opener with no intervening closer."""
        openers_before = sum(1 for o in openers if o.start() < closer.start())
        closers_up_to = sum(1 for c in closers if c.start() <= closer.start())
        if openers_before < closers_up_to:
            return False  # not balanced — extra closer
        next_opener = next((o for o in openers if o.start() > closer.end()), None)
        if next_opener is None:
            return False
        for c in closers:
            if closer.end() < c.start() < next_opener.start():
                return False
        return True

    signal = _search_results_closer_signal()
    hits: list[_Hit] = []
    for c in closers:
        if c is terminal_match and terminal_legit:
            continue
        if c is not terminal_match and _is_inner_wrapper_seam(c):
            continue
        hits.append(_Hit(signal=signal, match_text=c.group()[:120]))

    return hits, terminal_legit


def _detect_context_bomb(
    text: str,
    source_kind: SourceKind,
    metadata: dict[str, object] | None,
) -> _Hit | None:
    """Flag oversized low-value payloads before they dominate prompt budget."""
    if source_kind in _LIGHT_TREATMENT_SOURCES:
        return None

    original_length = 0
    if metadata:
        raw_original_length = metadata.get("original_length")
        if isinstance(raw_original_length, int):
            original_length = raw_original_length

    observed_length = max(len(text), original_length)
    if observed_length <= _CONTEXT_BOMB_THRESHOLD:
        return None

    unique_chars = len(set(text))
    if unique_chars >= 20:
        return None

    tokens = re.findall(r"\S+", text[:4000])
    unique_tokens = len(set(tokens))
    if unique_tokens > 3:
        return None

    return _Hit(
        signal=_Signal(re.compile(r"$^"), 2.0, "context_bomb"),
        match_text=(
            "low-entropy oversized payload "
            f"({observed_length} chars, {unique_chars} unique chars, "
            f"{unique_tokens} unique tokens)"
        ),
    )


def _control_char_density(text: str) -> float:
    """Return fraction of invisible / control characters in *text*.

    Excludes bidi / zero-width / BOM marks already covered by
    ``_FORMATTING_CHARS_PATTERN`` — those are attributed to the
    ``formatting_chars`` signal so we don't double-count them. Without this,
    a short Arabic / Hebrew / CJK passage with a few legitimate bidi marks
    can cross the 3% control-char density threshold and be misclassified as
    suspicious.
    """
    if len(text) < _CONTROL_CHAR_MIN_LENGTH:
        return 0.0
    count = 0
    for ch in text:
        if not unicodedata.category(ch).startswith("C"):
            continue
        if ch in ("\n", "\r", "\t"):
            continue
        if _FORMATTING_CHARS_PATTERN.match(ch):
            continue
        count += 1
    return count / len(text)


def _strip_known_breakouts(
    text: str,
    *,
    preserve_terminal_search_results: bool = False,
    strip_formatting_chars: bool = False,
) -> str:
    """Remove delimiter-breakout tags (and optionally inert formatting chars).

    Bidi / zero-width / BOM characters carry no semantic value for the LLM
    but appear naturally in CJK web text. ``strip_formatting_chars=True``
    scrubs them along with any strippable delimiters.
    """
    result = text
    sentinel = "__EXPECTED_SEARCH_RESULTS_FOOTER__"
    if preserve_terminal_search_results:
        result = re.sub(
            r"</search_results>\s*$",
            sentinel,
            result,
            count=1,
            flags=re.I,
        )
    for pat in _STRIPPABLE_DELIMITERS:
        result = pat.sub("", result)
    if strip_formatting_chars:
        result = _FORMATTING_CHARS_PATTERN.sub("", result)
    if preserve_terminal_search_results:
        result = result.replace(sentinel, "</search_results>")
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
        search_result_hits, expected_search_results_wrapper = (
            _detect_search_results_breakouts(text)
        )
        hits.extend(search_result_hits)
        context_bomb_hit = _detect_context_bomb(
            text,
            envelope.source_kind,
            envelope.metadata,
        )
        if context_bomb_hit is not None:
            hits.append(context_bomb_hit)
        formatting_char_hit = _detect_formatting_char_artifact(text)
        if formatting_char_hit is not None:
            hits.append(formatting_char_hit)

        if not hits and not cc_hit:
            return InspectionDecision(action="allow", threat_level="safe")

        total_weight = sum(h.signal.weight for h in hits)
        if cc_hit:
            total_weight += 2.0

        # Adjust weight for structured MCP output
        if envelope.source_kind == SourceKind.mcp_tool_output:
            payload_profile = (envelope.metadata or {}).get(
                "payload_profile", "free_text"
            )
            trust_tier = (envelope.metadata or {}).get("trust_tier", "unknown")
            if (
                payload_profile == "structured_financial"
                and trust_tier == "official_vendor"
            ):
                total_weight *= 0.5

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

        # Can we safely sanitize? Only if every hit is a scrubbable artifact —
        # delimiter breakouts (Tavily wrappers etc.) and inert formatting chars
        # (bidi marks, zero-width, BOM) both qualify. They convey no LLM-
        # actionable instruction; stripping is preferable to discarding the
        # whole tool output.
        scrubbable_types = {"delimiter_breakout", "formatting_chars"}
        all_scrubbable = (
            all(h.signal.threat_type in scrubbable_types for h in hits) and not cc_hit
        )
        if all_scrubbable:
            preserve_terminal_wrapper = expected_search_results_wrapper and bool(
                search_result_hits
            )
            strip_fmt = any(h.signal.threat_type == "formatting_chars" for h in hits)
            sanitized = _strip_known_breakouts(
                text,
                preserve_terminal_search_results=preserve_terminal_wrapper,
                strip_formatting_chars=strip_fmt,
            )
            reason_parts = []
            if any(h.signal.threat_type == "delimiter_breakout" for h in hits):
                reason_parts.append("delimiter-breakout tags")
            if strip_fmt:
                reason_parts.append("inert formatting characters")
            return InspectionDecision(
                action="sanitize",
                threat_level=severity,
                threat_types=threat_types,
                sanitized_content=sanitized,
                findings=findings,
                reason=f"stripped {' + '.join(reason_parts)}",
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
