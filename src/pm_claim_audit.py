"""Deterministic PM-claim audit (post-PM, pre-persistence).

Two gates over the Portfolio Manager decision text. Neither ever changes the verdict —
they append an advisory caveat block (same `>` note shape the verdict-policy hooks use)
and return structured log records.

- **2a number-consistency**: a ``(KEY: VALUE)`` / backticked citation whose KEY is a
  *hard* ground-truth DATA_BLOCK field and whose value contradicts the DATA_BLOCK →
  caveat. Non-hard keys and free-prose derived numbers (intrinsic value, upside %,
  weighted targets) are never caveated — a mismatch on a non-hard key is debug-logged.
  Deliberately NOT expanded to free-prose alias number-matching: that both misses the
  GTT overclaim (its number matches the DATA_BLOCK) and mis-fires on threshold/kill-
  criteria prose ("backlog coverage drops below 2.0 years").

- **2b hard/estimated provenance**: a field with WEAK provenance (underlying base is
  ``N/A``/absent while a derived value is shown, or the field carries a reliability
  marker) that the PM asserts as a HARD contractual/filing-confirmed fact. Fires ONLY
  on the conjunction — a weak-provenance signal AND a hard-certainty term in the *same
  sentence* as an alias for the field — which keeps false positives below the number-
  matching approach (the GTT `REVENUE_BACKLOG: N/A` + "contractually secured" case).

Reuses ``src.article_audit`` normalization and ``src.data_block_utils`` field lookup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from src.article_audit import _citation_values_match
from src.data_block_utils import (
    extract_block_field_from_text,
    extract_block_field_from_text_raw,
    extract_last_data_block,
)

logger = structlog.get_logger(__name__)

# --- 2a: hard ground-truth fields the PM must quote correctly -----------------
_HARD_FIELDS = frozenset(
    {
        "PE_RATIO_TTM",
        "ADJUSTED_HEALTH_SCORE",
        "ADJUSTED_GROWTH_SCORE",
        "CURRENT_PRICE",
        "ANALYST_COVERAGE_ENGLISH",
        "DE_RATIO",
        "US_REVENUE_PERCENT",
    }
)

# Backticked or bare ``(KEY: VALUE)`` citation. Value runs to the next comma unless
# the comma is a thousands separator (comma directly before a digit), mirroring the
# article auditor so "3,057M" stays whole while "16.9, cheap" cites 16.9.
_PM_CITATION_RE = re.compile(
    r"`?\(?\b([A-Z][A-Z0-9_]{2,}):\s*([^)\n,]*(?:,\d[^)\n,]*)*)\)?`?"
)

# --- 2b: provenance rules -----------------------------------------------------


@dataclass(frozen=True)
class ProvenanceRule:
    """A field that is 'soft' under a stated condition, keyed to prose aliases.

    Weakness is proven by exactly one of:
      - ``base`` is N/A/absent while ``derived`` carries a value (base-N/A-underlying);
      - ``weak_when_flag`` field is present with a weak reliability value.
    ``aliases`` are the prose terms a hard-certainty claim must sit next to.
    """

    aliases: tuple[str, ...]
    derived: str | None = None
    base: str | None = None
    field: str | None = None
    weak_when_flag: str | None = None


_PROVENANCE_RULES: tuple[ProvenanceRule, ...] = (
    # GTT: REVENUE_BACKLOG: N/A while REVENUE_BACKLOG_COVERAGE shows an estimated ratio.
    ProvenanceRule(
        derived="REVENUE_BACKLOG_COVERAGE",
        base="REVENUE_BACKLOG",
        aliases=(
            "backlog coverage",
            "revenue backlog",
            "year backlog",
            "backlog provides",
        ),
    ),
    # Reliability-flagged valuation asserted as hard fact.
    ProvenanceRule(
        field="PE_RATIO_TTM",
        weak_when_flag="VALUATION_INPUT_RELIABILITY",
        aliases=("p/e", "pe of", "price-to-earnings", "earnings multiple", "valuation"),
    ),
)

_HARD_CERTAINTY_TERMS = (
    "contractually secured",
    "filing-confirmed",
    "filing confirmed",
    "guaranteed",
    "locked-in",
    "locked in",
    "ground truth",
    "confirmed",
    "certain",
    "verified",
)

# A field value or its DATA_BLOCK line counts as an estimate/range when it carries one
# of these markers (used as a secondary weak-provenance signal).
_ESTIMATE_MARKERS = ("estimat", "approx", "~", "roughly", "circa", "≈")
_WEAK_RELIABILITY_VALUES = (
    "LOW",
    "MEDIUM",
    "QUARANTINE",
    "SUSPECT",
    "UNRELIABLE",
    "UNVERIFIED",
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NA_VALUES = frozenset({"N/A", "NA", "NONE", "UNKNOWN", ""})


def _is_na(value: str | None) -> bool:
    return value is None or value.strip().upper().rstrip(".") in _NA_VALUES


def _lookup(block_text: str, key: str) -> str | None:
    actual = extract_block_field_from_text(block_text, key)
    if actual is None:
        actual = extract_block_field_from_text_raw(block_text, key)
    return actual


def _audit_number_consistency(
    pm_output: str, block_text: str, ticker: str
) -> list[dict[str, str]]:
    """2a: caveat hard-field citation mismatches; debug-log non-hard mismatches."""
    caveats: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in _PM_CITATION_RE.finditer(pm_output):
        key = match.group(1)
        cited = match.group(2).strip()
        if not cited or not re.search(r"\d", cited):
            continue
        actual = _lookup(block_text, key)
        if actual is None or _citation_values_match(cited, actual):
            continue
        if key not in _HARD_FIELDS:
            logger.debug(
                "pm_claim_nonhard_mismatch", ticker=ticker, key=key, cited=cited
            )
            continue
        dedup = (key, cited)
        if dedup in seen:
            continue
        seen.add(dedup)
        caveats.append(
            {
                "claim": f"PM cites {key}: {cited}",
                "ground_truth": f"DATA_BLOCK shows {key}: {actual}",
            }
        )
    return caveats


def _sentences_with_alias_and_certainty(
    pm_output: str, aliases: tuple[str, ...]
) -> bool:
    """True when one sentence contains both an alias and a hard-certainty term."""
    for sentence in _SENTENCE_SPLIT_RE.split(pm_output):
        low = sentence.lower()
        if any(alias in low for alias in aliases) and any(
            term in low for term in _HARD_CERTAINTY_TERMS
        ):
            return True
    return False


def _rule_is_weak(rule: ProvenanceRule, block_text: str) -> tuple[bool, str, str]:
    """Return (weak, shown_value, why) for a provenance rule against the DATA_BLOCK."""
    if rule.base is not None and rule.derived is not None:
        base_val = _lookup(block_text, rule.base)
        derived_val = _lookup(block_text, rule.derived)
        if derived_val and not _is_na(derived_val) and _is_na(base_val):
            return True, derived_val, f"underlying {rule.base} is N/A"
        # Secondary: derived value literally reads as an estimate/range.
        if derived_val and any(m in derived_val.lower() for m in _ESTIMATE_MARKERS):
            return True, derived_val, "value is estimated/approximate"
    if rule.field is not None and rule.weak_when_flag is not None:
        flag_val = _lookup(block_text, rule.weak_when_flag)
        shown = _lookup(block_text, rule.field) or ""
        if flag_val and flag_val.strip().upper() in _WEAK_RELIABILITY_VALUES:
            return True, shown, f"{rule.weak_when_flag} is {flag_val.strip()}"
    return False, "", ""


def _audit_provenance(
    pm_output: str, block_text: str, ticker: str
) -> list[dict[str, str]]:
    """2b: caveat weak-provenance fields asserted with hard-certainty language."""
    caveats: list[dict[str, str]] = []
    for rule in _PROVENANCE_RULES:
        weak, shown, why = _rule_is_weak(rule, block_text)
        if not weak:
            continue
        if not _sentences_with_alias_and_certainty(pm_output, rule.aliases):
            continue
        field_name = rule.derived or rule.field or ""
        caveats.append(
            {
                "claim": f"PM asserts {field_name} ({shown}) as a hard, secured fact",
                "ground_truth": (
                    f"{field_name} has weak provenance ({why}); treat the "
                    "'secured/confirmed' framing as unverified."
                ),
            }
        )
        logger.info(
            "pm_claim_provenance_caveat",
            ticker=ticker,
            field=field_name,
            reason=why,
        )
    return caveats


def _format_caveat_block(caveats: list[dict[str, str]]) -> str:
    lines = [
        "\n\n> **PM CLAIM CAVEAT — unverified/overstated figure(s) flagged deterministically**"
    ]
    for c in caveats:
        lines.append(f"> - {c['claim']} — {c['ground_truth']}")
    return "\n".join(lines) + "\n"


def audit_pm_claims(
    pm_output: str,
    *,
    fundamentals: str | None,
    valuation_params: str | None = None,
    ticker: str = "UNKNOWN",
) -> tuple[str, list[dict[str, str]]]:
    """Audit the PM decision text; append a caveat block if anything fires.

    Returns ``(pm_output, caveats)``. ``caveats`` is the list of fired items (empty
    when nothing fired). Idempotent: never appends a second caveat block. Never
    modifies the verdict token.
    """
    if not pm_output or not fundamentals:
        return pm_output, []
    if "PM CLAIM CAVEAT" in pm_output:
        return pm_output, []
    block_text = extract_last_data_block(fundamentals)
    if block_text is None:
        logger.debug("pm_claim_audit_no_datablock", ticker=ticker)
        return pm_output, []

    caveats = _audit_number_consistency(pm_output, block_text, ticker)
    caveats += _audit_provenance(pm_output, block_text, ticker)
    if not caveats:
        return pm_output, []
    return pm_output.rstrip() + _format_caveat_block(caveats), caveats
