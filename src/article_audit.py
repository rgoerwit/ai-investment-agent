from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

import structlog

from src.claim_policy import MATERIAL_CLAIM_POLICIES, claim_source_context_fields
from src.data_block_utils import (
    extract_block_field_from_text,
    extract_block_field_from_text_raw,
    extract_last_data_block,
)
from src.text_patterns import SENTENCE_SPLIT_RE

logger = structlog.get_logger(__name__)

_HARD_CERTAINTY_TERMS = (
    "confirmed",
    "verified",
    "proven",
    "filing-confirmed",
    "contractually secured",
    "guaranteed",
)

_ARTICLE_CITATION_PATTERN = re.compile(r"`\(([A-Z][A-Z0-9_]+):\s*([^)]+?)\)`")
# Un-backticked parentheticals: the writer sometimes cites DATA_BLOCK keys as
# plain prose parentheses (3393.T shipped a hallucinated FIFTY_TWO_WEEK_LOW
# that way). Audited more conservatively than backticked citations: only keys
# that exist in the DATA_BLOCK, and only numeric-like values.
_BARE_PARENTHETICAL_PATTERN = re.compile(r"(?<!`)\(([^()`\n]+)\)")
# Value runs to the next comma unless the comma is a thousands separator
# (comma directly followed by a digit), so "1.95, a moderate level" cites
# 1.95 while "3,057M JPY" stays whole.
_BARE_PAIR_PATTERN = re.compile(r"\b([A-Z][A-Z0-9_]{2,}):\s*([^,]*(?:,\d[^,]*)*)")
_UNVERIFIED_TAG_PATTERN = re.compile(r"\s*\[unverified\]\s*$", re.IGNORECASE)
_CLAIM_USAGE_PATTERN = re.compile(
    r"\n*```CLAIM_USAGE\b\s*(.*?)```[ \t]*",
    re.DOTALL,
)
_SOURCE_CONFIDENCE_EXTRA_FIELDS = (
    "OPERATING_CASH_FLOW_SOURCE",
    "OCF_FILING_REASON",
    "GUIDANCE_SOURCE_TYPE",
    "GUIDANCE_MANAGEMENT_IDENTIFIED",
    "R_AND_D_CAPEX_BACKLOG_EVIDENCE",
    "R_AND_D_CAPEX_BACKLOG_EVIDENCE_ADJUSTMENT",
    "MOAT_CFO_NI_AVG",
    "MOAT_CFO_NI_YEARS",
    "MOAT_CFO_NI_SOURCE",
    "NET_CASH_TO_MARKET_CAP",
    "EARNINGS_GROWTH_FY_SOURCE",
    "MRQ_COMPARISON_BASE_STATUS",
    "ANALYST_COVERAGE_DATA_QUALITY_NOTE",
    "BALANCE_SHEET_DATA_QUALITY_NOTE",
    "GROWTH_DATA_QUALITY_NOTE",
    "PFIC_ASSET_NOTE",
    "LATEST_RESULTS_PERIOD",
    "LATEST_RESULTS_PRIOR_PERIOD",
    "LATEST_RESULTS_EARNINGS_SCOPE",
)


def _source_confidence_fields() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (*claim_source_context_fields(), *_SOURCE_CONFIDENCE_EXTRA_FIELDS)
        )
    )


def _clean_citation_text(value: str) -> str:
    """Shared surface cleaning for a citation literal (keeps the unit suffix)."""
    text = value.strip().strip("`").strip()
    text = _UNVERIFIED_TAG_PATTERN.sub("", text)
    # A DATA_BLOCK value may carry a trailing parenthetical qualifier the
    # article legitimately omits, e.g. "91.7% (based on 12 available points)".
    # Strip it on both sides of the comparison — but never strip a value that
    # is *only* a parenthetical (accounting-negative style "(3,057)").
    qualifier_stripped = re.sub(r"\s*\([^()]*\)\s*$", "", text)
    if qualifier_stripped:
        text = qualifier_stripped
    text = text.strip("'\"").replace(",", "")
    return re.sub(r"\s+", "", text)


def _normalize_citation_value(value: str) -> str:
    text = _clean_citation_text(value)
    if text.endswith("%"):
        text = text[:-1]
    try:
        return f"{float(text):.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return text.upper()


# Simple numeric citation: an optionally-signed number with at most one unit
# suffix. `%` and a bare number share a class (SCALAR) because
# _normalize_citation_value already treats them as equivalent; a multiple ("x")
# is a distinct class so "13.6x" can never match "13.6%".
_CITATION_NUMBER_RE = re.compile(r"^([+-]?\d+(?:\.(\d+))?)([%xX]?)$")


def _parse_citation_number(value: str) -> tuple[Decimal, str, int] | None:
    """Parse a plain numeric citation into (magnitude, unit_class, decimals).

    Returns None for anything that is not a plain number with at most one of the
    recognized suffixes, so the precision-aware fallback declines and the exact
    string comparison result stands.
    """
    match = _CITATION_NUMBER_RE.match(_clean_citation_text(value))
    if not match:
        return None
    number, fraction, suffix = match.group(1), match.group(2), match.group(3)
    try:
        magnitude = Decimal(number)
    except InvalidOperation:
        return None
    unit_class = "MULTIPLE" if suffix in {"x", "X"} else "SCALAR"
    return magnitude, unit_class, len(fraction) if fraction else 0


def _precision_aware_match(cited: str, actual: str) -> bool:
    """Allow legitimate display rounding of the canonical value to the cited one.

    Additive: only reached after exact matching fails. Matches only when both
    sides are plain numerics of the *same* unit class and the canonical value,
    quantized down to the cited value's display precision, equals the cited
    value. An article may round the canonical value (fewer decimals) but never
    fabricate precision or change magnitude/scale.
    """
    cited_parsed = _parse_citation_number(cited)
    actual_parsed = _parse_citation_number(actual)
    if cited_parsed is None or actual_parsed is None:
        return False
    cited_value, cited_class, cited_decimals = cited_parsed
    actual_value, actual_class, _ = actual_parsed
    if cited_class != actual_class:
        return False
    quantum = Decimal(10) ** -cited_decimals
    return actual_value.quantize(quantum, rounding=ROUND_HALF_UP) == cited_value


def _citation_values_match(cited: str, actual: str) -> bool:
    if _normalize_citation_value(cited) == _normalize_citation_value(actual):
        return True
    return _precision_aware_match(cited, actual)


def _first_number(value: str | None) -> float | None:
    match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value or "")
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def audit_article_citations(
    article: str,
    data_block_text: str | None,
) -> list[dict[str, str]]:
    """Return deterministic factual errors for article DATA_BLOCK citation drift."""
    block_text = extract_last_data_block(data_block_text)
    if not article or not data_block_text:
        return []
    if block_text is None:
        logger.warning("article_citation_audit_no_parseable_datablock")
        return []

    def _lookup(key: str) -> str | None:
        actual = extract_block_field_from_text(block_text, key)
        if actual is None:
            actual = extract_block_field_from_text_raw(block_text, key)
        return actual

    errors: list[dict[str, str]] = []
    for match in _ARTICLE_CITATION_PATTERN.finditer(article):
        key = match.group(1)
        cited = match.group(2).strip()
        actual = _lookup(key)
        if actual is None:
            errors.append(
                {
                    "location": "DATA_BLOCK citation audit",
                    "claim": f"Article cites ({key}: {cited})",
                    "ground_truth": f"No `{key}` field exists in DATA_BLOCK.",
                    "action": "Remove this citation or replace it with a real DATA_BLOCK key.",
                }
            )
        elif not _citation_values_match(cited, actual):
            errors.append(
                {
                    "location": "DATA_BLOCK citation audit",
                    "claim": f"Article cites ({key}: {cited})",
                    "ground_truth": f"DATA_BLOCK shows {key}: {actual}",
                    "action": "Correct the cited value and any narrative built on it.",
                }
            )

    for paren_match in _BARE_PARENTHETICAL_PATTERN.finditer(article):
        for pair_match in _BARE_PAIR_PATTERN.finditer(paren_match.group(1)):
            key = pair_match.group(1)
            cited = _UNVERIFIED_TAG_PATTERN.sub("", pair_match.group(2).strip())
            if not re.search(r"\d", cited):
                continue
            actual = _lookup(key)
            if actual is None or _citation_values_match(cited, actual):
                continue
            errors.append(
                {
                    "location": "DATA_BLOCK citation audit",
                    "claim": f"Article cites ({key}: {cited})",
                    "ground_truth": f"DATA_BLOCK shows {key}: {actual}",
                    "action": "Correct the cited value and any narrative built on it.",
                }
            )

    current_price = _first_number(_lookup("CURRENT_PRICE"))
    forward_eps = _first_number(_lookup("FORWARD_EPS"))
    forward_pe = _first_number(_lookup("PE_RATIO_FORWARD"))
    if (
        current_price is not None
        and forward_eps is not None
        and forward_pe is not None
        and current_price > 0
        and forward_eps > 0
        and forward_pe > 0
    ):
        implied_forward_pe = current_price / forward_eps
        if abs(implied_forward_pe - forward_pe) / forward_pe > 0.03:
            errors.append(
                {
                    "location": "Forward P/E identity audit",
                    "claim": (
                        f"PE_RATIO_FORWARD is {forward_pe:g} with CURRENT_PRICE "
                        f"{current_price:g} and FORWARD_EPS {forward_eps:g}."
                    ),
                    "ground_truth": (
                        "CURRENT_PRICE / FORWARD_EPS implies forward P/E "
                        f"{implied_forward_pe:.2f}."
                    ),
                    "action": (
                        "Reconcile the forward P/E inputs and do not attribute the "
                        "multiple to a different EPS estimate or provider."
                    ),
                }
            )
    return errors


def audit_article_claim_support(
    article: str,
    snapshot: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Reject unsupported registered operating claims after editorial review."""
    if not article or not snapshot or snapshot.get("contract_status") != "VALID":
        return []
    claims = snapshot.get("claims", {})
    sentences = SENTENCE_SPLIT_RE.split(article)
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for claim in claims.values():
        if not isinstance(claim, dict) or claim.get("decision_eligible"):
            continue
        field = str(claim.get("field") or "")
        policy = MATERIAL_CLAIM_POLICIES.get(field)
        if not policy or not policy.source_required or not policy.aliases:
            continue
        value = str(claim.get("value") or "")
        for sentence in sentences:
            folded = sentence.casefold()
            if not any(alias in folded for alias in policy.aliases):
                continue
            asserts_value = bool(value and value.casefold() in folded)
            asserts_certainty = any(
                re.search(rf"(?<!\w){re.escape(term)}(?!\w)", folded)
                for term in _HARD_CERTAINTY_TERMS
            )
            if not (asserts_value or asserts_certainty):
                continue
            if field in seen:
                break
            seen.add(field)
            errors.append(
                {
                    "location": "Canonical claim audit",
                    "claim": sentence.strip()[:500],
                    "ground_truth": (
                        f"{field} is {claim.get('coverage')} with authority "
                        f"{claim.get('authority')} and is not assertion-eligible."
                    ),
                    "action": "Remove the assertion or make the uncertainty explicit.",
                }
            )
            break
    return errors


def strip_claim_usage(article: str) -> str:
    """Remove the temporary claim-usage manifest before publication."""
    if not article or "```CLAIM_USAGE" not in article:
        return article
    return _CLAIM_USAGE_PATTERN.sub("\n", article).rstrip() + "\n"


def audit_article_claim_usage(
    article: str,
    snapshot: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """Validate source-sensitive prose against an explicit canonical claim manifest."""
    if not article or not snapshot or snapshot.get("contract_status") != "VALID":
        return []
    body = strip_claim_usage(article)
    match = _CLAIM_USAGE_PATTERN.search(article)
    claims = snapshot.get("claims", {})
    usage: dict[str, list[str]] = {}
    errors: list[dict[str, str]] = []
    if match:
        for raw_line in match.group(1).splitlines():
            line = raw_line.strip().removeprefix("-").strip()
            if not line:
                continue
            claim_id, separator, excerpt = line.partition("|")
            claim_id = claim_id.strip()
            excerpt = excerpt.strip()
            claim = claims.get(claim_id)
            if not separator or not excerpt or not isinstance(claim, dict):
                errors.append(
                    {
                        "location": "CLAIM_USAGE manifest",
                        "claim": raw_line.strip()[:500],
                        "ground_truth": "Each row must use a registered claim ID and exact article excerpt.",
                        "action": "Correct or remove the manifest row.",
                    }
                )
                continue
            if not claim.get("decision_eligible"):
                errors.append(
                    {
                        "location": "CLAIM_USAGE manifest",
                        "claim": raw_line.strip()[:500],
                        "ground_truth": f"{claim_id} is not assertion-eligible.",
                        "action": "Remove the assertion or state the uncertainty explicitly.",
                    }
                )
                continue
            if excerpt not in body:
                errors.append(
                    {
                        "location": "CLAIM_USAGE manifest",
                        "claim": raw_line.strip()[:500],
                        "ground_truth": "The quoted excerpt does not occur verbatim in the article body.",
                        "action": "Use an exact excerpt from the final article.",
                    }
                )
                continue
            usage.setdefault(str(claim.get("field") or ""), []).append(excerpt)

    sentences = SENTENCE_SPLIT_RE.split(body)
    for claim in claims.values():
        if not isinstance(claim, dict):
            continue
        field = str(claim.get("field") or "")
        policy = MATERIAL_CLAIM_POLICIES.get(field)
        if (
            not claim.get("decision_eligible")
            or not policy
            or not policy.source_required
            or not policy.aliases
        ):
            continue
        for sentence in sentences:
            folded = sentence.casefold()
            if not any(alias in folded for alias in policy.aliases):
                continue
            if not (
                re.search(r"\d", sentence)
                or any(term in folded for term in _HARD_CERTAINTY_TERMS)
            ):
                continue
            if not any(excerpt in sentence for excerpt in usage.get(field, [])):
                errors.append(
                    {
                        "location": "Canonical claim usage audit",
                        "claim": sentence.strip()[:500],
                        "ground_truth": (
                            f"Numeric or certainty-bearing {field} prose is not "
                            "bound to its canonical claim ID."
                        ),
                        "action": (
                            "Add an exact CLAIM_USAGE row for the canonical claim, "
                            "or remove/qualify the assertion."
                        ),
                    }
                )
            break
    return errors


def prepend_verification_caveats(
    article: str,
    factual_errors: list[dict[str, Any]],
) -> str:
    if not factual_errors or re.search(
        r"^## Verification Caveats\b", article, flags=re.MULTILINE
    ):
        return article

    lines = [
        "## Verification Caveats",
        "",
        "The following deterministic citation checks were still unresolved after editorial revision:",
    ]
    for error in factual_errors:
        lines.append(
            f"- {error.get('claim', 'Citation mismatch')}: "
            f"{error.get('ground_truth', 'No ground truth available')}"
        )
    caveat_block = "\n".join(lines)
    # The caveat block is QA scaffolding, not the lede — place it under the
    # article's own H1 title when one exists instead of above it.
    title_match = re.search(r"^# .+$", article, flags=re.MULTILINE)
    if title_match:
        end = title_match.end()
        return article[:end] + "\n\n" + caveat_block + "\n" + article[end:]
    return caveat_block + "\n\n" + article


def extract_source_confidence_context(
    data_block: str | None,
    consultant_review: str | None,
) -> str:
    block_text = extract_last_data_block(data_block)
    lines: list[str] = []
    field_values: dict[str, str] = {}

    if block_text:
        for field_name in _source_confidence_fields():
            value = extract_block_field_from_text(block_text, field_name)
            if value:
                field_values[field_name] = value
                lines.append(f"{field_name}: {value}")

    if consultant_review:
        for raw_line in consultant_review.splitlines():
            line = raw_line.strip()
            if re.search(r"\b(?:SPOT_CHECK|COVERAGE_GAP)\b", line, re.IGNORECASE):
                lines.append(line)

    if not lines:
        return ""

    lines.append(
        "Editor instruction: Do not describe weak-source or coverage-gap metrics as "
        "company-reported or filing-confirmed. Use qualified wording such as "
        "'aggregator-indicated' unless filing/IR support is explicit."
    )
    if field_values.get("GUIDANCE_SOURCE_AUTHORITY") in {"THIRD_PARTY", "UNKNOWN"}:
        lines.append(
            "Guidance instruction: Treat these figures as third-party estimates or "
            "unresolved sourcing, never as management/company guidance."
        )
    if field_values.get("CAPACITY_EVIDENCE_STATUS") in {
        "SECONDARY",
        "UNSUPPORTED",
        "UNKNOWN",
    }:
        lines.append(
            "Capacity instruction: Qualify secondary capacity/buildout claims; omit "
            "unsupported claims. If sourcing is secondary or its as-of date is "
            "unknown, keep every operating inference conditional rather than stating "
            "that the facility is currently near full utilization."
        )
    if field_values.get("FORWARD_EPS_SOURCE") or field_values.get(
        "PE_RATIO_FORWARD_SOURCE"
    ):
        lines.append(
            "Forward-multiple instruction: Forward P/E belongs to CURRENT_PRICE / "
            "FORWARD_EPS and their recorded sources. Never say it was computed from "
            "a different EPS estimate or provider."
        )
    if field_values.get("MOAT_CFO_NI_SOURCE") not in {None, "FILING", "PRIMARY"}:
        lines.append(
            "Cash-conversion instruction: Describe the multi-year ratio as "
            "aggregator-indicated and period-limited, not real/proven cash quality "
            "or manipulation-proof."
        )
    net_cash_ratio = _first_number(field_values.get("NET_CASH_TO_MARKET_CAP"))
    if net_cash_ratio is not None and net_cash_ratio < 10:
        lines.append(
            "Balance-sheet instruction: Net cash below 10% of market value is a "
            "modest cushion, not a valuation floor or backstop."
        )
    if field_values.get("EARNINGS_GROWTH_FY_SOURCE"):
        lines.append(
            "Trend instruction: One annual year-over-year comparison is a growth "
            "observation, not proof of a durable multi-year earnings trend."
        )
    if field_values.get("MRQ_COMPARISON_BASE_STATUS") == "DEPRESSED":
        lines.append(
            "Quarterly-growth instruction: Describe MRQ growth as base-sensitive, "
            "not structural acceleration, when the comparison base is depressed."
        )
    if field_values.get("ANALYST_COVERAGE_ENGLISH"):
        lines.append(
            "Coverage instruction: ANALYST_COVERAGE_ENGLISH is an aggregator "
            "analyst-opinion count, not proof of that many identifiable "
            "English-language analysts. Describe it as aggregator coverage unless "
            "analyst identities and source languages are separately established."
        )
    if field_values.get("LATEST_RESULTS_SOURCE_AUTHORITY") == "PRIMARY":
        lines.append(
            "Latest-results instruction: Treat LATEST_RESULTS_* as historical actual "
            "results in their stated scope and period. Keep MRQ metrics tied to their "
            "own period, and never present actual YoY growth as management guidance "
            "or projected growth."
        )
    growth_quality_note = field_values.get("GROWTH_DATA_QUALITY_NOTE", "")
    if any(
        marker in growth_quality_note
        for marker in (
            "Newer quarter metadata exists",
            "Newer primary results exist",
            "newer-period results candidate",
        )
    ):
        lines.append(
            "Period instruction: Keep MRQ growth tied to its stated period, and never "
            "call that statement period the latest reported quarter when newer-period "
            "evidence is identified."
        )
    lines.append(
        "Monitoring instruction: Copy numeric thesis-break or review thresholds "
        "exactly from the Portfolio Manager or Trader source text; otherwise omit "
        "them rather than synthesizing new cutoffs."
    )
    lines.append(
        "Valuation instruction: Use 'margin of safety' only when a downside anchor "
        "(bear value, asset floor, or normalized-earnings range) supports it."
    )
    return "=== SOURCE CONFIDENCE ===\n" + "\n".join(lines)
