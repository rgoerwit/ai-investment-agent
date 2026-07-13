from __future__ import annotations

import re
from typing import Any

import structlog

from src.data_block_utils import (
    extract_block_field_from_text,
    extract_block_field_from_text_raw,
    extract_last_data_block,
)

logger = structlog.get_logger(__name__)

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
_SOURCE_CONFIDENCE_FIELDS = (
    "OPERATING_CASH_FLOW_SOURCE",
    "OCF_FILING_REASON",
    "ANALYST_COVERAGE_DATA_QUALITY_NOTE",
    "BALANCE_SHEET_DATA_QUALITY_NOTE",
    "GROWTH_DATA_QUALITY_NOTE",
    "PFIC_ASSET_NOTE",
)


def _normalize_citation_value(value: str) -> str:
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
    text = re.sub(r"\s+", "", text)
    if text.endswith("%"):
        text = text[:-1]
    try:
        return f"{float(text):.10f}".rstrip("0").rstrip(".")
    except ValueError:
        return text.upper()


def _citation_values_match(cited: str, actual: str) -> bool:
    return _normalize_citation_value(cited) == _normalize_citation_value(actual)


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

    if block_text:
        for field_name in _SOURCE_CONFIDENCE_FIELDS:
            value = extract_block_field_from_text(block_text, field_name)
            if value:
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
    return "=== SOURCE CONFIDENCE ===\n" + "\n".join(lines)
