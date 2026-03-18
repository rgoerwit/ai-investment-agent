from __future__ import annotations

import re


def _compile_named_block_pattern(block_name: str) -> re.Pattern[str]:
    escaped_name = re.escape(block_name)
    return re.compile(
        rf"### --- START {escaped_name}[^\n]*---(.+?)### --- END {escaped_name} ---",
        re.DOTALL,
    )


DATA_BLOCK_PATTERN = _compile_named_block_pattern("DATA_BLOCK")
_LEGACY_DATA_BLOCK_HEADER_PATTERN = re.compile(r"(?m)^### DATA_BLOCK\s*$")
_SECTION_HEADER_PATTERN = re.compile(r"(?m)^### ")
_LIKELY_DATA_BLOCK_KEYS = (
    "SECTOR",
    "RAW_HEALTH_SCORE",
    "ADJUSTED_HEALTH_SCORE",
    "RAW_GROWTH_SCORE",
    "ADJUSTED_GROWTH_SCORE",
    "US_REVENUE_PERCENT",
    "ANALYST_COVERAGE_ENGLISH",
    "PE_RATIO_TTM",
    "ADR_EXISTS",
    "IBKR_ACCESSIBILITY",
    "PFIC_RISK",
)


def extract_last_fenced_block(
    report: str | None,
    block_name: str,
    *,
    include_markers: bool = False,
) -> str | None:
    """Return the last parseable fenced block for the given structured block name."""
    if not report or not isinstance(report, str):
        return None

    blocks = list(_compile_named_block_pattern(block_name).finditer(report))
    if not blocks:
        return None

    last = blocks[-1]
    return last.group(0 if include_markers else 1)


def has_parseable_fenced_block(report: str | None, block_name: str) -> bool:
    """Return True only when the named fenced block can actually be parsed."""
    return (
        extract_last_fenced_block(report, block_name, include_markers=True) is not None
    )


def extract_last_data_block(
    report: str | None, *, include_markers: bool = False
) -> str | None:
    """Return the last parseable fenced DATA_BLOCK, if present."""
    return extract_last_fenced_block(
        report, "DATA_BLOCK", include_markers=include_markers
    )


def has_parseable_data_block(report: str | None) -> bool:
    """Return True only when a fenced DATA_BLOCK can actually be parsed."""
    return has_parseable_fenced_block(report, "DATA_BLOCK")


def normalize_legacy_data_block_report(report: str | None) -> str | None:
    """Repair the exact legacy ``### DATA_BLOCK`` shape into fenced format.

    This keeps downstream parsing strict while recovering a known LLM format drift
    from the Fundamentals Analyst. Narrative mentions like ``DATA_BLOCK:`` remain
    untouched.
    """
    if not report or not isinstance(report, str) or has_parseable_data_block(report):
        return report

    legacy_match = _LEGACY_DATA_BLOCK_HEADER_PATTERN.search(report)
    if not legacy_match:
        return report

    body_start = legacy_match.end()
    next_section = _SECTION_HEADER_PATTERN.search(report, body_start)
    body_end = next_section.start() if next_section else len(report)
    body = report[body_start:body_end].strip()
    if not body:
        return report

    present_keys = sum(
        1 for key in _LIKELY_DATA_BLOCK_KEYS if re.search(rf"(?m)^{key}:", body)
    )
    if present_keys < 4:
        return report

    repaired_block = (
        "### --- START DATA_BLOCK ---\n" f"{body}\n" "### --- END DATA_BLOCK ---"
    )
    repaired_report = (
        report[: legacy_match.start()] + repaired_block + report[body_end:]
    )
    return repaired_report
