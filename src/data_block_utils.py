from __future__ import annotations

import re

import structlog

logger = structlog.get_logger(__name__)


def _compile_named_block_pattern(block_name: str) -> re.Pattern[str]:
    escaped_name = re.escape(block_name)
    return re.compile(
        rf"### --- START {escaped_name}[^\n]*---(.+?)### --- END {escaped_name} ---",
        re.DOTALL,
    )


DATA_BLOCK_PATTERN = _compile_named_block_pattern("DATA_BLOCK")
_LEGACY_DATA_BLOCK_HEADER_PATTERN = re.compile(r"(?m)^### DATA_BLOCK\b[^\n]*$")
_DASHED_DATA_BLOCK_HEADER_PATTERN = re.compile(r"(?m)^### --- DATA_BLOCK ---[ \t]*$")
_DASHED_DATA_BLOCK_END_PATTERN = re.compile(r"(?m)^### --- END DATA_BLOCK ---")
_SECTION_HEADER_PATTERN = re.compile(r"(?m)^### ")
_TABLE_ROW_PATTERN = re.compile(r"^\|(.+)\|$")
_STRUCTURED_BLOCK_BOUNDARY_NAMES = ("DATA_BLOCK", "PM_BLOCK")
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


def _extract_legacy_data_block_body(report: str) -> str | None:
    legacy_match = _LEGACY_DATA_BLOCK_HEADER_PATTERN.search(report)
    if not legacy_match:
        return None

    body_start = legacy_match.end()
    next_section = _SECTION_HEADER_PATTERN.search(report, body_start)
    body_end = next_section.start() if next_section else len(report)
    body = report[body_start:body_end].strip()
    return body or None


def _extract_dashed_data_block_body(report: str) -> tuple[str, int, int] | None:
    dashed_match = _DASHED_DATA_BLOCK_HEADER_PATTERN.search(report)
    if not dashed_match:
        return None

    body_start = dashed_match.end()
    end_match = _DASHED_DATA_BLOCK_END_PATTERN.search(report, body_start)
    if not end_match:
        return None

    body_end = end_match.start()
    body = report[body_start:body_end].strip()
    if not body or _count_likely_keys(body) < 4:
        return None

    return body, dashed_match.start(), end_match.end()


def _count_likely_keys(body: str) -> int:
    return sum(1 for key in _LIKELY_DATA_BLOCK_KEYS if re.search(rf"(?m)^{key}:", body))


def _parse_legacy_key_value_body(body: str) -> str | None:
    if _count_likely_keys(body) < 4:
        return None
    return body


def _is_alignment_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def _parse_legacy_table_body(body: str) -> str | None:
    rows: list[tuple[str, str]] = []
    saw_table_row = False

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row_match = _TABLE_ROW_PATTERN.fullmatch(line)
        if not row_match:
            return None

        saw_table_row = True
        cells = [cell.strip() for cell in row_match.group(1).split("|")]
        if len(cells) != 2:
            return None
        if [cell.lower() for cell in cells] == ["metric", "value"]:
            continue
        if _is_alignment_row(cells):
            continue

        key, value = cells
        if not key or not value:
            return None
        rows.append((key, value))

    if not saw_table_row or not rows:
        return None

    normalized = "\n".join(f"{key}: {value}" for key, value in rows)
    if _count_likely_keys(normalized) < 4:
        return None
    return normalized


def detect_legacy_data_block_shape(report: str | None) -> str | None:
    """Return the recognized malformed DATA_BLOCK family, if any."""
    if not report or not isinstance(report, str):
        return None

    body = _extract_legacy_data_block_body(report)
    if not body:
        return None
    if _parse_legacy_key_value_body(body):
        return "colon"
    if _parse_legacy_table_body(body):
        return "table"
    dashed_body = _extract_dashed_data_block_body(report)
    if dashed_body:
        return "dashed"
    return None


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
    block = extract_last_fenced_block(
        report, "DATA_BLOCK", include_markers=include_markers
    )
    if block is not None:
        return block

    normalized_report = normalize_legacy_data_block_report(report)
    if normalized_report == report:
        return None
    return extract_last_fenced_block(
        normalized_report, "DATA_BLOCK", include_markers=include_markers
    )


def has_parseable_data_block(report: str | None) -> bool:
    """Return True only when a fenced DATA_BLOCK can actually be parsed."""
    return extract_last_data_block(report, include_markers=True) is not None


def normalize_structured_block_boundaries(report: str | None) -> str | None:
    """Insert missing line breaks after known structured block end markers.

    This only repairs the narrow glued-heading defect where a recognized end
    marker is immediately followed by the next markdown heading, e.g.
    ``### --- END DATA_BLOCK ---### FINANCIAL HEALTH DETAIL``.
    """
    if not report or not isinstance(report, str):
        return report

    normalized = report
    for block_name in _STRUCTURED_BLOCK_BOUNDARY_NAMES:
        normalized = re.sub(
            rf"(### --- END {re.escape(block_name)} ---)(?=###\s)",
            r"\1\n\n",
            normalized,
        )
    return normalized


def normalize_legacy_data_block_report(report: str | None) -> str | None:
    """Repair the exact legacy ``### DATA_BLOCK`` shape into fenced format.

    This keeps downstream parsing strict while recovering a known LLM format drift
    from the Fundamentals Analyst. Narrative mentions like ``DATA_BLOCK:`` remain
    untouched.
    """
    if (
        not report
        or not isinstance(report, str)
        or has_parseable_fenced_block(report, "DATA_BLOCK")
    ):
        return report

    legacy_match = _LEGACY_DATA_BLOCK_HEADER_PATTERN.search(report)
    if legacy_match:
        body = _extract_legacy_data_block_body(report)
        if body:
            normalized_body = _parse_legacy_key_value_body(
                body
            ) or _parse_legacy_table_body(body)
            if normalized_body:
                logger.info(
                    "data_block_repaired",
                    repair_kind="legacy_subtitle",
                    repair_confidence="medium",
                    key_count=_count_likely_keys(normalized_body),
                )
                repaired_block = (
                    "### --- START DATA_BLOCK ---\n"
                    f"{normalized_body}\n"
                    "### --- END DATA_BLOCK ---"
                )
                body_start = legacy_match.end()
                next_section = _SECTION_HEADER_PATTERN.search(report, body_start)
                body_end = next_section.start() if next_section else len(report)
                return (
                    report[: legacy_match.start()] + repaired_block + report[body_end:]
                )

    dashed_data = _extract_dashed_data_block_body(report)
    if not dashed_data:
        return report

    body, block_start, block_end = dashed_data
    logger.info(
        "data_block_repaired",
        repair_kind="dashed_header",
        repair_confidence="high",
        key_count=_count_likely_keys(body),
    )
    repaired_block = (
        "### --- START DATA_BLOCK ---\n" f"{body}\n" "### --- END DATA_BLOCK ---"
    )
    return report[:block_start] + repaired_block + report[block_end:]
