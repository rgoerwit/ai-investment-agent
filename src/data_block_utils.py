from __future__ import annotations

import re
from enum import Enum


class BlockShape(Enum):
    """Structured block delimiter family."""

    FENCED = "fenced"
    UNFENCED = "unfenced"


_FENCED_START = "### --- START {name} ---"
_FENCED_END = "### --- END {name} ---"


def fenced_start(name: str) -> str:
    """Return the canonical emitted fenced-block start marker."""
    return _FENCED_START.format(name=name)


def fenced_end(name: str) -> str:
    """Return the canonical emitted fenced-block end marker."""
    return _FENCED_END.format(name=name)


def build_fenced_block(name: str, body: str) -> str:
    """Return a canonical fenced structured block."""
    return f"{fenced_start(name)}\n{body}\n{fenced_end(name)}"


def unfenced_label(name: str) -> str:
    """Return the canonical label for an unfenced structured block."""
    return f"{name}:"


BLOCK_SHAPES: dict[str, BlockShape] = {
    "DATA_BLOCK": BlockShape.FENCED,
    "PM_BLOCK": BlockShape.FENCED,
    "VALUE_TRAP_BLOCK": BlockShape.FENCED,
    "VALUATION_PARAMS": BlockShape.FENCED,
    "VALUATION_SCENARIOS": BlockShape.FENCED,
    "KILL_CRITERIA": BlockShape.FENCED,
    "TRADE_BLOCK": BlockShape.UNFENCED,
    "FORENSIC_DATA_BLOCK": BlockShape.UNFENCED,
    "MACRO_REGIME_BLOCK": BlockShape.UNFENCED,
    "CONSULTANT_RESOLUTION": BlockShape.UNFENCED,
    "APAC_RESOLUTION": BlockShape.UNFENCED,
    "AUDITOR_RESOLUTION": BlockShape.UNFENCED,
}

FENCED_BLOCK_NAMES = tuple(
    name for name, shape in BLOCK_SHAPES.items() if shape is BlockShape.FENCED
)
UNFENCED_BLOCK_NAMES = tuple(
    name for name, shape in BLOCK_SHAPES.items() if shape is BlockShape.UNFENCED
)
_FENCED_NAMES_ALT = "|".join(re.escape(name) for name in FENCED_BLOCK_NAMES)
_GLUED_FENCED_MARKER_RE = re.compile(
    r"(?m)^(?P<pre>.*[^\s#-])[ \t]*"
    rf"(?P<marker>#{{2,}}[ \t]*-{{2,}}[ \t]*(?:START|END)[ \t]+"
    rf"(?:{_FENCED_NAMES_ALT})\b[^\n]*)$"
)

_NULL_TOKENS = frozenset({"N/A", "NA", "NONE", "-", ""})
_FIELD_VALUE_PATTERN = r"(?m)^{field_name}:\s*(.+?)\s*$"
_NUMBER_TOKEN_PATTERN = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _compile_named_block_pattern(block_name: str) -> re.Pattern[str]:
    start_fragment = fenced_marker_fragment(block_name, "START")
    end_fragment = fenced_marker_fragment(block_name, "END")
    return re.compile(
        rf"(?m)^{start_fragment}\s*$" rf"(.+?)^{end_fragment}\s*$",
        re.DOTALL | re.MULTILINE,
    )


def fenced_marker_fragment(block_name: str, edge: str) -> str:
    """Return the shared tolerant regex fragment for a fenced block marker.

    Emitters use exactly three hashes. Matchers deliberately accept two or more
    hashes and two or more dashes so repair/strip paths catch common LLM
    heading drift.
    """
    normalized_edge = edge.upper()
    if normalized_edge not in {"START", "END"}:
        raise ValueError("edge must be START or END")
    escaped_name = re.escape(block_name)
    return rf"[ \t]*#{{2,}}\s*-{{2,}}\s*{normalized_edge}\s+{escaped_name}\b[^\n]*"


def fenced_block_pattern(block_name: str) -> re.Pattern[str]:
    """Return the shared tolerant regex for a named fenced block."""
    return _compile_named_block_pattern(block_name)


DATA_BLOCK_PATTERN = _compile_named_block_pattern("DATA_BLOCK")
_LEGACY_DATA_BLOCK_HEADER_PATTERN = re.compile(
    r"(?m)^#{2,} DATA_BLOCK(?:\s*\([^\n]*\))?\s*$"
)
_DASHED_DATA_BLOCK_HEADER_PATTERN = re.compile(
    r"(?m)^#{2,}\s*-{2,}\s*DATA_BLOCK\s*-{2,}\s*$"
)
# Intentionally narrow: this pairs only with the legacy ### --- DATA_BLOCK --- opener.
_EXPLICIT_DATA_BLOCK_END_PATTERN = re.compile(
    r"(?m)^#{2,}\s*-{2,}\s*END DATA_BLOCK\s*-{2,}\s*$"
)
_SECTION_HEADER_PATTERN = re.compile(r"(?m)^#{2,} ")
_TABLE_ROW_PATTERN = re.compile(r"^\|(.+)\|$")
_STRUCTURED_BLOCK_BOUNDARY_NAMES = FENCED_BLOCK_NAMES
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


def _find_legacy_data_block_region(report: str) -> tuple[int, int, str] | None:
    legacy_match = _LEGACY_DATA_BLOCK_HEADER_PATTERN.search(report)
    dashed_match = _DASHED_DATA_BLOCK_HEADER_PATTERN.search(report)
    header_match = legacy_match or dashed_match
    if header_match is None:
        return None

    body_start = header_match.end()
    if dashed_match is not None and header_match is dashed_match:
        explicit_end = _EXPLICIT_DATA_BLOCK_END_PATTERN.search(report, body_start)
        if explicit_end is None:
            return None
        body_end = explicit_end.start()
        replace_end = explicit_end.end()
    else:
        next_section = _SECTION_HEADER_PATTERN.search(report, body_start)
        body_end = next_section.start() if next_section else len(report)
        replace_end = body_end

    body = report[body_start:body_end].strip()
    if not body:
        return None
    return header_match.start(), replace_end, body


def _extract_legacy_data_block_body(report: str) -> str | None:
    region = _find_legacy_data_block_region(report)
    if region is None:
        return None
    return region[2]


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

    normalized_report = normalize_structured_block_boundaries(report) or report
    blocks = list(_compile_named_block_pattern(block_name).finditer(normalized_report))
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


def find_fenced_block_spans(
    report: str | None,
    block_name: str,
    *,
    include_markers: bool = False,
) -> list[tuple[int, int, str]]:
    """Return spans and content for each parseable fenced block."""
    if not report or not isinstance(report, str):
        return []

    normalized_report = normalize_structured_block_boundaries(report) or report
    matches = list(_compile_named_block_pattern(block_name).finditer(normalized_report))
    group_index = 0 if include_markers else 1
    return [
        (match.start(group_index), match.end(group_index), match.group(group_index))
        for match in matches
    ]


def extract_block_field_from_text(
    block_text: str | None, field_name: str
) -> str | None:
    """Extract a normalized field value from an already extracted block body."""
    value = extract_block_field_from_text_raw(block_text, field_name)
    if value is None:
        return None
    return None if value.upper() in _NULL_TOKENS else value


def extract_block_field_from_text_raw(
    block_text: str | None, field_name: str
) -> str | None:
    """Extract a literal field value from an already extracted block body."""
    if not block_text or not isinstance(block_text, str):
        return None

    pattern = _FIELD_VALUE_PATTERN.format(field_name=re.escape(field_name))
    match = re.search(pattern, block_text, re.IGNORECASE)
    if not match:
        return None

    return match.group(1).strip()


def extract_block_text_value(block_text: str, field_name: str) -> str:
    """Extract a literal field value; convenience wrapper for .upper() call sites."""
    return extract_block_field_from_text_raw(block_text, field_name) or ""


def replace_or_append_block_line(body: str, field_name: str, value: str) -> str:
    """Replace a KEY: value line in a block body, or append it if missing."""
    pattern = re.compile(rf"(?m)^{re.escape(field_name)}:\s*.*$")
    replacement = f"{field_name}: {value}"
    if pattern.search(body):
        return pattern.sub(replacement, body, count=1)
    suffix = "" if body.endswith("\n") else "\n"
    return f"{body}{suffix}{replacement}"


def has_block_field_value(block_text: str, field_name: str) -> bool:
    """Return True when a field exists, including fields set to N/A."""
    return bool(extract_block_text_value(block_text, field_name))


def has_non_na_block_field_value(block_text: str, field_name: str) -> bool:
    """Return True when a field exists and is not a null token."""
    value = extract_block_text_value(block_text, field_name)
    return bool(value) and value.upper() not in _NULL_TOKENS


def extract_block_number_from_text(
    block_text: str | None, field_name: str
) -> float | None:
    """Extract a numeric field value from an already extracted block body."""
    raw = extract_block_field_from_text(block_text, field_name)
    if raw is None:
        return None

    match = _NUMBER_TOKEN_PATTERN.search(raw)
    if not match:
        return None

    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def extract_block_field(
    report: str | None,
    block_name: str,
    field_name: str,
) -> str | None:
    """Extract a normalized field value from the last parseable structured block."""
    if block_name == "DATA_BLOCK":
        block_text = extract_last_data_block(report)
    else:
        block_text = extract_last_fenced_block(report, block_name)
    return extract_block_field_from_text(block_text, field_name)


def extract_block_number(
    report: str | None,
    block_name: str,
    field_name: str,
) -> float | None:
    """Extract a numeric field value from the last parseable structured block."""
    if block_name == "DATA_BLOCK":
        block_text = extract_last_data_block(report)
    else:
        block_text = extract_last_fenced_block(report, block_name)
    return extract_block_number_from_text(block_text, field_name)


def extract_data_block_field(report: str | None, field_name: str) -> str | None:
    """Extract a normalized field value from the last parseable DATA_BLOCK."""
    return extract_block_field(report, "DATA_BLOCK", field_name)


def extract_data_block_number(report: str | None, field_name: str) -> float | None:
    """Extract a numeric field value from the last parseable DATA_BLOCK."""
    return extract_block_number(report, "DATA_BLOCK", field_name)


def normalize_structured_block_boundaries(report: str | None) -> str | None:
    """Insert missing line breaks around known structured fenced markers.

    Repairs two common LLM boundary defects:
    - a recognized fenced marker glued onto the previous prose line, e.g.
      ``...assets come online.### --- START DATA_BLOCK ---``;
    - a recognized end marker immediately followed by the next markdown heading,
      e.g. ``### --- END DATA_BLOCK ---### FINANCIAL HEALTH DETAIL``.
    """
    if not report or not isinstance(report, str):
        return report

    normalized = _GLUED_FENCED_MARKER_RE.sub(r"\g<pre>\n\g<marker>", report)
    for block_name in _STRUCTURED_BLOCK_BOUNDARY_NAMES:
        normalized = re.sub(
            rf"(#{{2,}}[ \t]*-{{2,}}[ \t]*END[ \t]+{re.escape(block_name)}[ \t]*-{{2,}}[ \t]*)(?=#{{2,}}[ \t])",
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

    region = _find_legacy_data_block_region(report)
    if region is None:
        return report

    region_start, region_end, body = region

    normalized_body = _parse_legacy_key_value_body(body) or _parse_legacy_table_body(
        body
    )
    if not normalized_body:
        return report

    repaired_block = build_fenced_block("DATA_BLOCK", normalized_body)
    repaired_report = report[:region_start] + repaired_block + report[region_end:]
    return repaired_report
