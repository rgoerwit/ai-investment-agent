"""Deterministic provenance checks for Foreign Language Analyst claims."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

import structlog
from langchain_core.messages import BaseMessage

from src.data_block_utils import (
    extract_last_fenced_block,
    fenced_block_pattern,
    fenced_marker_fragment,
    replace_or_append_block_line,
)
from src.text_patterns import (
    EXCHANGE_QUALIFIED_TICKER_RE,
    RESULT_ENVELOPE_BODY_RE,
    RESULT_ENVELOPE_RE,
    URL_RE,
)

from .message_utils import (
    ToolEvidenceRecord,
    normalize_http_url,
    tool_evidence_records,
)

logger = structlog.get_logger(__name__)

_UNKNOWN = {"", "N/A", "NONE", "UNKNOWN", "NOT FOUND"}
_CONTROL_BASIS_TERMS = {
    "BOARD_MAJORITY": ("board majority", "majority of the board"),
    "CONTRACTUAL_RIGHTS": ("contractual control", "contractual rights"),
    "CONSOLIDATED_SUBSIDIARY": (
        "consolidated subsidiary",
        "consolidated financial statements",
    ),
    "VOTING_AGREEMENT": ("voting agreement", "voting rights agreement"),
}
_NON_CONTROL_RELATIONSHIPS = {
    "EQUITY METHOD": (
        "SIGNIFICANT_INFLUENCE_ONLY",
        ("equity method", "equity-method", "significant influence", "associate"),
    ),
    "SIGNIFICANT INFLUENCE": (
        "SIGNIFICANT_INFLUENCE_ONLY",
        ("significant influence", "equity method", "equity-method", "associate"),
    ),
    "ASSOCIATE": (
        "SIGNIFICANT_INFLUENCE_ONLY",
        ("associate", "significant influence", "equity method", "equity-method"),
    ),
    "INDEPENDENT": (
        "NONE",
        ("independent", "not controlled", "no control"),
    ),
}
_CORPORATE_SUFFIXES = {
    "ag",
    "co",
    "company",
    "corp",
    "corporation",
    "inc",
    "limited",
    "ltd",
    "plc",
    "sa",
}
_CAPACITY_EXPANSION_TERMS = (
    "capacity expansion",
    "capacity expansions",
    "facility expansion",
    "facility buildout",
    "facility build-out",
    "new production line",
    "new production lines",
    "capex",
)
_FACILITY_STATUS_TERMS = {
    "UNDER_CONSTRUCTION": ("under construction", "being built", "construction"),
    "RAMPING": ("ramping", "ramp-up", "ramp up"),
    "AT_CAPACITY": ("at capacity", "full capacity"),
    "NONE": ("no expansion", "no buildout", "no facility expansion"),
}
_PREFLIGHT_RESULT_RE = RESULT_ENVELOPE_BODY_RE
LATEST_RESULTS_SOURCE_FIELDS = (
    "LATEST_RESULTS_PERIOD",
    "LATEST_RESULTS_PERIOD_END",
    "LATEST_RESULTS_PRIOR_PERIOD",
    "LATEST_RESULTS_PRIOR_PERIOD_END",
    "LATEST_RESULTS_PERIOD_MONTHS",
    "LATEST_RESULTS_CURRENCY",
    "LATEST_RESULTS_REPORTING_UNIT",
    "LATEST_RESULTS_REVENUE",
    "LATEST_RESULTS_PRIOR_REVENUE",
    "LATEST_RESULTS_EARNINGS",
    "LATEST_RESULTS_PRIOR_EARNINGS",
    "LATEST_RESULTS_EARNINGS_SCOPE",
    "LATEST_RESULTS_SOURCE_URL",
)
_LATEST_RESULTS_NUMERIC_FIELDS = (
    "LATEST_RESULTS_REVENUE",
    "LATEST_RESULTS_PRIOR_REVENUE",
    "LATEST_RESULTS_EARNINGS",
    "LATEST_RESULTS_PRIOR_EARNINGS",
)
_TABLE_NUMBER_RE = re.compile(r"-?\d{1,3}(?:[ ,]\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_TABLE_CHANGE_RE = re.compile(r"(?:%\s*change|qoq|yoy)", re.IGNORECASE)
_REVENUE_ROW_LABELS = {"revenue", "total sales", "operating revenue", "營業收入"}
_URL_RE = URL_RE
_REQUIRED_BLOCK_START_RE = re.compile(
    r"(?im)^###\s+---\s+START\s+(?:MANAGEMENT_GUIDANCE|LATEST_RESULTS)\s+---\s*$"
)
_TOOL_PROTOCOL_LINE_RE = re.compile(
    r"(?im)^\s*(?:assistant\s+)?(?:to|recipient)\s*=\s*functions\.[\w.-]+"
)
_TOOL_ARGUMENT_LINE_RE = re.compile(r"^\s*\{.*\}\s*$")


def _field(report: str, label: str) -> str:
    match = re.search(
        rf"(?im)^\s*(?:[-*]\s*)?{re.escape(label)}\s*:\s*(.+?)\s*$",
        report,
    )
    return match.group(1).strip() if match else ""


def unique_latest_results_block(report: str) -> str | None:
    """Return the sole complete block only when its field names are unambiguous."""
    openers = re.findall(
        rf"(?m)^{fenced_marker_fragment('LATEST_RESULTS', 'START')}[ \t]*$",
        report,
    )
    if len(openers) != 1:
        return None
    matches = list(fenced_block_pattern("LATEST_RESULTS").finditer(report))
    if len(matches) != 1:
        return None
    block = matches[0].group(1)
    fields = re.findall(
        r"(?im)^[ \t]*(?:[-*][ \t]*)?(LATEST_RESULTS_[A-Z_]+)[ \t]*:",
        block,
    )
    if len(fields) != len({field.upper() for field in fields}):
        return None
    return block


def _replace_latest_results_field(report: str, label: str, value: str) -> str:
    match = next(fenced_block_pattern("LATEST_RESULTS").finditer(report))
    body = match.group(1)
    pattern = re.compile(rf"(?im)^([ \t]*(?:[-*][ \t]*)?){re.escape(label)}[ \t]*:.*$")
    if pattern.search(body):
        updated = pattern.sub(lambda item: f"{item.group(1)}{label}: {value}", body)
    else:
        updated = body.rstrip() + f"\n{label}: {value}\n"
    return report[: match.start(1)] + updated + report[match.end(1) :]


def has_foreign_language_protocol_residue(report: str) -> bool:
    """Whether provider tool-call wire syntax leaked into the report text."""
    return bool(_TOOL_PROTOCOL_LINE_RE.search(report))


def _strip_leading_protocol_preamble(report: str) -> str:
    """Remove only a contiguous tool-call transcript before the first contract block."""
    block_start = _REQUIRED_BLOCK_START_RE.search(report)
    if block_start is None:
        return report
    prefix_lines = report[: block_start.start()].splitlines(keepends=True)
    protocol_lines = [
        index
        for index, line in enumerate(prefix_lines)
        if _TOOL_PROTOCOL_LINE_RE.match(line)
    ]
    if not protocol_lines:
        return report

    preamble_start = protocol_lines[0]
    while preamble_start > 0 and (
        not prefix_lines[preamble_start - 1].strip()
        or _TOOL_ARGUMENT_LINE_RE.fullmatch(prefix_lines[preamble_start - 1].strip())
    ):
        preamble_start -= 1

    candidate = prefix_lines[preamble_start:]
    if any(
        line.strip()
        and not _TOOL_PROTOCOL_LINE_RE.match(line)
        and not _TOOL_ARGUMENT_LINE_RE.fullmatch(line.strip())
        for line in candidate
    ):
        return report
    return "".join(prefix_lines[:preamble_start]) + report[block_start.start() :]


def _split_search_result_records(
    records: Sequence[ToolEvidenceRecord],
) -> list[ToolEvidenceRecord]:
    """Make source matching operate on one search result, not an entire result page."""
    split_records: list[ToolEvidenceRecord] = []
    for record in records:
        tool_name, content, urls = record
        blocks = RESULT_ENVELOPE_RE.findall(content)
        if not blocks:
            split_records.append(record)
            continue
        for block in blocks:
            block_urls = {
                normalized
                for match in _URL_RE.finditer(block)
                if (normalized := normalize_http_url(match.group(0)))
            }
            if block_urls:
                split_records.append(
                    ToolEvidenceRecord(
                        tool_name=tool_name,
                        content=block,
                        urls=block_urls,
                        evidence_status=record.evidence_status,
                        authority=record.authority,
                    )
                )
    return split_records


def _replace_or_add_field(
    report: str,
    label: str,
    value: str,
    *,
    section: str,
) -> str:
    pattern = re.compile(rf"(?im)^(\s*(?:[-*]\s*)?){re.escape(label)}\s*:\s*.+?\s*$")
    if pattern.search(report):
        return pattern.sub(
            lambda match: f"{match.group(1)}{label}: {value}",
            report,
            count=1,
        )

    header = re.search(rf"(?im)^.*{re.escape(section)}.*$", report)
    if not header:
        return f"{report.rstrip()}\n{label}: {value}\n"
    insertion = header.end()
    return f"{report[:insertion]}\n- {label}: {value}{report[insertion:]}"


def _holder_parts(value: str) -> tuple[str, float | None]:
    if value.strip().upper() in _UNKNOWN:
        return "", None
    pct_match = re.search(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%", value)
    pct = float(pct_match.group(1)) if pct_match else None
    name = re.sub(r"\([^)]*\d{1,3}(?:\.\d+)?\s*%[^)]*\)", " ", value)
    name = re.sub(r"\s+", " ", name).strip(" -–—,;")
    return name, pct


def _claim_in_text(text: str, holder: str, pct: float | None) -> bool:
    folded = " ".join(text.casefold().split())
    holder_tokens = [
        token
        for token in re.findall(r"\w+", holder.casefold())
        if len(token) > 1 and token not in _CORPORATE_SUFFIXES
    ]
    if not holder_tokens or not all(token in folded for token in holder_tokens):
        return False
    if pct is None:
        return False
    pct_token = f"{pct:g}"
    return bool(re.search(rf"(?<!\d){re.escape(pct_token)}(?:0+)?\s*%?", folded))


def _holder_in_text(text: str, holder: str) -> bool:
    folded = " ".join(text.casefold().split())
    holder_tokens = [
        token
        for token in re.findall(r"\w+", holder.casefold())
        if len(token) > 1 and token not in _CORPORATE_SUFFIXES
    ]
    return bool(holder_tokens) and all(token in folded for token in holder_tokens)


def _is_primary_evidence(record: ToolEvidenceRecord) -> bool:
    return record.evidence_status == "EVIDENCE_FOUND" and record.authority in {
        "PRIMARY_REGISTRY",
        "PRIMARY_ISSUER",
    }


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _exact_decimal(value: str) -> Decimal | None:
    candidate = value.strip()
    if not re.fullmatch(r"-?\d[\d,]*(?:\.\d+)?", candidate):
        return None
    try:
        return Decimal(candidate.replace(",", ""))
    except InvalidOperation:
        return None


def _valid_comparative_period(
    current_period_end: str,
    prior_period_end: str,
    period_months: str,
) -> bool:
    try:
        current = date.fromisoformat(current_period_end)
        prior = date.fromisoformat(prior_period_end)
        months = int(period_months)
    except (TypeError, ValueError):
        return False
    delta_days = (current - prior).days
    return 1 <= months <= 12 and 320 <= delta_days <= 410


def _latest_results_record_supports(
    record: ToolEvidenceRecord,
    values: dict[str, str],
    decimals: dict[str, Decimal],
) -> bool:
    """Bind both metric pairs to labelled rows under one retained period header."""
    lines = [line.strip() for line in record.content.splitlines() if line.strip()]
    current = _normalized_text(values["LATEST_RESULTS_PERIOD"])
    prior = _normalized_text(values["LATEST_RESULTS_PRIOR_PERIOD"])
    scope = _normalized_text(values["LATEST_RESULTS_EARNINGS_SCOPE"])
    months = int(values["LATEST_RESULTS_PERIOD_MONTHS"])
    if not all(
        _header_period_matches(values[label], values[end], months)
        and _source_has_period_end(record.content, values[end])
        for label, end in (
            ("LATEST_RESULTS_PERIOD", "LATEST_RESULTS_PERIOD_END"),
            ("LATEST_RESULTS_PRIOR_PERIOD", "LATEST_RESULTS_PRIOR_PERIOD_END"),
        )
    ):
        return False
    for start, line in enumerate(lines):
        if _normalized_text(line) != current:
            continue
        header = lines[start : start + 16]
        prior_positions = [
            index
            for index, item in enumerate(header)
            if _normalized_text(item) == prior
        ]
        if len(prior_positions) != 1:
            continue
        row_start = next(
            (
                index
                for index in range(start + 1, min(start + 32, len(lines)))
                if _normalized_text(lines[index]) in _REVENUE_ROW_LABELS
            ),
            None,
        )
        if row_start is None or row_start <= start + prior_positions[0]:
            continue
        heading = lines[max(0, start - 5) : row_start]
        heading_text = _normalized_text(" ".join(heading))
        unit = _normalized_text(values["LATEST_RESULTS_REPORTING_UNIT"])
        currency = _normalized_text(values["LATEST_RESULTS_CURRENCY"])
        if unit not in heading_text or not (
            currency in heading_text or (currency == "eur" and "€" in heading_text)
        ):
            continue
        columns = [
            item
            for item in lines[start:row_start]
            if _normalized_text(item) in {current, prior}
            or _TABLE_CHANGE_RE.fullmatch(item)
            or re.fullmatch(r"\d{3}年[QH][1-4]|[QH][1-4]\s+\d{4}", item, re.I)
        ]
        if len(columns) < 2 or columns[0] != line:
            continue
        prior_column = next(
            (i for i, item in enumerate(columns) if _normalized_text(item) == prior),
            None,
        )
        if prior_column is None:
            continue
        earnings_row = next(
            (
                index
                for index in range(row_start + 1, min(row_start + 45, len(lines)))
                if _normalized_text(lines[index]) == scope
            ),
            None,
        )
        if earnings_row is None:
            continue
        revenue_cells = _table_row_cells(lines, row_start, len(columns))
        earnings_cells = _table_row_cells(lines, earnings_row, len(columns))
        if revenue_cells is None or earnings_cells is None:
            continue
        if (
            revenue_cells[0] == decimals["LATEST_RESULTS_REVENUE"]
            and revenue_cells[prior_column] == decimals["LATEST_RESULTS_PRIOR_REVENUE"]
            and earnings_cells[0] == decimals["LATEST_RESULTS_EARNINGS"]
            and earnings_cells[prior_column]
            == decimals["LATEST_RESULTS_PRIOR_EARNINGS"]
        ):
            return True
    return False


def _source_has_period_end(content: str, period_end: str) -> bool:
    """Require both asserted end dates in the same inspected document."""
    try:
        end = date.fromisoformat(period_end)
    except ValueError:
        return False
    month_name = end.strftime("%B")
    variants = (
        end.isoformat(),
        f"{end.year}/{end.month:02d}/{end.day:02d}",
        f"{end.day} {month_name} {end.year}",
        f"{month_name} {end.day}, {end.year}",
        f"{end.year}年{end.month}月{end.day}日",
    )
    return any(variant.casefold() in content.casefold() for variant in variants)


def _header_period_matches(label: str, period_end: str, months: int) -> bool:
    """Only accept period labels whose calendar end can be checked exactly."""
    match = re.fullmatch(r"H([12]) (20\d{2})", label, re.I)
    if match:
        half, year = map(int, match.groups())
        day = 30 if half == 1 else 31
        return months == 6 and period_end == f"{year}-{half * 6:02d}-{day}"
    match = re.fullmatch(r"(\d{3})年Q([1-4])", label)
    if match:
        roc_year, quarter = map(int, match.groups())
        month = quarter * 3
        day = 30 if month in {6, 9} else 31
        return months == 3 and period_end == f"{roc_year + 1911}-{month:02d}-{day}"
    match = re.fullmatch(
        r"(Three|Six|Twelve) months ended ([A-Za-z]+ \d{1,2}, 20\d{2})",
        label,
        re.I,
    )
    if match:
        expected_months = {"three": 3, "six": 6, "twelve": 12}[match.group(1).lower()]
        try:
            parsed_end = datetime.strptime(match.group(2), "%B %d, %Y").date()
        except ValueError:
            return False
        return months == expected_months and period_end == parsed_end.isoformat()
    return False


def _table_row_cells(
    lines: list[str], row_start: int, count: int
) -> list[Decimal | None] | None:
    cells: list[Decimal | None] = []
    for line in lines[row_start + 1 : row_start + count + 1]:
        if line == "-" or "%" in line:
            cells.append(None)
            continue
        if not _TABLE_NUMBER_RE.fullmatch(line):
            return None
        try:
            cells.append(Decimal(line.replace(",", "").replace(" ", "")))
        except InvalidOperation:
            return None
    return cells if len(cells) == count else None


_LATEST_RESULTS_START = "### --- START LATEST_RESULTS ---"
_LATEST_RESULTS_END = "### --- END LATEST_RESULTS ---"
_LATEST_RESULTS_OPENER_RE = re.compile(
    r"^#{2,}[ \t]*(?:-{2,}[ \t]*START[ \t]+)?LATEST_RESULTS\b[ \t-]*$"
)
_LATEST_RESULTS_LOOSE_END_RE = re.compile(
    r"^#{2,}[ \t]*-*[ \t]*END[ \t]+LATEST_RESULTS\b"
)
# The optional bullet matches _replace_or_add_field, which inserts "- FIELD: value".
_LATEST_RESULTS_FIELD_LINE_RE = re.compile(
    r"^(?:[-*][ \t]*)?LATEST_RESULTS_[A-Z_]+[ \t]*:"
)
_ANNOTATED_URL_RE = re.compile(r"^(\S+)\s+\([^()]*\)\s*$")
# A heading or a horizontal rule ends the block; anything else is prose.
_SECTION_BOUNDARY_RE = re.compile(r"^(?:#{2,}|-{3,}$|\*{3,}$)")


def _reframe_latest_results_block(report: str) -> str:
    """Restore canonical LATEST_RESULTS markers when only the markers drifted.

    Oct 2026: 19 of 29 FLA contract failures were complete blocks behind a bare
    ``### LATEST_RESULTS`` heading or an END marker missing its dashes. Only an
    unbroken run of field lines is re-framed: prose before the next heading leaves
    the report untouched, so no surrounding text can be absorbed into the block.
    """
    if extract_last_fenced_block(report, "LATEST_RESULTS") is not None:
        return report
    lines = report.splitlines()
    opener = next(
        (i for i, line in enumerate(lines) if _LATEST_RESULTS_OPENER_RE.match(line)),
        None,
    )
    if opener is None:
        return report
    cursor, fields = opener + 1, 0
    while cursor < len(lines) and (
        not lines[cursor].strip() or _LATEST_RESULTS_FIELD_LINE_RE.match(lines[cursor])
    ):
        fields += bool(lines[cursor].strip())
        cursor += 1
    at_end = cursor == len(lines)
    if not fields or not (at_end or _SECTION_BOUNDARY_RE.match(lines[cursor].strip())):
        return report
    closes_here = not at_end and _LATEST_RESULTS_LOOSE_END_RE.match(lines[cursor])
    body = [line for line in lines[opener + 1 : cursor] if line.strip()]
    tail = lines[cursor + 1 :] if closes_here else lines[cursor:]
    rebuilt = [
        *lines[:opener],
        _LATEST_RESULTS_START,
        *body,
        _LATEST_RESULTS_END,
        *([""] if tail else []),
        *tail,
    ]
    return "\n".join(rebuilt) + ("\n" if report.endswith("\n") else "")


def _strip_source_url_annotation(report: str) -> str:
    """Drop a trailing ``(note)`` after the source URL when the rest is a URL."""
    block = unique_latest_results_block(report)
    if block is None:
        return report
    raw = _field(block, "LATEST_RESULTS_SOURCE_URL")
    match = _ANNOTATED_URL_RE.match(raw)
    if not match or not URL_RE.fullmatch(match.group(1)):
        return report
    return _replace_latest_results_field(
        report, "LATEST_RESULTS_SOURCE_URL", match.group(1)
    )


def _normalize_latest_results(
    report: str,
    records: list[ToolEvidenceRecord],
) -> str:
    report = _strip_source_url_annotation(_reframe_latest_results_block(report))
    block = unique_latest_results_block(report)
    if block is None:
        return report
    coverage = _field(block, "LATEST_RESULTS_COVERAGE_STATUS").upper()
    asserted = any(_field(block, field) for field in LATEST_RESULTS_SOURCE_FIELDS)
    if coverage != "FOUND":
        if not asserted:
            return report
        return _replace_latest_results_field(
            report, "LATEST_RESULTS_SOURCE_AUTHORITY", "UNKNOWN"
        )

    values = {field: _field(block, field) for field in LATEST_RESULTS_SOURCE_FIELDS}
    source_url = normalize_http_url(values["LATEST_RESULTS_SOURCE_URL"])
    if source_url is None:
        # A citation-free candidate cannot supply growth, but it must not erase
        # an independently valid MANAGEMENT_GUIDANCE block in the same report.
        normalized = _replace_latest_results_field(
            report, "LATEST_RESULTS_SOURCE_URL", "N/A"
        )
        for field in _LATEST_RESULTS_NUMERIC_FIELDS:
            normalized = _replace_latest_results_field(normalized, field, "N/A")
        normalized = _replace_latest_results_field(
            normalized, "LATEST_RESULTS_SOURCE_AUTHORITY", "UNSUPPORTED"
        )
        for field in (
            "LATEST_RESULTS_REVENUE_GROWTH_YOY",
            "LATEST_RESULTS_EARNINGS_GROWTH_YOY",
        ):
            normalized = _replace_latest_results_field(normalized, field, "N/A")
        return normalized
    decimals = {
        field: parsed
        for field in _LATEST_RESULTS_NUMERIC_FIELDS
        if (parsed := _exact_decimal(values[field])) is not None
    }
    required_text_values = (
        values["LATEST_RESULTS_PERIOD"],
        values["LATEST_RESULTS_PRIOR_PERIOD"],
        values["LATEST_RESULTS_CURRENCY"],
        values["LATEST_RESULTS_REPORTING_UNIT"],
        values["LATEST_RESULTS_EARNINGS_SCOPE"],
    )
    structurally_valid = (
        all(value and value.upper() not in _UNKNOWN for value in required_text_values)
        and len(decimals) == len(_LATEST_RESULTS_NUMERIC_FIELDS)
        and _valid_comparative_period(
            values["LATEST_RESULTS_PERIOD_END"],
            values["LATEST_RESULTS_PRIOR_PERIOD_END"],
            values["LATEST_RESULTS_PERIOD_MONTHS"],
        )
    )

    candidate_records = [
        record
        for record in records
        if (
            source_url in record[2]
            if source_url
            else record[0] == "get_official_filings"
        )
    ]
    supporting_records = (
        [
            record
            for record in candidate_records
            if _latest_results_record_supports(record, values, decimals)
        ]
        if structurally_valid
        else []
    )
    primary = any(_is_primary_evidence(record) for record in supporting_records)
    authority = (
        "PRIMARY" if primary else "SECONDARY" if supporting_records else "UNSUPPORTED"
    )

    normalized = _replace_latest_results_field(
        report, "LATEST_RESULTS_SOURCE_AUTHORITY", authority
    )
    if not primary:
        normalized = _replace_latest_results_field(
            normalized,
            "LATEST_RESULTS_REVENUE_GROWTH_YOY",
            "N/A",
        )
        normalized = _replace_latest_results_field(
            normalized,
            "LATEST_RESULTS_EARNINGS_GROWTH_YOY",
            "N/A",
        )
        return normalized

    revenue_prior = decimals["LATEST_RESULTS_PRIOR_REVENUE"]
    earnings_prior = decimals["LATEST_RESULTS_PRIOR_EARNINGS"]
    revenue_growth = (
        (decimals["LATEST_RESULTS_REVENUE"] - revenue_prior) / revenue_prior
        if revenue_prior > 0
        else None
    )
    earnings_growth = (
        (decimals["LATEST_RESULTS_EARNINGS"] - earnings_prior) / earnings_prior
        if earnings_prior > 0
        else None
    )
    normalized = _replace_latest_results_field(
        normalized,
        "LATEST_RESULTS_REVENUE_GROWTH_YOY",
        f"{revenue_growth * 100:.1f}%" if revenue_growth is not None else "N/A",
    )
    return _replace_latest_results_field(
        normalized,
        "LATEST_RESULTS_EARNINGS_GROWTH_YOY",
        f"{earnings_growth * 100:.1f}%" if earnings_growth is not None else "N/A",
    )


def _preflight_capacity_records(
    supplemental_evidence: str,
    pct_token: str,
) -> list[ToolEvidenceRecord]:
    """Recover exact capacity claims from code-owned preflight search results."""
    records: list[ToolEvidenceRecord] = []
    pct_pattern = re.compile(rf"(?<!\d){re.escape(pct_token)}(?:0+)?\s*%")
    for block in _PREFLIGHT_RESULT_RE.findall(supplemental_evidence or ""):
        if "capacity" not in block.casefold() or not pct_pattern.search(block):
            continue
        urls = {
            normalized
            for raw_url in re.findall(r"(?is)<url>\s*(.*?)\s*</url>", block)
            if (normalized := normalize_http_url(raw_url))
        }
        if urls:
            records.append(
                ToolEvidenceRecord(
                    tool_name="management_guidance_preflight",
                    content=block,
                    urls=urls,
                    evidence_status="RESULTS_FOUND",
                    authority="SECONDARY",
                )
            )
    return records


def _normalized_relationship(relationship: str) -> str:
    return " ".join(relationship.strip().upper().replace("_", " ").split())


def _record_domains(record: ToolEvidenceRecord) -> set[str]:
    return {hostname for url in record[2] if (hostname := urlsplit(url).hostname)}


def _has_independent_corroboration(
    records: list[ToolEvidenceRecord],
) -> bool:
    """Require two single-source tool records from two distinct web domains."""

    domains = {
        next(iter(record_domains))
        for record in records
        if len(record_domains := _record_domains(record)) == 1
    }
    return len(domains) >= 2


def _validated_control_status(
    *,
    claimed_status: str,
    relationship: str,
    basis: str,
    pct: float | None,
    supporting_records: list[ToolEvidenceRecord],
) -> tuple[str, str]:
    non_control_rule = _NON_CONTROL_RELATIONSHIPS.get(
        _normalized_relationship(relationship)
    )
    if non_control_rule:
        basis_value, terms = non_control_rule
        relationship_records = [
            record
            for record in supporting_records
            if any(term in record[1].casefold() for term in terms)
        ]
        if relationship_records:
            return "NOT_CONTROLLED", basis_value
        return "UNKNOWN", "UNKNOWN"

    if pct is not None and pct > 50.0:
        return "CONTROLLED", "MAJORITY_VOTING_RIGHTS"

    if claimed_status.upper() != "CONTROLLED":
        return "UNKNOWN", "UNKNOWN"

    normalized_basis = basis.strip().upper().replace(" ", "_")
    control_terms = _CONTROL_BASIS_TERMS.get(normalized_basis)
    if not control_terms:
        return "UNKNOWN", "UNKNOWN"

    corroborating_records = [
        record
        for record in supporting_records
        if any(term in record[1].casefold() for term in control_terms)
    ]
    official_support = any(
        _is_primary_evidence(record) for record in corroborating_records
    )
    if official_support or _has_independent_corroboration(corroborating_records):
        return "CONTROLLED", normalized_basis
    return "UNKNOWN", "UNKNOWN"


def _normalize_ownership(
    report: str,
    records: list[ToolEvidenceRecord],
    *,
    ticker: str,
) -> str:
    ownership_start = report.casefold().find("ownership structure")
    ownership_fields_present = any(
        _field(report, label)
        for label in (
            "Largest Shareholder",
            "Controlling Shareholder",
            "Ownership Evidence Status",
            "Ownership Source URL",
        )
    )
    if ownership_start < 0 and not ownership_fields_present:
        return report

    largest_raw = _field(report, "Largest Shareholder") or _field(
        report, "Controlling Shareholder"
    )
    holder, pct = _holder_parts(largest_raw)
    source_url = _field(report, "Ownership Source URL")
    if source_url.upper() in _UNKNOWN:
        source_url = ""
    if not source_url and ownership_start >= 0:
        ownership_end = report.casefold().find("filing cash flow", ownership_start)
        ownership_section = report[
            ownership_start : ownership_end if ownership_end >= 0 else None
        ]
        source_url = _field(ownership_section, "Source")
        if source_url.upper() in _UNKNOWN:
            source_url = ""

    normalized_source = normalize_http_url(source_url)
    claim_records = [
        record
        for record in records
        if _claim_in_text(record[1], holder, pct)
        or (
            holder
            and pct is None
            and _holder_in_text(record[1], holder)
            and "largest shareholder" in record[1].casefold()
        )
    ]
    supporting = [
        record
        for record in claim_records
        if (
            normalized_source in record[2]
            if normalized_source
            else _is_primary_evidence(record)
        )
    ]
    claimed_evidence_status = _field(report, "Ownership Evidence Status").upper()
    relationship = _field(report, "Relationship") or "UNKNOWN"
    non_control_rule = _NON_CONTROL_RELATIONSHIPS.get(
        _normalized_relationship(relationship)
    )
    relationship_records = [
        record
        for record in records
        if _holder_in_text(record[1], holder)
        and non_control_rule
        and any(term in record[1].casefold() for term in non_control_rule[1])
    ]
    relationship_supporting = [
        record
        for record in relationship_records
        if (
            normalized_source in record[2]
            if normalized_source
            else _is_primary_evidence(record)
        )
    ]
    no_ownership_found = (
        claimed_evidence_status == "NOT_FOUND"
        or "ownership data not found" in report.casefold()
    )
    if normalized_source and supporting:
        evidence_status = "VERIFIED_URL"
    elif supporting:
        evidence_status = "VERIFIED_OFFICIAL_FILING"
    elif relationship_supporting:
        evidence_status = "DISCLOSED_UNVERIFIED"
    elif not holder and no_ownership_found:
        evidence_status = "NOT_FOUND"
    elif holder or source_url:
        evidence_status = "REJECTED"
    else:
        evidence_status = "UNKNOWN"
    verified = evidence_status.startswith("VERIFIED")

    claimed_status = _field(report, "Control Status") or "UNKNOWN"
    basis = _field(report, "Control Basis") or "UNKNOWN"
    control_status, control_basis = (
        _validated_control_status(
            claimed_status=claimed_status,
            relationship=relationship,
            basis=basis,
            pct=pct,
            supporting_records=(claim_records if verified else relationship_supporting),
        )
        if verified or evidence_status == "DISCLOSED_UNVERIFIED"
        else ("UNKNOWN", "UNKNOWN")
    )

    largest_value = (
        (f"{holder} ({pct:g}%)" if pct is not None else holder)
        if verified and holder
        else "UNKNOWN"
    )
    influential_entity = (
        holder
        if evidence_status == "DISCLOSED_UNVERIFIED"
        and control_basis == "SIGNIFICANT_INFLUENCE_ONLY"
        else "UNKNOWN"
    )
    controller_value = "NONE" if control_status == "NOT_CONTROLLED" else "UNKNOWN"
    if control_status == "CONTROLLED":
        controller_raw = _field(report, "Controlling Shareholder")
        controller_name, controller_pct = _holder_parts(controller_raw)
        controller_records = [
            record
            for record in records
            if _claim_in_text(record[1], controller_name, controller_pct)
        ]
        controller_supported = bool(controller_records) and (
            any(
                normalized_source in urls
                for _name, _content, urls in controller_records
            )
            if normalized_source
            else any(_is_primary_evidence(record) for record in controller_records)
        )
        if controller_supported:
            controller_value = f"{controller_name} ({controller_pct:g}%)"
        elif control_basis == "MAJORITY_VOTING_RIGHTS":
            controller_value = largest_value
        else:
            control_status = "UNKNOWN"
            control_basis = "UNKNOWN"
    parent_value = _field(report, "Parent Company")
    if control_status != "CONTROLLED":
        parent_value = "NONE" if control_status == "NOT_CONTROLLED" else "UNKNOWN"
    elif parent_value.upper() not in _UNKNOWN:
        parent_tokens = [
            token
            for token in re.findall(r"\w+", parent_value.casefold())
            if len(token) > 1 and token not in _CORPORATE_SUFFIXES
        ]
        support_text = " ".join(content.casefold() for _n, content, _u in supporting)
        if not parent_tokens or not all(
            token in support_text for token in parent_tokens
        ):
            parent_value = "UNKNOWN"

    related = _field(report, "Related Listed Tickers")
    if verified and related.upper() not in _UNKNOWN:
        supported_text = "\n".join(content for _name, content, _urls in supporting)
        tickers = EXCHANGE_QUALIFIED_TICKER_RE.findall(related)
        if not tickers or any(
            ticker_value.upper() not in supported_text.upper()
            for ticker_value in tickers
        ):
            related = "UNKNOWN"
    else:
        related = "UNKNOWN"

    as_of = _field(report, "Ownership As Of") or "UNKNOWN"
    supporting_text = "\n".join(content for _name, content, _urls in supporting)
    if as_of.upper() not in _UNKNOWN and as_of not in supporting_text:
        as_of = "UNKNOWN"

    entity_role = _field(report, "ENTITY_ROLE_OBSERVED") or "UNKNOWN"

    updates = {
        "Largest Shareholder": largest_value,
        "Influential Entity": influential_entity,
        "Controlling Shareholder": controller_value,
        "Control Status": control_status,
        "Control Basis": control_basis,
        "Parent Company": parent_value or "UNKNOWN",
        "ENTITY_ROLE_OBSERVED": entity_role,
        "Related Listed Tickers": related,
        "Ownership Evidence Status": evidence_status,
        "Ownership Source URL": (
            source_url
            if normalized_source
            and evidence_status in {"VERIFIED_URL", "DISCLOSED_UNVERIFIED"}
            else "N/A"
        ),
        "Ownership As Of": as_of,
    }
    if not holder:
        updates = {
            label: value
            for label, value in updates.items()
            if label == "Ownership Evidence Status" or _field(report, label)
        }
    normalized = report
    for label, value in updates.items():
        normalized = _replace_or_add_field(
            normalized,
            label,
            value,
            section="OWNERSHIP STRUCTURE",
        )

    if evidence_status == "REJECTED":
        logger.warning(
            "fla_ownership_evidence_rejected",
            ticker=ticker,
            source_url_present=normalized_source is not None,
            holder_present=bool(holder),
            percentage_present=pct is not None,
        )
    return normalized


def _normalize_capacity(
    report: str,
    records: list[ToolEvidenceRecord],
    *,
    ticker: str,
    supplemental_evidence: str = "",
) -> str:
    capacity = _field(report, "CAPACITY_UTILIZATION")
    source_field = _field(report, "CAPACITY_UTILIZATION_SOURCE_URL")
    if not capacity and not source_field:
        return report
    if capacity.upper() in _UNKNOWN and not source_field:
        return report

    pct_match = re.search(r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%", capacity)
    source_url = source_field
    normalized_source = normalize_http_url(source_url)
    matching_records: list[ToolEvidenceRecord] = []
    if pct_match:
        pct_token = pct_match.group(1)
        candidate_records = [
            *records,
            *_preflight_capacity_records(supplemental_evidence, pct_token),
        ]
        matching_records = [
            record
            for record in candidate_records
            if "capacity" in record[1].casefold()
            and re.search(
                rf"(?<!\d){re.escape(pct_token)}(?:0+)?\s*%",
                record[1],
            )
        ]
        if normalized_source:
            matching_records = [
                record for record in matching_records if normalized_source in record[2]
            ]
        elif (
            len(
                candidate_urls := {
                    url for _name, _content, urls in matching_records for url in urls
                }
            )
            == 1
        ):
            normalized_source = next(iter(candidate_urls))
            source_url = normalized_source

    supported = bool(matching_records and normalized_source)
    evidence_status = (
        "PRIMARY"
        if supported
        and any(_is_primary_evidence(record) for record in matching_records)
        else "SECONDARY"
        if supported
        else "UNSUPPORTED"
        if capacity and capacity.upper() not in _UNKNOWN
        else "UNKNOWN"
    )
    supporting_text = "\n".join(content for _name, content, _urls in matching_records)

    normalized = report
    if capacity and capacity.upper() not in _UNKNOWN and not supported:
        normalized = _replace_or_add_field(
            normalized,
            "CAPACITY_UTILIZATION",
            "N/A",
            section="OUTPUT FORMAT",
        )
        logger.warning(
            "fla_capacity_evidence_rejected",
            ticker=ticker,
            source_url_present=normalized_source is not None,
        )
    normalized = _replace_or_add_field(
        normalized,
        "CAPACITY_UTILIZATION_SOURCE_URL",
        source_url if supported else "N/A",
        section="OUTPUT FORMAT",
    )
    normalized = _replace_or_add_field(
        normalized,
        "CAPACITY_EVIDENCE_STATUS",
        evidence_status,
        section="OUTPUT FORMAT",
    )
    capacity_as_of = _field(report, "CAPACITY_UTILIZATION_AS_OF")
    if not capacity_as_of or capacity_as_of not in supporting_text:
        capacity_as_of = "UNKNOWN"
    normalized = _replace_or_add_field(
        normalized,
        "CAPACITY_UTILIZATION_AS_OF",
        capacity_as_of if supported else "UNKNOWN",
        section="OUTPUT FORMAT",
    )

    facility_status = _field(report, "FACILITY_BUILDOUT_STATUS").upper()
    facility_supported = bool(
        supported
        and facility_status in _FACILITY_STATUS_TERMS
        and any(
            term in supporting_text.casefold()
            for term in _FACILITY_STATUS_TERMS[facility_status]
        )
    )
    if facility_status not in _UNKNOWN and not facility_supported:
        normalized = _replace_or_add_field(
            normalized,
            "FACILITY_BUILDOUT_STATUS",
            "N/A",
            section="OUTPUT FORMAT",
        )

    capex_evidence_status = (
        evidence_status
        if supported
        and any(
            term in supporting_text.casefold() for term in _CAPACITY_EXPANSION_TERMS
        )
        else "UNSUPPORTED"
        if facility_status not in _UNKNOWN or capacity.upper() not in _UNKNOWN
        else "UNKNOWN"
    )
    return _replace_or_add_field(
        normalized,
        "R_AND_D_CAPEX_BACKLOG_EVIDENCE",
        capex_evidence_status,
        section="OUTPUT FORMAT",
    )


def normalize_foreign_language_evidence(
    report: str,
    evidence_messages: Sequence[BaseMessage],
    *,
    ticker: str,
    supplemental_evidence: str = "",
    additional_records: Sequence[ToolEvidenceRecord] = (),
) -> str:
    """Fail closed on unsupported ownership, capacity, and latest-results claims."""

    if not report.strip():
        return report
    report = _strip_leading_protocol_preamble(report)
    records = _split_search_result_records(
        [
            *tool_evidence_records(evidence_messages),
            *additional_records,
        ]
    )
    normalized = _normalize_ownership(report, records, ticker=ticker)
    normalized = _normalize_capacity(
        normalized,
        records,
        ticker=ticker,
        supplemental_evidence=supplemental_evidence,
    )
    return _normalize_latest_results(normalized, records)


FOREIGN_GROWTH_PROMOTION_FIELDS: dict[str, str] = {
    "CAPACITY_UTILIZATION": "CAPACITY_UTILIZATION",
    "CAPACITY_UTILIZATION_SOURCE_URL": "CAPACITY_UTILIZATION_SOURCE_URL",
    "CAPACITY_UTILIZATION_AS_OF": "CAPACITY_UTILIZATION_AS_OF",
    "CAPACITY_EVIDENCE_STATUS": "CAPACITY_EVIDENCE_STATUS",
    "FACILITY_BUILDOUT_STATUS": "FACILITY_BUILDOUT_STATUS",
    "R_AND_D_CAPEX_BACKLOG_EVIDENCE": "R_AND_D_CAPEX_BACKLOG_EVIDENCE",
}
LATEST_RESULTS_CONTEXT_PROMOTION_FIELDS: tuple[str, ...] = (
    "LATEST_RESULTS_COVERAGE_STATUS",
    "LATEST_RESULTS_PERIOD",
    "LATEST_RESULTS_PERIOD_END",
    "LATEST_RESULTS_PRIOR_PERIOD",
    "LATEST_RESULTS_PRIOR_PERIOD_END",
    "LATEST_RESULTS_PERIOD_MONTHS",
    "LATEST_RESULTS_CURRENCY",
    "LATEST_RESULTS_REPORTING_UNIT",
    "LATEST_RESULTS_EARNINGS_SCOPE",
    "LATEST_RESULTS_SOURCE_URL",
    "LATEST_RESULTS_SOURCE_AUTHORITY",
)
LATEST_RESULTS_NUMERIC_PROMOTION_FIELDS: tuple[str, ...] = (
    "LATEST_RESULTS_REVENUE",
    "LATEST_RESULTS_PRIOR_REVENUE",
    "LATEST_RESULTS_EARNINGS",
    "LATEST_RESULTS_PRIOR_EARNINGS",
    "LATEST_RESULTS_REVENUE_GROWTH_YOY",
    "LATEST_RESULTS_EARNINGS_GROWTH_YOY",
)
LATEST_RESULTS_PROMOTION_FIELDS = (
    LATEST_RESULTS_CONTEXT_PROMOTION_FIELDS + LATEST_RESULTS_NUMERIC_PROMOTION_FIELDS
)


def promote_foreign_growth_evidence(body: str, foreign_data: str) -> tuple[str, bool]:
    """Copy code-normalized operating evidence into Senior DATA_BLOCK."""
    updated = body
    promoted = False
    for source_field, target_field in FOREIGN_GROWTH_PROMOTION_FIELDS.items():
        value = _field(foreign_data, source_field)
        if not value:
            continue
        updated = replace_or_append_block_line(updated, target_field, value)
        promoted = True
    if not _field(foreign_data, "R_AND_D_CAPEX_BACKLOG_EVIDENCE") and re.search(
        r"\bR_AND_D_CAPEX_BACKLOG=(?:0\.5|1)\b",
        body,
    ):
        updated = replace_or_append_block_line(
            updated,
            "R_AND_D_CAPEX_BACKLOG_EVIDENCE",
            "UNKNOWN",
        )
        promoted = True
    latest_block = unique_latest_results_block(foreign_data)
    for field in LATEST_RESULTS_CONTEXT_PROMOTION_FIELDS:
        value = _field(latest_block, field) if latest_block is not None else ""
        if not value:
            continue
        updated = replace_or_append_block_line(updated, field, value)
        promoted = True
    if (
        latest_block is not None
        and _field(latest_block, "LATEST_RESULTS_SOURCE_AUTHORITY").upper() == "PRIMARY"
    ):
        for field in LATEST_RESULTS_NUMERIC_PROMOTION_FIELDS:
            value = _field(latest_block, field)
            if not value:
                continue
            updated = replace_or_append_block_line(updated, field, value)
            promoted = True
    return updated, promoted
