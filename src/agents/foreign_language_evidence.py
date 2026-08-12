"""Deterministic provenance checks for Foreign Language Analyst claims."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import urlsplit

import structlog
from langchain_core.messages import BaseMessage

from src.data_block_utils import replace_or_append_block_line

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
_PREFLIGHT_RESULT_RE = re.compile(r"(?is)<result\b[^>]*>(.*?)</result>")
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
_NUMBER_TOKEN_RE = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])")
_URL_RE = re.compile(r"https?://[^\s<>\"]+", re.IGNORECASE)


def _field(report: str, label: str) -> str:
    match = re.search(
        rf"(?im)^\s*(?:[-*]\s*)?{re.escape(label)}\s*:\s*(.+?)\s*$",
        report,
    )
    return match.group(1).strip() if match else ""


def _split_search_result_records(
    records: Sequence[ToolEvidenceRecord],
) -> list[ToolEvidenceRecord]:
    """Make source matching operate on one search result, not an entire result page."""
    split_records: list[ToolEvidenceRecord] = []
    for record in records:
        tool_name, content, urls = record
        blocks = re.findall(r"(?is)<result\b[^>]*>.*?</result>", content)
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


def _evidence_has_decimal(content: str, expected: Decimal) -> bool:
    return any(
        parsed == expected
        for token in _NUMBER_TOKEN_RE.findall(content)
        if (parsed := _exact_decimal(token)) is not None
    )


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
    content = record[1]
    normalized_content = _normalized_text(content)
    required_text = (
        "LATEST_RESULTS_PERIOD",
        "LATEST_RESULTS_PERIOD_END",
        "LATEST_RESULTS_PRIOR_PERIOD",
        "LATEST_RESULTS_PRIOR_PERIOD_END",
        "LATEST_RESULTS_CURRENCY",
        "LATEST_RESULTS_REPORTING_UNIT",
        "LATEST_RESULTS_EARNINGS_SCOPE",
    )
    if any(
        _normalized_text(values[field]) not in normalized_content
        for field in required_text
    ):
        return False
    return all(
        _evidence_has_decimal(content, decimals[field])
        for field in _LATEST_RESULTS_NUMERIC_FIELDS
    )


def _normalize_latest_results(
    report: str,
    records: list[ToolEvidenceRecord],
    *,
    ticker: str,
) -> str:
    coverage = _field(report, "LATEST_RESULTS_COVERAGE_STATUS").upper()
    asserted = any(_field(report, field) for field in LATEST_RESULTS_SOURCE_FIELDS)
    if coverage != "FOUND":
        if not asserted:
            return report
        return _replace_or_add_field(
            report,
            "LATEST_RESULTS_SOURCE_AUTHORITY",
            "UNKNOWN",
            section="START LATEST_RESULTS",
        )

    values = {field: _field(report, field) for field in LATEST_RESULTS_SOURCE_FIELDS}
    source_url = normalize_http_url(values["LATEST_RESULTS_SOURCE_URL"])
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

    normalized = _replace_or_add_field(
        report,
        "LATEST_RESULTS_SOURCE_AUTHORITY",
        authority,
        section="START LATEST_RESULTS",
    )
    if not primary:
        normalized = _replace_or_add_field(
            normalized,
            "LATEST_RESULTS_REVENUE_GROWTH_YOY",
            "N/A",
            section="START LATEST_RESULTS",
        )
        normalized = _replace_or_add_field(
            normalized,
            "LATEST_RESULTS_EARNINGS_GROWTH_YOY",
            "N/A",
            section="START LATEST_RESULTS",
        )
        if authority == "UNSUPPORTED":
            logger.warning(
                "fla_latest_results_evidence_rejected",
                ticker=ticker,
                source_url_present=source_url is not None,
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
    normalized = _replace_or_add_field(
        normalized,
        "LATEST_RESULTS_REVENUE_GROWTH_YOY",
        f"{revenue_growth * 100:.1f}%" if revenue_growth is not None else "N/A",
        section="START LATEST_RESULTS",
    )
    return _replace_or_add_field(
        normalized,
        "LATEST_RESULTS_EARNINGS_GROWTH_YOY",
        f"{earnings_growth * 100:.1f}%" if earnings_growth is not None else "N/A",
        section="START LATEST_RESULTS",
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
        record for record in records if _claim_in_text(record[1], holder, pct)
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
        f"{holder} ({pct:g}%)" if verified and holder and pct is not None else "UNKNOWN"
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
        tickers = re.findall(r"\b[A-Z0-9]{1,8}(?:[.-][A-Z0-9]{1,6})\b", related)
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
    return _normalize_latest_results(normalized, records, ticker=ticker)


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
    for field in LATEST_RESULTS_CONTEXT_PROMOTION_FIELDS:
        value = _field(foreign_data, field)
        if not value:
            continue
        updated = replace_or_append_block_line(updated, field, value)
        promoted = True
    if _field(foreign_data, "LATEST_RESULTS_SOURCE_AUTHORITY").upper() == "PRIMARY":
        for field in LATEST_RESULTS_NUMERIC_PROMOTION_FIELDS:
            value = _field(foreign_data, field)
            if not value:
                continue
            updated = replace_or_append_block_line(updated, field, value)
            promoted = True
    return updated, promoted
