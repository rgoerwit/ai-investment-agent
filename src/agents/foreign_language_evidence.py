"""Deterministic provenance checks for Foreign Language Analyst claims."""

from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import urlsplit

import structlog
from langchain_core.messages import BaseMessage

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


def _field(report: str, label: str) -> str:
    match = re.search(
        rf"(?im)^\s*(?:[-*]\s*)?{re.escape(label)}\s*:\s*(.+?)\s*$",
        report,
    )
    return match.group(1).strip() if match else ""


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


def _is_official_filing(tool_name: str | None) -> bool:
    return tool_name == "get_official_filings"


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
        if any(
            _is_official_filing(record[0]) for record in relationship_records
        ) or any(len(_record_domains(record)) == 1 for record in relationship_records):
            return "NOT_CONTROLLED", basis_value
        return "UNKNOWN", "UNKNOWN"

    if pct is not None and pct > 50.0:
        return "CONTROLLED", "MAJORITY_VOTING_RIGHTS"

    if claimed_status.upper() != "CONTROLLED":
        return "UNKNOWN", "UNKNOWN"

    normalized_basis = basis.strip().upper().replace(" ", "_")
    terms = _CONTROL_BASIS_TERMS.get(normalized_basis)
    if not terms:
        return "UNKNOWN", "UNKNOWN"

    corroborating_records = [
        (name, content, urls)
        for name, content, urls in supporting_records
        if any(term in content.casefold() for term in terms)
    ]
    official_support = any(
        _is_official_filing(name) for name, _content, _urls in corroborating_records
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
            else _is_official_filing(record[0])
        )
    ]
    claimed_evidence_status = _field(report, "Ownership Evidence Status").upper()
    no_ownership_found = (
        claimed_evidence_status == "NOT_FOUND"
        or "ownership data not found" in report.casefold()
    )
    if normalized_source and supporting:
        evidence_status = "VERIFIED_URL"
    elif supporting:
        evidence_status = "VERIFIED_OFFICIAL_FILING"
    elif not holder and no_ownership_found:
        evidence_status = "NOT_FOUND"
    elif holder or source_url:
        evidence_status = "REJECTED"
    else:
        evidence_status = "UNKNOWN"
    verified = evidence_status.startswith("VERIFIED")

    relationship = _field(report, "Relationship") or "UNKNOWN"
    claimed_status = _field(report, "Control Status") or "UNKNOWN"
    basis = _field(report, "Control Basis") or "UNKNOWN"
    control_status, control_basis = (
        _validated_control_status(
            claimed_status=claimed_status,
            relationship=relationship,
            basis=basis,
            pct=pct,
            supporting_records=claim_records,
        )
        if verified
        else ("UNKNOWN", "UNKNOWN")
    )

    largest_value = (
        f"{holder} ({pct:g}%)" if verified and holder and pct is not None else "UNKNOWN"
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
            else any(
                _is_official_filing(name)
                for name, _content, _urls in controller_records
            )
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
        "Controlling Shareholder": controller_value,
        "Control Status": control_status,
        "Control Basis": control_basis,
        "Parent Company": parent_value or "UNKNOWN",
        "ENTITY_ROLE_OBSERVED": entity_role,
        "Related Listed Tickers": related,
        "Ownership Evidence Status": evidence_status,
        "Ownership Source URL": source_url if verified and normalized_source else "N/A",
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
    supported = False
    if pct_match and normalized_source:
        pct_token = pct_match.group(1)
        supported = any(
            normalized_source in urls
            and "capacity" in content.casefold()
            and re.search(rf"(?<!\d){re.escape(pct_token)}(?:0+)?\s*%", content)
            for _name, content, urls in records
        )

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
    capacity_as_of = _field(report, "CAPACITY_UTILIZATION_AS_OF")
    evidence_text = "\n".join(
        content for _name, content, urls in records if normalized_source in urls
    )
    if not capacity_as_of or capacity_as_of not in evidence_text:
        capacity_as_of = "UNKNOWN"
    normalized = _replace_or_add_field(
        normalized,
        "CAPACITY_UTILIZATION_AS_OF",
        capacity_as_of if supported else "UNKNOWN",
        section="OUTPUT FORMAT",
    )
    return normalized


def normalize_foreign_language_evidence(
    report: str,
    evidence_messages: Sequence[BaseMessage],
    *,
    ticker: str,
) -> str:
    """Fail closed on unsupported ownership and exact capacity claims."""

    if not report.strip():
        return report
    records = tool_evidence_records(evidence_messages)
    normalized = _normalize_ownership(report, records, ticker=ticker)
    return _normalize_capacity(normalized, records, ticker=ticker)
