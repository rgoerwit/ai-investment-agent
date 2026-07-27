"""Shared deterministic evidence constraints for downstream narrative agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.data_block_utils import extract_block_field
from src.runtime_diagnostics import get_valid_artifact_content
from src.validators.supplemental_extractors import extract_capital_efficiency_signals

AUTHORITATIVE_CORRECTION_MARKER = "AUTHORITATIVE_METRIC_CORRECTION"
_VALUE_TRAP_CONFLICT_TYPE = "VALUE_TRAP_DATA_CONFLICT"


def downstream_evidence_constraints(state: Mapping[str, Any]) -> str:
    """Return prompt constraints that must survive report summarization."""
    fundamentals = get_valid_artifact_content(state, "fundamentals_report") or ""
    if not isinstance(fundamentals, str):
        fundamentals = ""

    constraints: list[str] = []
    if AUTHORITATIVE_CORRECTION_MARKER in fundamentals:
        constraints.append(
            "The Fundamentals report contains an AUTHORITATIVE_METRIC_CORRECTION. "
            "For any listed conflict, use the reconciled DATA_BLOCK value and do "
            "not repeat or score the superseded narrative value."
        )

    signals = extract_capital_efficiency_signals(fundamentals)
    execution = signals.get("shareholder_return_execution", "UNKNOWN")
    if execution != "PROVEN":
        constraints.append(
            "Shareholder-return execution is "
            f"{execution}. Do not treat a buyback authorization, announcement, or "
            "unverified program as proven governance mitigation, management "
            "alignment, a valuation floor, or an executed cash-return catalyst."
        )

    red_flag_types = {
        str(flag.get("type") or "").upper()
        for flag in state.get("red_flags", []) or []
        if isinstance(flag, Mapping)
    }
    if "NO_CATALYST_DETECTED" in red_flag_types:
        constraints.append(
            "NO_CATALYST_DETECTED is limited to activist, index, tender, and "
            "restructuring mechanisms. Do not generalize it to an absence of "
            "operating, earnings, product, capacity, or industry catalysts."
        )
    if _VALUE_TRAP_CONFLICT_TYPE in red_flag_types:
        constraints.append(
            "The raw Value Trap score and TRAP verdict are UNRECONCILED because "
            "authoritative governance evidence contradicts score-bearing inputs. "
            "Do not use that raw score/verdict as a hard fail. Use only independently "
            "verified control, capital-allocation, and catalyst facts plus the "
            "VALUE_TRAP_DATA_CONFLICT review penalty."
        )

    value_trap = get_valid_artifact_content(state, "value_trap_report") or ""
    m_and_a_evidence = extract_block_field(
        value_trap if isinstance(value_trap, str) else "",
        "VALUE_TRAP_BLOCK",
        "M&A_CONTEXT_EVIDENCE",
    )
    if value_trap and m_and_a_evidence != "CITED":
        constraints.append(
            "Value Trap M&A context is not source-verified. Do not name an "
            "acquisition, infer acquisition-led growth, or score M&A quality from "
            "that report unless M&A_CONTEXT_EVIDENCE is CITED."
        )

    capacity = extract_block_field(
        fundamentals,
        "DATA_BLOCK",
        "CAPACITY_UTILIZATION",
    )
    capacity_status = (
        extract_block_field(
            fundamentals,
            "DATA_BLOCK",
            "CAPACITY_EVIDENCE_STATUS",
        )
        or "UNKNOWN"
    ).upper()
    capacity_as_of = (
        extract_block_field(
            fundamentals,
            "DATA_BLOCK",
            "CAPACITY_UTILIZATION_AS_OF",
        )
        or "UNKNOWN"
    ).upper()
    if (
        capacity
        and capacity.upper() not in {"N/A", "NA", "NONE", "UNKNOWN"}
        and (capacity_status != "PRIMARY" or capacity_as_of in {"N/A", "UNKNOWN"})
    ):
        constraints.append(
            f"Capacity utilization {capacity} is {capacity_status} with as-of "
            f"{capacity_as_of}. Retain it as a conditional reference only. Do not "
            "treat it as a confirmed current operating constraint, volume ceiling, "
            "spare-capacity fact, or independent risk/bonus."
        )

    analyst_count = extract_block_field(
        fundamentals,
        "DATA_BLOCK",
        "ANALYST_COVERAGE_ENGLISH",
    )
    if analyst_count and analyst_count.upper() not in {"N/A", "NA", "NONE", "UNKNOWN"}:
        constraints.append(
            f"ANALYST_COVERAGE_ENGLISH ({analyst_count}) is an aggregator "
            "analyst-opinion count. Do not claim that many identifiable "
            "English-language analysts unless separately sourced."
        )

    latest_results_authority = (
        extract_block_field(
            fundamentals,
            "DATA_BLOCK",
            "LATEST_RESULTS_SOURCE_AUTHORITY",
        )
        or ""
    ).upper()
    latest_results_period = extract_block_field(
        fundamentals,
        "DATA_BLOCK",
        "LATEST_RESULTS_PERIOD",
    )
    if latest_results_authority == "PRIMARY" and latest_results_period:
        constraints.append(
            f"LATEST_RESULTS_* contains primary historical actuals for "
            f"{latest_results_period}. Keep MRQ metrics tied to their own statement "
            "period, and do not use actual YoY growth as management guidance, a "
            "forecast, or projected-growth evidence."
        )

    governance_card = state.get("entity_governance_card")
    if isinstance(governance_card, Mapping):
        control_status = str(governance_card.get("control_status") or "UNKNOWN").upper()
        largest = governance_card.get("largest_shareholder")
        largest_pct = largest.get("pct") if isinstance(largest, Mapping) else None
        if control_status != "CONTROLLED":
            constraints.append(
                "Entity Governance Card control status is "
                f"{control_status}. Do not call the issuer a controlled subsidiary, "
                "attribute decisions to a parent/controller, or add parent-control "
                "risk without stronger primary evidence that establishes control."
            )
        if isinstance(largest_pct, int | float) and largest_pct <= 50.0:
            constraints.append(
                f"The largest disclosed stake is {largest_pct:g}%, so "
                "MAJORITY_HOLDER: NONE is compatible with the ownership evidence "
                "and is not an internal data conflict."
            )

    constraints.append(
        "Internal agent disagreement, missing tool coverage, and contract-required "
        "N/A fields are analysis-quality issues, not issuer risk. In particular, "
        "Value Trap ROIC: N/A is expected because that parallel agent does not receive "
        "the later DATA_BLOCK. Score 0.0 risk unless a primary source confirms an "
        "economic issuer fact."
    )

    constraints.append(
        "Assess trading liquidity relative to the proposed order, not as an "
        "institutional absolute. It is a hard fail only when the order would exceed "
        "10% of 30-day average daily turnover, market access is impaired, or patient "
        "limit execution is impracticable. Below that threshold, note liquidity and "
        "sizing, but do not make it a primary rejection reason. If order notional is "
        "unknown, do not infer a hard fail."
    )

    if not constraints:
        return ""
    return "\n\nDETERMINISTIC EVIDENCE CONSTRAINTS:\n- " + "\n- ".join(constraints)
