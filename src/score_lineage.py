"""Deterministic score lineage and fundamentals-report projection."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from src.analysis_snapshot import (
    ClaimRecord,
    claim_id,
    reconcile_data_block_projection,
)
from src.data_block_utils import (
    build_fenced_block,
    extract_last_data_block,
    replace_or_append_block_line,
)

_FIELD_RE = re.compile(r"(?m)^\s*(?:[-*]\s*)?([A-Z][A-Z0-9_]{2,})\s*:\s*(.*?)\s*$")
_SCORE_CRITERION_DEPENDENCIES: dict[
    str,
    dict[str, tuple[tuple[str, ...], ...]],
] = {
    "HEALTH": {
        "ROE": (("ROE_PERCENT",),),
        "ROA": (("ROA_PERCENT",),),
        "OPERATING_MARGIN": (("OPERATING_MARGIN_PERCENT",),),
        "DE_RATIO": (("DE_RATIO_RAW",),),
        "NET_DEBT_EBITDA": (
            ("NET_DEBT_EBITDA_RAW",),
            ("TOTAL_DEBT_RAW", "TOTAL_CASH_RAW", "EBITDA_RAW"),
        ),
        "CURRENT_RATIO": (("CURRENT_RATIO_RAW",),),
        "OCF_POSITIVE": (("OPERATING_CASH_FLOW_RAW",),),
        "FCF_POSITIVE": (("FREE_CASH_FLOW_RAW",),),
        "FCF_YIELD": (("FREE_CASH_FLOW_RAW", "MARKET_CAP_RAW"),),
        "PE_OR_PEG": (("PE_RATIO_TTM",), ("PEG_RATIO",)),
        "EV_EBITDA": (("EV_EBITDA_RAW",),),
        "PB_OR_PS": (("PB_RATIO",), ("PS_RATIO_RAW",)),
    },
    "GROWTH": {
        "REVENUE_GROWTH": (
            ("LATEST_RESULTS_REVENUE_GROWTH_YOY",),
            ("REVENUE_GROWTH_MRQ",),
            ("REVENUE_GROWTH_FY",),
            ("REVENUE_GROWTH_TTM",),
        ),
        "EPS_GROWTH": (
            ("LATEST_RESULTS_EARNINGS_GROWTH_YOY",),
            ("EARNINGS_GROWTH_MRQ",),
            ("EARNINGS_GROWTH_FY",),
            ("EARNINGS_GROWTH_TTM",),
        ),
        "ROA_ROE_IMPROVING": (("PROFITABILITY_TREND",),),
        "GROSS_MARGIN": (("GROSS_MARGIN_PERCENT",),),
        # These remain advisory until their producers emit structured evidence.
        "GLOBAL_EXPANSION": (),
        "R_AND_D_CAPEX_BACKLOG": (),
    },
}


def _replace_report_section(report: str, heading: str, replacement: str) -> str:
    pattern = re.compile(rf"(?ims)^###\s+{re.escape(heading)}\s*$.*?(?=^###\s+|\Z)")
    if pattern.search(report):
        return pattern.sub(replacement.rstrip() + "\n\n", report, count=1)
    return report.rstrip() + "\n\n" + replacement.rstrip() + "\n"


def _render_score_detail(
    kind: str,
    scorecard: Mapping[str, Any],
    claims: Mapping[str, Any],
) -> str:
    title = (
        "FINANCIAL HEALTH DETAIL" if kind == "HEALTH" else "GROWTH TRANSITION DETAIL"
    )
    lines = [
        f"### {title}",
        (
            f"**Score**: {float(scorecard['earned']):g}/"
            f"{float(scorecard['rubric_total']):g} "
            f"(Adjusted: {float(scorecard['percentage']):.1f}%)"
        ),
        "",
        "**Canonical rubric projection**:",
    ]
    criteria = scorecard.get("criteria", {})
    for criterion, component in criteria.items():
        award = str(component.get("award") or "N/A")
        dependencies = tuple(component.get("derived_from") or ())
        support = []
        for dependency_id in dependencies:
            claim = claims.get(dependency_id)
            if isinstance(claim, Mapping):
                support.append(f"{claim.get('field')}={claim.get('value')}")
        support_text = "; ".join(support) if support else "lineage unavailable"
        lines.append(
            f"- {criterion}: {award}/{float(component['max_points']):g} — "
            f"{support_text}"
        )
    lines.append(
        "- Decision use: "
        + (
            "eligible; every included criterion has canonical lineage."
            if scorecard.get("decision_eligible")
            else "advisory only; score consistency validation failed."
        )
    )
    return "\n".join(lines)


def project_analysis_report(
    report: str,
    snapshot: Mapping[str, Any] | None,
) -> str:
    """Render canonical facts and scorecards once into the fundamentals report."""
    if not snapshot or snapshot.get("contract_status") != "VALID":
        return report
    block = extract_last_data_block(report, include_markers=False)
    block_with_markers = extract_last_data_block(report, include_markers=True)
    if block is None or block_with_markers is None:
        return report

    projected, _ = reconcile_data_block_projection(block, snapshot)
    scorecards = snapshot.get("scorecards", {})
    for kind in ("HEALTH", "GROWTH"):
        scorecard = scorecards.get(kind)
        if not isinstance(scorecard, Mapping):
            projected = replace_or_append_block_line(
                projected,
                f"ADJUSTED_{kind}_SCORE",
                "N/A",
            )
            projected = replace_or_append_block_line(
                projected,
                f"{kind}_SCORE_LINEAGE_STATUS",
                "MISSING",
            )
            continue
        criteria = scorecard.get("criteria", {})
        breakdown = "; ".join(
            f"{criterion}={component.get('award', 'N/A')}"
            for criterion, component in criteria.items()
        )
        projected = replace_or_append_block_line(
            projected,
            f"{kind}_SCORE_BREAKDOWN",
            breakdown,
        )
        projected = replace_or_append_block_line(
            projected,
            f"RAW_{kind}_SCORE",
            f"{float(scorecard['earned']):g}/{float(scorecard['rubric_total']):g}",
        )
        projected = replace_or_append_block_line(
            projected,
            f"ADJUSTED_{kind}_SCORE",
            "N/A"
            if not scorecard.get("decision_eligible")
            else (
                f"{float(scorecard['percentage']):.1f}% "
                f"(based on {float(scorecard['available']):g} available points)"
            ),
        )
        projected = replace_or_append_block_line(
            projected,
            f"{kind}_SCORE_LINEAGE_STATUS",
            "COMPLETE" if scorecard.get("decision_eligible") else "ADVISORY",
        )

    block_index = report.rfind(block_with_markers)
    updated = (
        report[:block_index]
        + build_fenced_block("DATA_BLOCK", projected.rstrip())
        + report[block_index + len(block_with_markers) :]
    )
    claims = snapshot.get("claims", {})
    for kind in ("HEALTH", "GROWTH"):
        scorecard = scorecards.get(kind)
        if isinstance(scorecard, Mapping):
            heading = (
                "FINANCIAL HEALTH DETAIL"
                if kind == "HEALTH"
                else "GROWTH TRANSITION DETAIL"
            )
            updated = _replace_report_section(
                updated,
                heading,
                _render_score_detail(kind, scorecard, claims),
            )
    return updated


def add_validated_derivations(
    snapshot: Mapping[str, Any] | None,
    fundamentals_report: str,
    *,
    conflicts: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Add score assessments only when their rubric projection is coherent."""
    if not snapshot or snapshot.get("contract_status") != "VALID":
        return dict(snapshot or {})
    from src.agents.fundamentals_reconciler import parse_score_breakdown
    from src.thesis_constants import (
        GROWTH_SCORE_CRITERIA,
        HEALTH_SCORE_CRITERIA,
        SCORE_PCT_TOLERANCE,
    )

    block = extract_last_data_block(fundamentals_report)
    if not block:
        return dict(snapshot)
    fields = {
        match.group(1): match.group(2).strip() for match in _FIELD_RE.finditer(block)
    }
    claims = dict(snapshot.get("claims", {}))
    scorecards = dict(snapshot.get("scorecards", {}))
    eligible_facts = {
        str(claim.get("field")): str(existing_claim_id)
        for existing_claim_id, claim in claims.items()
        if isinstance(claim, Mapping)
        and claim.get("kind") == "FACT"
        and claim.get("decision_eligible")
    }
    for kind, score_field, criteria in (
        ("HEALTH", "ADJUSTED_HEALTH_SCORE", HEALTH_SCORE_CRITERIA),
        ("GROWTH", "ADJUSTED_GROWTH_SCORE", GROWTH_SCORE_CRITERIA),
    ):
        value = fields.get(score_field)
        breakdown = parse_score_breakdown(
            fields.get(f"{kind}_SCORE_BREAKDOWN", ""),
            kind,
        )
        suspect = (
            fields.get(f"{kind}_SCORE_CONSISTENCY", "").upper().startswith("SUSPECT")
        )
        if not value or breakdown is None or set(breakdown) != set(criteria):
            continue
        numeric_awards = {
            key: float(token)
            for key, token in breakdown.items()
            if token not in {"N/A", "REMOVED"}
        }
        score_match = re.search(r"-?\d+(?:\.\d+)?", value)
        reported_available = sum(criteria[key] for key in numeric_awards)
        reported_pct = (
            sum(numeric_awards.values()) / reported_available * 100.0
            if reported_available
            else None
        )
        if (
            score_match is None
            or reported_pct is None
            or abs(float(score_match.group()) - reported_pct) > SCORE_PCT_TOLERANCE
        ):
            continue

        criterion_dependencies: dict[str, tuple[str, ...]] = {}
        for criterion in numeric_awards:
            resolved: tuple[str, ...] = ()
            for dependency_group in _SCORE_CRITERION_DEPENDENCIES[kind].get(
                criterion,
                (),
            ):
                if all(field in eligible_facts for field in dependency_group):
                    resolved = tuple(
                        eligible_facts[field] for field in dependency_group
                    )
                    break
            if resolved:
                criterion_dependencies[criterion] = resolved
        # A criterion with no configured dependency group at all (e.g.
        # GLOBAL_EXPANSION, R_AND_D_CAPEX_BACKLOG — "advisory until their
        # producers emit structured evidence") can never resolve by
        # construction; that is not the same failure as a criterion that HAS
        # a configured dependency but couldn't find a matching eligible fact
        # this run. Only the latter should veto the scorecard's eligibility —
        # otherwise a structurally-unbacked, routinely-awarded criterion
        # permanently zeroes the whole score regardless of how well every
        # other criterion is corroborated.
        lineage_gaps = tuple(
            criterion
            for criterion, award in numeric_awards.items()
            if award > 0
            and criterion not in criterion_dependencies
            and _SCORE_CRITERION_DEPENDENCIES[kind].get(criterion, ()) != ()
        )
        available = reported_available
        earned = sum(numeric_awards.values())
        expected_pct = reported_pct
        decision_eligible = not suspect and not lineage_gaps

        score_claim_id = claim_id(score_field, None)
        lineage_id = f"derived:score_reconciler:{kind.lower()}"
        resolved_dependencies = tuple(
            dependency_id
            for criterion in criteria
            for dependency_id in criterion_dependencies.get(criterion, ())
        )
        normalized_value = (
            f"{expected_pct:.1f}% (based on {available:g} available points)"
        )
        scorecards[kind] = {
            "criteria": {
                criterion: {
                    "award": breakdown[criterion],
                    "max_points": criteria[criterion],
                    "derived_from": list(criterion_dependencies.get(criterion, ())),
                }
                for criterion in criteria
            },
            "earned": earned,
            "available": available,
            "rubric_total": sum(criteria.values()),
            "percentage": round(expected_pct, 1),
            "decision_eligible": decision_eligible,
            "lineage_gaps": list(lineage_gaps),
        }
        record = ClaimRecord(
            id=score_claim_id,
            field=score_field,
            value=normalized_value,
            period=None,
            authority="AGGREGATOR",
            exactness="CALCULATED",
            coverage="FOUND" if decision_eligible else "UNSUPPORTED",
            source_url=None,
            evidence_id=lineage_id,
            decision_eligible=decision_eligible,
            kind="DERIVED_ASSESSMENT",
            decision_role="GATE_INPUT",
            source_provider="score_reconciler",
            lineage_ids=(lineage_id,),
            derived_from=tuple(dict.fromkeys(resolved_dependencies)),
        )
        claims[score_claim_id] = asdict(record)
    return {
        **snapshot,
        "stage": "POST_SENIOR_DERIVED",
        "claims": claims,
        "scorecards": scorecards,
        "conflicts": [
            *(snapshot.get("conflicts", []) or []),
            *(dict(conflict) for conflict in conflicts),
        ],
    }
