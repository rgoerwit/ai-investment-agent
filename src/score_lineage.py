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
from src.provenance_schema import (
    Scorecard,
    ScorecardCriterion,
)
from src.thesis_constants import (
    FCF_YIELD_MIN_PCT,
    NET_DEBT_EBITDA_MAX,
    UTILITIES_FCF_YIELD_MIN_PCT,
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
        "ROA_ROE_IMPROVING": (
            ("ROA_YOY_CHANGE_PERCENT",),
            ("ROE_YOY_CHANGE_PERCENT",),
        ),
        "GROSS_MARGIN": (("GROSS_MARGIN_PERCENT",),),
        # These remain advisory until their producers emit structured evidence.
        "GLOBAL_EXPANSION": (),
        "R_AND_D_CAPEX_BACKLOG": (),
    },
}


def _claim_percent_value(claim: Mapping[str, Any]) -> float | None:
    """Parse a canonical percent claim (``30.0%`` -> ``30.0``)."""
    if not claim.get("decision_eligible"):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", str(claim.get("value", "")))
    return float(match.group()) if match else None


def _profitability_growth_award(
    claims: Mapping[str, Any],
) -> tuple[str, tuple[str, ...]]:
    """Code-own the ROA/ROE >30% YoY rubric criterion.

    Either return series clearing the threshold earns the single rubric point.
    With neither annual comparison available, the criterion is removed from the
    adaptive denominator instead of trusting a model inference.
    """
    supporting_ids: list[str] = []
    changes: list[float] = []
    for existing_claim_id, claim in claims.items():
        if not isinstance(claim, Mapping) or claim.get("field") not in {
            "ROA_YOY_CHANGE_PERCENT",
            "ROE_YOY_CHANGE_PERCENT",
        }:
            continue
        value = _claim_percent_value(claim)
        if value is not None:
            supporting_ids.append(str(existing_claim_id))
            changes.append(value)
    if not changes:
        return "N/A", ()
    return ("1" if any(change > 30.0 for change in changes) else "0"), tuple(
        supporting_ids
    )


def _eligible_fact_number(
    claims: Mapping[str, Any], field: str
) -> tuple[float | None, str | None]:
    """Return one decision-eligible numeric fact without guessing among conflicts."""
    matches: list[tuple[float, str]] = []
    for existing_claim_id, claim in claims.items():
        if (
            not isinstance(claim, Mapping)
            or claim.get("kind") != "FACT"
            or claim.get("field") != field
            or not claim.get("decision_eligible")
        ):
            continue
        value = _text_number(str(claim.get("value", "")))
        if value is not None:
            matches.append((value, str(existing_claim_id)))
    if not matches:
        return None, None
    first_value, first_id = matches[0]
    if any(value != first_value for value, _ in matches[1:]):
        return None, None
    return first_value, first_id


def _text_number(value: str) -> float | None:
    """Parse the leading numeric amount from ratios and currency-formatted text."""
    normalized = value.replace("−", "-").replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", normalized)
    if match is None:
        return None
    number = float(match.group())
    prefix = normalized[: match.start()]
    if "-" in prefix or ("(" in prefix and ")" in normalized[match.end() :]):
        number = -abs(number)
    return number


def _health_objective_overrides(
    claims: Mapping[str, Any],
    fields: Mapping[str, str],
    model_breakdown: Mapping[str, str],
) -> tuple[
    dict[str, str],
    dict[str, tuple[str, ...]],
    list[dict[str, str]],
]:
    """Code-own only sector-invariant health awards with one coherent basis.

    A disagreement between the finalized DATA_BLOCK and the structured raw fact is
    uncertainty, not permission to choose the more favorable value. Both affected
    FCF criteria are therefore withheld from the adaptive denominator.
    """
    overrides: dict[str, str] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    conflicts: list[dict[str, str]] = []

    def override(
        criterion: str,
        award: str,
        supporting_ids: tuple[str, ...],
        *,
        reason: str = "CODE_OWNED_RUBRIC_OVERRIDE",
    ) -> None:
        overrides[criterion] = award
        dependencies[criterion] = supporting_ids
        reported = model_breakdown[criterion]
        if award != reported:
            conflicts.append(
                {
                    "field": criterion,
                    "canonical": award,
                    "reported": reported,
                    "reason": reason,
                }
            )

    direct_ratio, direct_ratio_id = _eligible_fact_number(claims, "NET_DEBT_EBITDA_RAW")
    debt, debt_id = _eligible_fact_number(claims, "TOTAL_DEBT_RAW")
    cash, cash_id = _eligible_fact_number(claims, "TOTAL_CASH_RAW")
    ebitda, ebitda_id = _eligible_fact_number(claims, "EBITDA_RAW")
    ratio = direct_ratio
    ratio_dependencies: tuple[str, ...] = (direct_ratio_id,) if direct_ratio_id else ()
    if ratio is None and None not in {debt, cash, ebitda} and ebitda is not None:
        if ebitda > 0:
            assert debt is not None and cash is not None
            ratio = (debt - cash) / ebitda
            ratio_dependencies = tuple(
                dependency
                for dependency in (debt_id, cash_id, ebitda_id)
                if dependency is not None
            )
    reported_ratio = _text_number(fields.get("NET_DEBT_EBITDA", ""))
    if reported_ratio is not None and ebitda is not None and ebitda <= 0:
        override(
            "NET_DEBT_EBITDA",
            "N/A",
            (),
            reason="CANONICAL_METRIC_BASIS_CONFLICT",
        )
    elif ratio is not None and reported_ratio is not None:
        if abs(ratio - reported_ratio) <= 0.05:
            override(
                "NET_DEBT_EBITDA",
                "1" if ratio < NET_DEBT_EBITDA_MAX else "0",
                ratio_dependencies,
            )
        else:
            override(
                "NET_DEBT_EBITDA",
                "N/A",
                (),
                reason="CANONICAL_METRIC_BASIS_CONFLICT",
            )

    raw_fcf, raw_fcf_id = _eligible_fact_number(claims, "FREE_CASH_FLOW_RAW")
    report_fcf = _text_number(fields.get("FREE_CASH_FLOW", fields.get("FCF", "")))
    if raw_fcf is None or raw_fcf_id is None or report_fcf is None:
        return overrides, dependencies, conflicts

    raw_sign = (raw_fcf > 0) - (raw_fcf < 0)
    report_sign = (report_fcf > 0) - (report_fcf < 0)
    if raw_sign != report_sign:
        for criterion in ("FCF_POSITIVE", "FCF_YIELD"):
            override(
                criterion,
                "N/A",
                (),
                reason="CANONICAL_METRIC_BASIS_CONFLICT",
            )
        return overrides, dependencies, conflicts

    sector = fields.get("SECTOR", "").strip().casefold()
    if raw_fcf > 0:
        override("FCF_POSITIVE", "1", (raw_fcf_id,))
    elif sector == "information technology":
        dependencies["FCF_POSITIVE"] = (raw_fcf_id,)
        if model_breakdown["FCF_POSITIVE"] not in {"0", "0.5"}:
            override("FCF_POSITIVE", "0", (raw_fcf_id,))
    else:
        override("FCF_POSITIVE", "0", (raw_fcf_id,))

    market_cap, market_cap_id = _eligible_fact_number(claims, "MARKET_CAP_RAW")
    if raw_fcf > 0 and market_cap is not None and market_cap > 0 and market_cap_id:
        threshold = (
            UTILITIES_FCF_YIELD_MIN_PCT if sector == "utilities" else FCF_YIELD_MIN_PCT
        )
        fcf_yield = raw_fcf / market_cap * 100.0
        override(
            "FCF_YIELD",
            "1" if fcf_yield > threshold else "0",
            (raw_fcf_id, market_cap_id),
        )
    elif raw_fcf < 0:
        override("FCF_YIELD", "N/A", (raw_fcf_id,))

    return overrides, dependencies, conflicts


def _replace_report_section(report: str, heading: str, replacement: str) -> str:
    pattern = re.compile(rf"(?ims)^###\s+{re.escape(heading)}\s*$.*?(?=^###\s+|\Z)")
    if pattern.search(report):
        return pattern.sub(replacement.rstrip() + "\n\n", report, count=1)
    return report.rstrip() + "\n\n" + replacement.rstrip() + "\n"


def _render_score_detail(
    kind: str,
    scorecard: Scorecard,
    claims: Mapping[str, Any],
) -> str:
    title = (
        "FINANCIAL HEALTH DETAIL" if kind == "HEALTH" else "GROWTH TRANSITION DETAIL"
    )
    lines = [
        f"### {title}",
        (
            f"**Score**: {float(scorecard.earned):g}/"
            f"{float(scorecard.rubric_total):g} "
            f"(Adjusted: {float(scorecard.percentage):.1f}%)"
        ),
        "",
        "**Canonical rubric projection**:",
    ]
    for criterion, component in scorecard.criteria:
        award = str(component.award or "N/A")
        support = []
        for dependency_id in component.derived_from:
            claim = claims.get(dependency_id)
            if isinstance(claim, Mapping):
                support.append(f"{claim.get('field')}={claim.get('value')}")
        support_text = "; ".join(support) if support else "lineage unavailable"
        lines.append(
            f"- {criterion}: {award}/{float(component.max_points):g} — {support_text}"
        )
    advisory_only = scorecard.advisory_only_awards
    if advisory_only:
        advisory_pct = float(scorecard.advisory_percentage)
        lines.append(
            "- Advisory-only (no evidence producer; excluded from the decision "
            f"score): {', '.join(advisory_only)} — model's raw score "
            f"{advisory_pct:.1f}%"
        )
    if not scorecard.decision_eligible:
        decision_use = "advisory only; score consistency validation failed."
    elif advisory_only:
        decision_use = (
            "decision-score eligible; advisory-only criteria excluded from the "
            "score (see above), every scored criterion has canonical lineage."
        )
    else:
        decision_use = "eligible; every included criterion has canonical lineage."
    lines.append(f"- Decision use: {decision_use}")
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
    raw_scorecards = snapshot.get("scorecards", {})
    # Decode each scorecard once, fail-closed: a future-schema or corrupt
    # scorecard becomes None and is treated exactly like a missing one (N/A).
    decoded: dict[str, Scorecard | None] = {
        kind: Scorecard.decode_or_none(raw_scorecards.get(kind))
        for kind in ("HEALTH", "GROWTH")
    }
    for kind in ("HEALTH", "GROWTH"):
        scorecard = decoded[kind]
        if scorecard is None:
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
        breakdown = "; ".join(
            f"{criterion}={component.award}"
            for criterion, component in scorecard.criteria
        )
        projected = replace_or_append_block_line(
            projected,
            f"{kind}_SCORE_BREAKDOWN",
            breakdown,
        )
        projected = replace_or_append_block_line(
            projected,
            f"RAW_{kind}_SCORE",
            f"{float(scorecard.earned):g}/{float(scorecard.rubric_total):g}",
        )
        projected = replace_or_append_block_line(
            projected,
            f"ADJUSTED_{kind}_SCORE",
            "N/A"
            if not scorecard.decision_eligible
            else (
                f"{float(scorecard.percentage):.1f}% "
                f"(based on {float(scorecard.available):g} available points)"
            ),
        )
        projected = replace_or_append_block_line(
            projected,
            f"{kind}_SCORE_LINEAGE_STATUS",
            "COMPLETE" if scorecard.decision_eligible else "ADVISORY",
        )

    block_index = report.rfind(block_with_markers)
    updated = (
        report[:block_index]
        + build_fenced_block("DATA_BLOCK", projected.rstrip())
        + report[block_index + len(block_with_markers) :]
    )
    claims = snapshot.get("claims", {})
    for kind in ("HEALTH", "GROWTH"):
        scorecard = decoded[kind]
        if scorecard is not None:
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
    derivation_conflicts: list[dict[str, str]] = []
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
        model_breakdown = breakdown
        model_numeric_awards = {
            key: float(token)
            for key, token in model_breakdown.items()
            if token not in {"N/A", "REMOVED"}
        }
        score_match = re.search(r"-?\d+(?:\.\d+)?", value)
        reported_available = sum(criteria[key] for key in model_numeric_awards)
        reported_pct = (
            sum(model_numeric_awards.values()) / reported_available * 100.0
            if reported_available
            else None
        )
        if (
            score_match is None
            or reported_pct is None
            or abs(float(score_match.group()) - reported_pct) > SCORE_PCT_TOLERANCE
        ):
            continue
        assert reported_available > 0

        breakdown = dict(model_breakdown)
        deterministic_dependencies: dict[str, tuple[str, ...]] = {}
        if kind == "HEALTH":
            (
                objective_awards,
                objective_dependencies,
                objective_conflicts,
            ) = _health_objective_overrides(claims, fields, model_breakdown)
            breakdown.update(objective_awards)
            deterministic_dependencies.update(objective_dependencies)
            derivation_conflicts.extend(objective_conflicts)
        else:
            award, dependencies = _profitability_growth_award(claims)
            deterministic_dependencies["ROA_ROE_IMPROVING"] = dependencies
            model_award = model_breakdown["ROA_ROE_IMPROVING"]
            breakdown["ROA_ROE_IMPROVING"] = award
            if award != model_award:
                derivation_conflicts.append(
                    {
                        "field": "ROA_ROE_IMPROVING",
                        "canonical": award,
                        "reported": model_award,
                        "reason": "CODE_OWNED_RUBRIC_OVERRIDE",
                    }
                )
        numeric_awards = {
            key: float(token)
            for key, token in breakdown.items()
            if token not in {"N/A", "REMOVED"}
        }

        criterion_dependencies: dict[str, tuple[str, ...]] = {}
        for criterion in numeric_awards:
            if criterion in deterministic_dependencies:
                deterministic_resolved = deterministic_dependencies[criterion]
                if deterministic_resolved:
                    criterion_dependencies[criterion] = deterministic_resolved
                continue
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
        # Class-1 "advisory" criteria: a positive award on a criterion that has NO
        # configured evidence producer at all (GLOBAL_EXPANSION,
        # R_AND_D_CAPEX_BACKLOG). Such credit is not decision-evidence, so it is
        # excluded from the *decision* score numerator while the full rubric stays
        # the denominator (conservative — an unbacked award can only lower the
        # decision score, never lift it across the 50% gate). The model's raw
        # percentage is preserved as advisory_percentage. Class-2 criteria
        # (configured dependency, unresolved this run) still veto eligibility via
        # lineage_gaps — unchanged, so a genuine lineage gap remains N/A.
        advisory_only_awards = tuple(
            criterion
            for criterion, award in numeric_awards.items()
            if award > 0
            and _SCORE_CRITERION_DEPENDENCIES[kind].get(criterion, ()) == ()
        )
        available = sum(criteria[key] for key in numeric_awards)
        raw_earned = sum(numeric_awards.values())
        decision_earned = raw_earned - sum(
            numeric_awards[criterion] for criterion in advisory_only_awards
        )
        advisory_pct = reported_pct
        decision_pct = decision_earned / available * 100.0
        decision_eligible = not suspect and not lineage_gaps

        score_claim_id = claim_id(score_field, None)
        lineage_id = f"derived:score_reconciler:{kind.lower()}"
        resolved_dependencies = tuple(
            dependency_id
            for criterion in criteria
            for dependency_id in criterion_dependencies.get(criterion, ())
        )
        normalized_value = (
            f"{decision_pct:.1f}% (based on {available:g} available points)"
        )
        scorecards[kind] = Scorecard(
            criteria=tuple(
                (
                    criterion,
                    ScorecardCriterion(
                        award=breakdown[criterion],
                        max_points=criteria[criterion],
                        derived_from=tuple(criterion_dependencies.get(criterion, ())),
                    ),
                )
                for criterion in criteria
            ),
            earned=raw_earned,
            available=available,
            rubric_total=sum(criteria.values()),
            percentage=round(decision_pct, 1),
            advisory_percentage=round(advisory_pct, 1),
            advisory_only_awards=tuple(advisory_only_awards),
            decision_eligible=decision_eligible,
            lineage_gaps=tuple(lineage_gaps),
        ).to_dict()
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
            *derivation_conflicts,
        ],
    }
