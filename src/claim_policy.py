"""Schema and policy registry for canonical analysis claims."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Authority = Literal[
    "PRIMARY",
    "SECONDARY",
    "AGGREGATOR",
    "UNSUPPORTED",
    "UNKNOWN",
]
Exactness = Literal["EXACT", "CALCULATED", "ESTIMATED", "UNKNOWN"]
Coverage = Literal[
    "FOUND",
    "COMPLETE_NO_MATCH",
    "SEARCHED_UNRESOLVED",
    "MISSING",
    "FAILED",
    "UNSUPPORTED",
]
ClaimKind = Literal["FACT", "DERIVED_ASSESSMENT"]
DecisionRole = Literal["SUPPORT", "GATE_INPUT", "CONTEXT"]
ClaimSource = Literal["RAW_METRICS", "FOREIGN_BLOCK", "DERIVED", "LEGACY"]
ValueFormat = Literal["RATIO", "PERCENT_RATIO", "TEXT"]
ProjectedMetadata = Literal["period", "source_url", "authority"]

RAW_FINANCIAL_METRICS_INPUT = "raw_financial_metrics"
STRUCTURED_INGRESS_SOURCES: dict[tuple[str, str], str] = {
    (
        "junior_fundamentals_analyst",
        "get_financial_metrics",
    ): RAW_FINANCIAL_METRICS_INPUT,
}


@dataclass(frozen=True, slots=True)
class ClaimPolicy:
    source: ClaimSource = "LEGACY"
    raw_field: str | None = None
    source_url_field: str | None = None
    authority_field: str | None = None
    period_field: str | None = None
    input_coverage_field: str | None = None
    projected_metadata: tuple[ProjectedMetadata, ...] = ()
    source_required: bool = False
    decision_role: DecisionRole = "CONTEXT"
    kind: ClaimKind = "FACT"
    aliases: tuple[str, ...] = ()
    value_format: ValueFormat = "RATIO"
    project_to_report: bool = True


MATERIAL_CLAIM_POLICIES: dict[str, ClaimPolicy] = {
    "PE_RATIO_TTM": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="trailingPE",
        decision_role="SUPPORT",
        aliases=("p/e", "valuation"),
    ),
    "PE_RATIO_FORWARD": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="forwardPE",
        authority_field="PE_RATIO_FORWARD_SOURCE",
        projected_metadata=("authority",),
        decision_role="SUPPORT",
        aliases=("forward p/e", "forward multiple"),
    ),
    "FORWARD_EPS": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="forwardEps",
        authority_field="FORWARD_EPS_SOURCE",
        projected_metadata=("authority",),
        decision_role="SUPPORT",
        aliases=("forward eps",),
    ),
    "ADJUSTED_HEALTH_SCORE": ClaimPolicy(
        source="DERIVED",
        kind="DERIVED_ASSESSMENT",
        decision_role="GATE_INPUT",
    ),
    "ADJUSTED_GROWTH_SCORE": ClaimPolicy(
        source="DERIVED",
        kind="DERIVED_ASSESSMENT",
        decision_role="GATE_INPUT",
    ),
    "CURRENT_PRICE": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="currentPrice",
        decision_role="SUPPORT",
    ),
    "DE_RATIO": ClaimPolicy(decision_role="GATE_INPUT"),
    "US_REVENUE_PERCENT": ClaimPolicy(decision_role="GATE_INPUT"),
    "ANALYST_COVERAGE_ENGLISH": ClaimPolicy(decision_role="GATE_INPUT"),
    "REVENUE_GROWTH_MRQ": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="revenueGrowth_MRQ",
        period_field="REVENUE_MRQ_PERIOD_END",
        projected_metadata=("period",),
        decision_role="SUPPORT",
        value_format="PERCENT_RATIO",
    ),
    "EARNINGS_GROWTH_MRQ": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="earningsGrowth_MRQ",
        period_field="EARNINGS_MRQ_PERIOD_END",
        projected_metadata=("period",),
        decision_role="SUPPORT",
        value_format="PERCENT_RATIO",
    ),
    "REVENUE_GROWTH_FY": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="revenueGrowth",
        decision_role="SUPPORT",
        value_format="PERCENT_RATIO",
    ),
    "EARNINGS_GROWTH_FY": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="earningsGrowth",
        decision_role="SUPPORT",
        value_format="PERCENT_RATIO",
    ),
    "REVENUE_GROWTH_TTM": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="revenueGrowth_TTM",
        decision_role="SUPPORT",
        value_format="PERCENT_RATIO",
    ),
    "EARNINGS_GROWTH_TTM": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="earningsGrowth_TTM",
        decision_role="SUPPORT",
        value_format="PERCENT_RATIO",
    ),
    "PEG_RATIO": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="pegRatio",
        decision_role="SUPPORT",
    ),
    "PB_RATIO": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="priceToBook",
        decision_role="SUPPORT",
    ),
    # Rubric inputs are registered for score lineage. They remain code-owned
    # snapshot facts but do not overwrite the richer, unit-aware DATA_BLOCK
    # presentation emitted by the existing fundamentals reconciler.
    "ROE_PERCENT": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="returnOnEquity",
        value_format="PERCENT_RATIO",
        project_to_report=False,
    ),
    "ROA_PERCENT": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="returnOnAssets",
        value_format="PERCENT_RATIO",
        project_to_report=False,
    ),
    "OPERATING_MARGIN_PERCENT": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="operatingMargins",
        value_format="PERCENT_RATIO",
        project_to_report=False,
    ),
    "DE_RATIO_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="debtToEquity",
        project_to_report=False,
    ),
    "NET_DEBT_EBITDA_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="netDebtToEbitda",
        project_to_report=False,
    ),
    "TOTAL_DEBT_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="totalDebt",
        project_to_report=False,
    ),
    "TOTAL_CASH_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="totalCash",
        project_to_report=False,
    ),
    "EBITDA_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="ebitda",
        project_to_report=False,
    ),
    "CURRENT_RATIO_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="currentRatio",
        project_to_report=False,
    ),
    "OPERATING_CASH_FLOW_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="operatingCashflow",
        project_to_report=False,
    ),
    "FREE_CASH_FLOW_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="freeCashflow",
        project_to_report=False,
    ),
    "MARKET_CAP_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="marketCap",
        project_to_report=False,
    ),
    "EV_EBITDA_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="enterpriseToEbitda",
        project_to_report=False,
    ),
    "PS_RATIO_RAW": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="priceToSalesTrailing12Months",
        project_to_report=False,
    ),
    "GROSS_MARGIN_PERCENT": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="grossMargins",
        value_format="PERCENT_RATIO",
        project_to_report=False,
    ),
    "PROFITABILITY_TREND": ClaimPolicy(
        source="RAW_METRICS",
        raw_field="profitability_trend",
        value_format="TEXT",
        project_to_report=False,
    ),
    "CAPACITY_UTILIZATION": ClaimPolicy(
        source="FOREIGN_BLOCK",
        source_url_field="CAPACITY_UTILIZATION_SOURCE_URL",
        authority_field="CAPACITY_EVIDENCE_STATUS",
        period_field="CAPACITY_UTILIZATION_AS_OF",
        projected_metadata=("period", "source_url", "authority"),
        source_required=True,
        decision_role="GATE_INPUT",
        aliases=("capacity utilization", "utilization"),
    ),
    "LATEST_RESULTS_REVENUE_GROWTH_YOY": ClaimPolicy(
        source="FOREIGN_BLOCK",
        source_url_field="LATEST_RESULTS_SOURCE_URL",
        authority_field="LATEST_RESULTS_SOURCE_AUTHORITY",
        period_field="LATEST_RESULTS_PERIOD_END",
        input_coverage_field="LATEST_RESULTS_COVERAGE_STATUS",
        source_required=True,
        decision_role="SUPPORT",
        aliases=("latest results revenue", "revenue grew", "revenue growth"),
    ),
    "LATEST_RESULTS_EARNINGS_GROWTH_YOY": ClaimPolicy(
        source="FOREIGN_BLOCK",
        source_url_field="LATEST_RESULTS_SOURCE_URL",
        authority_field="LATEST_RESULTS_SOURCE_AUTHORITY",
        period_field="LATEST_RESULTS_PERIOD_END",
        input_coverage_field="LATEST_RESULTS_COVERAGE_STATUS",
        source_required=True,
        decision_role="SUPPORT",
        aliases=("latest results earnings", "net income grew", "earnings growth"),
    ),
    "GUIDANCE_REVENUE": ClaimPolicy(
        source="FOREIGN_BLOCK",
        source_url_field="GUIDANCE_SOURCE_URL",
        authority_field="GUIDANCE_SOURCE_AUTHORITY",
        period_field="GUIDANCE_PERIOD",
        input_coverage_field="GUIDANCE_COVERAGE_STATUS",
        source_required=True,
        decision_role="SUPPORT",
        aliases=("revenue guidance", "sales guidance"),
    ),
    "GUIDANCE_NET_INCOME": ClaimPolicy(
        source="FOREIGN_BLOCK",
        source_url_field="GUIDANCE_SOURCE_URL",
        authority_field="GUIDANCE_SOURCE_AUTHORITY",
        period_field="GUIDANCE_PERIOD",
        input_coverage_field="GUIDANCE_COVERAGE_STATUS",
        source_required=True,
        decision_role="SUPPORT",
        aliases=("net income guidance", "earnings guidance"),
    ),
}


def material_claim_fields(*, decision_only: bool = False) -> frozenset[str]:
    """Return registered fields so consumers do not maintain parallel literals."""
    return frozenset(
        field
        for field, policy in MATERIAL_CLAIM_POLICIES.items()
        if not decision_only or policy.decision_role != "CONTEXT"
    )


def claim_source_context_fields() -> tuple[str, ...]:
    """Return claim and provenance fields in stable registration order."""
    fields: list[str] = []
    for field, policy in MATERIAL_CLAIM_POLICIES.items():
        fields.append(field)
        for related in (
            policy.source_url_field,
            policy.authority_field,
            policy.period_field,
            policy.input_coverage_field,
        ):
            if related:
                fields.append(related)
    return tuple(dict.fromkeys(fields))
