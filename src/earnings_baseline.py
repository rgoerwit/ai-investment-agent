"""Pure contracts for management-guidance and earnings-baseline handling."""

from __future__ import annotations

GUIDANCE_COVERAGE_STATUSES = frozenset(
    {
        "FOUND",
        "NOT_DISCLOSED_AFTER_TARGETED_SEARCH",
        "SEARCH_FAILED",
        "UNRESOLVED_AFTER_TARGETED_SEARCH",
        "NOT_APPLICABLE",
    }
)
EARNINGS_BASELINE_STATUSES = frozenset(
    {
        "DURABLE",
        "MIXED",
        "TEMPORARILY_BOOSTED",
        "TEMPORARILY_DEPRESSED",
        "REGIME_DEPENDENT",
        "UNKNOWN",
    }
)
GUIDANCE_BRIDGE_STATUSES = frozenset({"RECONCILED", "UNRESOLVED", "NOT_APPLICABLE"})
UNUSABLE_EARNINGS_BASELINE_STATUSES = frozenset(
    {
        "MIXED",
        "TEMPORARILY_BOOSTED",
        "TEMPORARILY_DEPRESSED",
        "REGIME_DEPENDENT",
        "UNKNOWN",
    }
)


def canonical_enum(value: object) -> str:
    """Return the canonical representation used by structured guidance fields."""
    return str(value or "").strip().upper()


def is_unusable_earnings_baseline(value: object) -> bool:
    """Return whether trailing earnings must not be treated as durable."""
    return canonical_enum(value) in UNUSABLE_EARNINGS_BASELINE_STATUSES


def requires_eps_growth_withholding(
    baseline_status: object,
    bridge_status: object,
) -> bool:
    """Return whether sustained EPS-growth credit must be withheld."""
    return is_unusable_earnings_baseline(baseline_status) or (
        canonical_enum(bridge_status) == "UNRESOLVED"
    )


def is_material_baseline_distortion(value: object) -> bool:
    """Return whether a classified baseline explicitly carries distortion."""
    return canonical_enum(value) in (UNUSABLE_EARNINGS_BASELINE_STATUSES - {"UNKNOWN"})
