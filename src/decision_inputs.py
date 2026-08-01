"""Typed, snapshot-authoritative carrier for the gate-critical decision metrics.

Stage 5a of the provenance-architecture completion. Today the financial-health
validator reads decision metrics by reparsing the Senior Fundamentals DATA_BLOCK
text (`RedFlagDetector.extract_metrics`). For the fields the canonical snapshot
OWNS — the health/growth decision scores — the DATA_BLOCK is already a projection
of the snapshot (`score_lineage.project_analysis_report`), so the two stores
cannot diverge in normal operation. `DecisionInputs` makes that authority
*explicit* and typed: it sources the health/growth decision percentages from the
snapshot scorecard when the canonical contract is VALID, falling back to the
DATA_BLOCK-parsed values only for legacy/no-snapshot artifacts, and logs any
divergence.

This is the seam a later stage (5b/5c) uses to switch `detect_red_flags` onto a
typed input and retire the LEGACY reparse for promoted fields. It is deliberately
additive — the red-flag engine still consumes the metrics dict — so it carries no
behavioural risk on its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _snapshot_decision_score(
    snapshot: Mapping[str, Any] | None, kind: str
) -> float | None:
    """The decision percentage the canonical scorecard owns for HEALTH/GROWTH.

    Returns None unless the contract is VALID and the scorecard is
    decision-eligible — mirroring what `project_analysis_report` would have
    written into the DATA_BLOCK (N/A otherwise).
    """
    if not isinstance(snapshot, Mapping) or snapshot.get("contract_status") != "VALID":
        return None
    scorecard = (snapshot.get("scorecards") or {}).get(kind)
    if not isinstance(scorecard, Mapping) or not scorecard.get("decision_eligible"):
        return None
    percentage = scorecard.get("percentage")
    return float(percentage) if isinstance(percentage, int | float) else None


def _scores_close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= 0.1


def _log_score_override(
    kind: str, snapshot_score: float | None, parsed: float | None, ticker: str | None
) -> None:
    """Log when the canonical score overrides a differing DATA_BLOCK value.

    Fires for a genuine numeric disagreement AND for the fail-closed drop case
    (snapshot says unusable/None while the DATA_BLOCK still shows a number).
    """
    if _scores_close(snapshot_score, parsed):
        return
    logger.warning(
        "decision_input_score_divergence",
        ticker=ticker,
        kind=kind,
        snapshot=snapshot_score,
        data_block=parsed,
        msg="Canonical snapshot score is authoritative; DATA_BLOCK value dropped.",
    )


@dataclass(frozen=True, slots=True)
class DecisionInputs:
    """Transitional authority wrapper for the deterministic red-flag engine.

    NOT yet a fully typed domain model: it carries the full decision-metric
    surface as ``decision_metrics`` (the reconciled dict the engine still reads)
    plus typed accessors for the gate-critical fields. Its load-bearing job today
    is *authority*, not typing — it makes the health/growth decision percentages
    snapshot-authoritative (sourced from the canonical scorecard when the
    contract is VALID; a VALID snapshot owns the field outright). The remaining
    fields are the Senior Fundamentals analyst's own DATA_BLOCK assessments
    (segment flags, ROIC quality, cycle position, cash-flow provenance, …).
    Those judgment fields legitimately originate in the analyst's structured
    output — they are not deterministic facts a canonical claim can own, so the
    engine correctly consumes them from the DATA_BLOCK rather than a fact ledger,
    and there is no competing store for them. Making the engine consume the
    typed fields directly (retiring the dict passthrough) is Stage 6.
    """

    decision_metrics: dict[str, Any]
    sector: str | None
    debt_to_equity_pct: float | None
    interest_coverage: float | None
    net_income: float | None
    fcf: float | None
    health_decision_pct: float | None
    growth_decision_pct: float | None
    health_score_reliable: bool
    growth_score_reliable: bool
    # True when a VALID canonical snapshot owns the scores (whether it supplied
    # a number or an authoritative None). In that state the DATA_BLOCK value is
    # never used — a snapshot None means "canonically unusable", not "fall back".
    snapshot_authoritative: bool

    @classmethod
    def from_metrics(
        cls, metrics: Mapping[str, Any], *, sector: Any = None
    ) -> DecisionInputs:
        """Build a typed input from parsed metrics alone (no snapshot overlay).

        Used by the deterministic golden tests and any caller without a
        canonical snapshot; the engine's output is identical to passing the
        raw metrics dict.
        """
        return cls.from_metrics_and_snapshot(metrics, sector, None)

    @classmethod
    def from_metrics_and_snapshot(
        cls,
        metrics: Mapping[str, Any],
        sector: Any,
        snapshot: Mapping[str, Any] | None,
        *,
        ticker: str | None = None,
    ) -> DecisionInputs:
        parsed_health = metrics.get("adjusted_health_score")
        parsed_growth = metrics.get("adjusted_growth_score")

        # A VALID snapshot is the SOLE authority for the decision scores. When it
        # is present it owns both fields: an eligible scorecard supplies the
        # number, and a missing/ineligible scorecard supplies an authoritative
        # None ("canonically unusable"). In neither case may the validator fall
        # back to the narrative DATA_BLOCK value — the canonical layer has
        # already spoken. Only a legacy / no-snapshot run uses the parsed value.
        snapshot_authoritative = (
            isinstance(snapshot, Mapping) and snapshot.get("contract_status") == "VALID"
        )
        if snapshot_authoritative:
            health = _snapshot_decision_score(snapshot, "HEALTH")
            growth = _snapshot_decision_score(snapshot, "GROWTH")
            _log_score_override("HEALTH", health, parsed_health, ticker)
            _log_score_override("GROWTH", growth, parsed_growth, ticker)
        else:
            health = parsed_health
            growth = parsed_growth

        # The reconciled dict the engine reads: the parsed metrics with the
        # snapshot-authoritative scores written in. Under a VALID snapshot this
        # can replace a stale narrative score with None (fail-closed); otherwise
        # it is a no-op (the DATA_BLOCK is already the Stage-2 projection).
        decision_metrics = dict(metrics)
        if snapshot_authoritative:
            decision_metrics["adjusted_health_score"] = health
            decision_metrics["adjusted_growth_score"] = growth

        sector_value = getattr(sector, "value", sector)
        return cls(
            decision_metrics=decision_metrics,
            sector=str(sector_value) if sector_value is not None else None,
            debt_to_equity_pct=metrics.get("debt_to_equity"),
            interest_coverage=metrics.get("interest_coverage"),
            net_income=metrics.get("net_income"),
            fcf=metrics.get("fcf"),
            health_decision_pct=health,
            growth_decision_pct=growth,
            health_score_reliable=metrics.get("health_score_consistency") != "SUSPECT",
            growth_score_reliable=metrics.get("growth_score_consistency") != "SUSPECT",
            snapshot_authoritative=snapshot_authoritative,
        )
