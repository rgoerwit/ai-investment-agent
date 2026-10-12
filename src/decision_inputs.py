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

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import structlog

from src.claim_policy import CURRENT_GROWTH_CLAIM_FIELDS
from src.liquidity_assessment import LiquidityAssessment
from src.provenance_schema import Scorecard

logger = structlog.get_logger(__name__)

# Contract statuses meaning "the canonical layer ran and would not stand behind
# this payload". DEGRADED is excluded by design — see from_metrics_and_snapshot.
_REJECTED_CONTRACT_STATUSES = frozenset({"INVALID", "DECODE_FAILED"})


def _canonical_ratio(value: Any) -> float | None:
    """A finite, non-negative canonical ratio, else None (bools and text rejected)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _contract_status(snapshot: Mapping[str, Any] | None) -> str | None:
    """Decoded contract status, or None when no snapshot is present.

    Deliberately routed through the one canonical decoder rather than reading
    the raw ``contract_status`` string: a future-schema payload that literally
    says ``"VALID"`` must not be authoritative here while the publication
    boundary independently decodes it to DECODE_FAILED.
    """
    if not isinstance(snapshot, Mapping) or not snapshot:
        return None
    # Lazy import mirrors the publication boundary: keeps this leaf module free
    # of the snapshot builder's heavier dependency graph.
    from src.analysis_snapshot import decoded_contract_status

    return decoded_contract_status(snapshot)


def _snapshot_decision_score(
    snapshot: Mapping[str, Any] | None, kind: str
) -> float | None:
    """The decision percentage the canonical scorecard owns for HEALTH/GROWTH.

    Callers must have already established that the contract is VALID. Returns
    None when the scorecard is missing or not decision-eligible — mirroring what
    `project_analysis_report` would have written into the DATA_BLOCK (N/A
    otherwise). A future-schema or corrupt scorecard decodes fail-closed to None
    (the conservative ineligible path).
    """
    scorecards = (snapshot or {}).get("scorecards") or {}
    scorecard = Scorecard.decode_or_none(scorecards.get(kind))
    if scorecard is None or not scorecard.decision_eligible:
        return None
    return float(scorecard.percentage)


def _scores_close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= 0.1


def _missing_current_growth_fields(
    snapshot: Mapping[str, Any] | None,
    *,
    authoritative: bool,
) -> frozenset[str]:
    """Return current growth fields absent from a VALID canonical snapshot."""
    if not authoritative:
        return frozenset()
    available: set[str] = set()
    raw_claims = (snapshot or {}).get("claims") or {}
    if isinstance(raw_claims, Mapping):
        for claim in raw_claims.values():
            if (
                isinstance(claim, Mapping)
                and claim.get("field") in CURRENT_GROWTH_CLAIM_FIELDS
                and claim.get("decision_eligible") is True
                and claim.get("coverage") == "FOUND"
            ):
                available.add(str(claim["field"]))
    return frozenset(CURRENT_GROWTH_CLAIM_FIELDS - available)


def _verified_gate_number(
    snapshot: Mapping[str, Any] | None, field: str, *, integer: bool = False
) -> float | None:
    """Read an exact, source-bound primary fact; legacy/aggregator counts are unknown."""
    claims = (snapshot or {}).get("claims") or {}
    if not isinstance(claims, Mapping):
        return None
    values = set()
    for claim in claims.values():
        if not isinstance(claim, Mapping) or claim.get("field") != field:
            continue
        if (
            claim.get("decision_eligible") is not True
            or claim.get("coverage") != "FOUND"
            or claim.get("authority") != "PRIMARY"
            or claim.get("exactness") != "EXACT"
            or not claim.get("source_url")
            or not str(claim.get("evidence_id") or "").startswith("evidence:")
        ):
            return None
        value = str(claim.get("value", "")).strip()
        if not re.fullmatch(r"\d+" if integer else r"\d+(?:\.\d+)?%?", value):
            return None
        try:
            number = int(value) if integer else float(value.rstrip("%"))
        except (ValueError, OverflowError):
            return None
        if not integer and number > 100:
            return None
        values.add(number)
    return next(iter(values)) if len(values) == 1 else None


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
    and there is no competing store for them. Retiring the ``decision_metrics``
    dict passthrough entirely was deliberately deferred by Stage 6 (the
    ``detect_red_flags`` dict adapter is kept for the golden dict-tests and the
    snapshot-free ``analysis_index`` caller) — this passthrough is the stable
    contract, not a transitional stopgap.
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
    # Current-growth coverage is distinct from the score denominator: the
    # latter also removes unrelated rubric criteria and cannot establish a
    # data-vacuum exception.
    missing_current_growth_fields: frozenset[str]
    growth_coverage_authoritative: bool
    # True when a VALID canonical snapshot owns the scores (whether it supplied
    # a number or an authoritative None). In that state the DATA_BLOCK value is
    # never used — a snapshot None means "canonically unusable", not "fall back".
    # False covers two distinct states, distinguished by `snapshot_status`:
    # a legacy/no-snapshot run (parsed scores stand), and a snapshot that ran
    # but did not validate (parsed scores stand but are marked SUSPECT).
    snapshot_authoritative: bool
    # Decoded contract status: None (no snapshot) | VALID | INVALID | DECODE_FAILED.
    snapshot_status: str | None = None
    # Aggregator opinion counts and analyst prose cannot establish these gates.
    english_analyst_count: int | None = None
    us_revenue_pct: float | None = None
    liquidity: LiquidityAssessment | None = None
    health_exception_pb_ratio: float | None = None
    health_exception_current_ratio: float | None = None
    health_exception_claim_ids: tuple[str, ...] = ()
    # Canonical raw debtToEquity RATIO (src/data/debt_equity.py), when the run's
    # raw metrics carry one. The leverage gates convert it to percent at their own
    # boundary; the report's DE_RATIO (``debt_to_equity_pct``) cannot override it.
    raw_debt_to_equity_ratio: float | None = None

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
        liquidity: Any = None,
        raw_debt_to_equity_ratio: Any = None,
    ) -> DecisionInputs:
        parsed_health = metrics.get("adjusted_health_score")
        parsed_growth = metrics.get("adjusted_growth_score")

        # A VALID snapshot is the SOLE authority for the decision scores. When it
        # is present it owns both fields: an eligible scorecard supplies the
        # number, and a missing/ineligible scorecard supplies an authoritative
        # None ("canonically unusable"). In neither case may the validator fall
        # back to the narrative DATA_BLOCK value — the canonical layer has
        # already spoken. Only a legacy / no-snapshot run uses the parsed value.
        snapshot_status = _contract_status(snapshot)
        snapshot_authoritative = snapshot_status == "VALID"
        missing_growth_fields = _missing_current_growth_fields(
            snapshot,
            authoritative=snapshot_authoritative,
        )

        # The reconciled dict the engine reads: the parsed metrics with the
        # snapshot-authoritative scores written in. Under a VALID snapshot this
        # can replace a stale narrative score with None (fail-closed); otherwise
        # it is a no-op (the DATA_BLOCK is already the Stage-2 projection).
        decision_metrics = dict(metrics)
        health = parsed_health
        growth = parsed_growth

        if snapshot_authoritative:
            health = _snapshot_decision_score(snapshot, "HEALTH")
            growth = _snapshot_decision_score(snapshot, "GROWTH")
            _log_score_override("HEALTH", health, parsed_health, ticker)
            _log_score_override("GROWTH", growth, parsed_growth, ticker)
            decision_metrics["adjusted_health_score"] = health
            decision_metrics["adjusted_growth_score"] = growth
        elif snapshot_status in _REJECTED_CONTRACT_STATUSES:
            # The canonical layer RAN and rejected the payload (INVALID) or could
            # not read it at all (DECODE_FAILED). `project_analysis_report` only
            # projects under a VALID contract, so the DATA_BLOCK still holds the
            # model's own unreconciled arithmetic — consuming it as a decision
            # score is exactly what the projection exists to prevent. Mark both
            # scores untrusted via the established SUSPECT contract: the gates
            # become indeterminate in BOTH directions (no auto-DNI on the value,
            # and it cannot count as a pass supporting BUY) rather than silently
            # authoritative.
            #
            # Deliberately NOT applied to DEGRADED: that is a pre-senior
            # input-availability state (no usable analytic RAW_METRICS fields),
            # not a verdict on the score arithmetic, and it predates the
            # scorecard entirely. A no-snapshot legacy run is untouched too.
            decision_metrics["health_score_consistency"] = "SUSPECT"
            decision_metrics["growth_score_consistency"] = "SUSPECT"
            logger.warning(
                "decision_input_snapshot_not_authoritative",
                ticker=ticker,
                snapshot_status=snapshot_status,
                health=parsed_health,
                growth=parsed_growth,
                msg=(
                    "Canonical contract did not validate; DATA_BLOCK scores are "
                    "unprojected model arithmetic and are marked untrusted."
                ),
            )

        verified_coverage = (
            _verified_gate_number(snapshot, "ANALYST_COVERAGE_ENGLISH", integer=True)
            if snapshot_authoritative
            else None
        )
        sector_value = getattr(sector, "value", sector)
        pb = current_ratio = None
        exception_ids: tuple[str, ...] = ()
        if snapshot_authoritative:
            from src.score_lineage import eligible_fact_number

            claims = (snapshot or {}).get("claims")
            if isinstance(claims, Mapping):
                pb, pb_id = eligible_fact_number(claims, "PB_RATIO", strict=True)
                current_ratio, ratio_id = eligible_fact_number(
                    claims, "CURRENT_RATIO_RAW", strict=True
                )
                exception_ids = tuple(
                    claim_id for claim_id in (pb_id, ratio_id) if claim_id is not None
                )
        return cls(
            decision_metrics=decision_metrics,
            sector=str(sector_value) if sector_value is not None else None,
            debt_to_equity_pct=metrics.get("debt_to_equity"),
            interest_coverage=metrics.get("interest_coverage"),
            net_income=metrics.get("net_income"),
            fcf=metrics.get("fcf"),
            health_decision_pct=health,
            growth_decision_pct=growth,
            # Read from the reconciled dict, not the raw metrics: a non-VALID
            # contract stamps SUSPECT above and must be visible here too.
            health_score_reliable=(
                decision_metrics.get("health_score_consistency") != "SUSPECT"
            ),
            growth_score_reliable=(
                decision_metrics.get("growth_score_consistency") != "SUSPECT"
            ),
            missing_current_growth_fields=missing_growth_fields,
            growth_coverage_authoritative=snapshot_authoritative,
            snapshot_authoritative=snapshot_authoritative,
            snapshot_status=snapshot_status,
            health_exception_pb_ratio=pb,
            health_exception_current_ratio=current_ratio,
            health_exception_claim_ids=exception_ids,
            english_analyst_count=int(verified_coverage)
            if verified_coverage is not None
            else None,
            us_revenue_pct=_verified_gate_number(snapshot, "US_REVENUE_PERCENT")
            if snapshot_authoritative
            else None,
            liquidity=LiquidityAssessment.from_dict(liquidity),
            raw_debt_to_equity_ratio=_canonical_ratio(raw_debt_to_equity_ratio),
        )

    @property
    def growth_data_vacuum(self) -> bool:
        """Whether all four current TTM/MRQ growth observations are absent."""
        return (
            self.growth_coverage_authoritative
            and self.missing_current_growth_fields == CURRENT_GROWTH_CLAIM_FIELDS
        )
