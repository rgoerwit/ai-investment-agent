"""
Lessons Learned / Retrospective System

Compares past analysis verdicts to actual market outcomes and generates
generalizable lessons for future analyses. Prediction snapshots are auto-saved
with every analysis; retrospective comparison runs automatically for the current
ticker on re-analysis (skipped with --no-memory).

Design principles:
- Deterministic where possible ($0 cost for snapshot extraction, comparison, confidence)
- One cheap Gemini Flash LLM call per significant delta (~$0.001)
- Early dedup: already-processed snapshots are skipped via ChromaDB metadata query (~50ms)
- Global lesson storage (cross-ticker, cross-sector) with geographic boost at retrieval
- Graceful degradation: failures never block analysis
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import structlog

from src.async_utils import run_with_hard_timeout
from src.config import config
from src.data_block_utils import (
    extract_data_block_field,
    extract_data_block_number,
    extract_kill_criteria,
)
from src.error_safety import summarize_exception
from src.exchange_metadata import SUFFIX_TO_CURRENCY_CODE
from src.ibkr.order_builder import parse_trade_block
from src.runtime_config import get_runtime_config
from src.runtime_diagnostics import classify_failure, get_runtime_provider
from src.runtime_diagnostics.failure_classification import ProviderName, get_model_name
from src.runtime_services import get_current_inspection_service
from src.ticker_policy import get_ticker_suffix
from src.tooling.inspector import InspectionEnvelope, SourceKind

logger = structlog.get_logger(__name__)

_get_capture_manager: Any


class _RetrospectiveMarketData(TypedDict, total=False):
    start_adj_close: float
    end_adj_close: float
    bench_start: float
    bench_end: float
    benchmark_fallback: str


try:
    from src.eval import get_active_capture_manager

    _get_capture_manager = get_active_capture_manager
except ImportError:

    def _fallback_capture_manager() -> None:
        return None

    _get_capture_manager = _fallback_capture_manager


def _record_capture_memory_event(payload: dict[str, Any]) -> None:
    try:
        manager = _get_capture_manager()
        if manager:
            manager.record_memory_event(payload)
    except Exception:
        pass


@dataclass(frozen=True, slots=True)
class SnapshotLoadProgress:
    """Progress update emitted while scanning saved prediction snapshots."""

    phase: str
    total_files: int
    processed_files: int
    loaded_tickers: int
    loaded_snapshots: int
    current_file: str | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

EXCHANGE_BENCHMARK: dict[str, str] = {
    ".T": "^N225",
    ".HK": "^HSI",
    ".TW": "^TWII",
    ".TWO": "^TWII",  # Taipei Exchange (OTC board) — same broad market benchmark
    ".KS": "^KS11",
    ".KQ": "^KS11",  # KOSDAQ — use KOSPI as broad Korea market proxy
    ".SS": "000001.SS",  # Shanghai SSE Composite
    ".SZ": "399001.SZ",  # Shenzhen Component Index
    ".AS": "^AEX",
    ".DE": "^GDAXI",
    ".L": "^FTSE",
    ".PA": "^FCHI",
    ".TO": "^GSPTSE",
    ".AX": "^AXJO",
    ".SI": "^STI",
    ".MI": "^FTSEMIB",
    ".ST": "^OMX",
    ".SA": "^BVSP",  # B3 / Bovespa
}
FALLBACK_BENCHMARK = "^GSPC"

EXCHANGE_CURRENCY: dict[str, str] = dict(SUFFIX_TO_CURRENCY_CODE)
FALLBACK_CURRENCY = "USD"

# Analysis capability, keyed on the binding layer's own intent enum rather than
# on a vendor model string. The predecessor (MODEL_QUALITY) was a model-name
# table whose newest entry was `gemini-3-pro-preview`: it matched 0 of 4,704
# stored snapshots, pinning this factor at 0.5 for every lesson ever generated
# and pushing most of them under the 0.4 retrieval cutoff in
# get_relevant_lessons(). An intent is a closed enum owned by the seat registry,
# so a new model generation needs no edit here — the failure mode that produced
# the bug is structurally gone.
INTENT_QUALITY: dict[str, float] = {
    "critical": 1.0,
    "reasoning": 1.0,
    "fast": 0.7,
}
# Snapshots written before `decision_intent` existed carry only a model name.
# Neutral rather than penalised: the old 0.5 was never a judgment about those
# runs, just an unmatched lookup.
UNKNOWN_INTENT_QUALITY = 1.0

# Applied to both rejection records and outcome lessons. Kept as one constant so
# the two paths cannot drift: they had, silently — `save_rejection_record`
# discounted strict runs and `compute_confidence` did not.
STRICT_MODE_CONFIDENCE_FACTOR = 0.7

# Temporal confidence decay curve
TEMPORAL_WEIGHTS: list[tuple[int, float]] = [
    (30, 0.3),  # 0-30 days: too early
    (90, 0.7),  # 30-90 days: early signal
    (270, 1.0),  # 90-270 days: optimal
    (540, 0.7),  # 270-540 days: degrading
]
TEMPORAL_STALE = 0.3  # 540+ days

# Lesson-trigger thresholds (excess return vs benchmark, stored as positive values)
# For BUY: wrong if excess < -15%, understated if excess > +40%
# For SELL/DNI: wrong if excess > +25%, understated if excess < -30%
# For HOLD: wrong if |excess| > 25%
THRESHOLDS: dict[str, dict[str, float]] = {
    "BUY": {"wrong": 15.0, "understated": 40.0},
    "HOLD": {"wrong": 25.0, "understated": 25.0},
    "DO_NOT_INITIATE": {"wrong": 25.0, "understated": 30.0},
    "SELL": {"wrong": 25.0, "understated": 30.0},
}

MINIMUM_DAYS_ELAPSED = 30
MAX_LESSONS_PER_TICKER = 3

# The only failure mode that asserts nothing, and the reason it exists: every
# other value names a mechanism, so a model told "say so when no mechanism is
# supported" still had to pick a causal category from the response format. The
# escape hatch was unsatisfiable one field over from where it was written.
#
# It is also the right default for an unparseable response. Inferring
# OPERATIONAL_MISS from a parse miss manufactures a finding out of a formatting
# failure — the same category error, arrived at by accident.
UNRESOLVED_PRICE_ONLY = "UNRESOLVED_PRICE_ONLY"

FAILURE_MODES = {
    "CYCLICAL_PEAK",
    "FX_DRIVEN",
    "GOVERNANCE_BLEED",
    "OPERATIONAL_MISS",
    "REGULATORY_SHIFT",
    "MACRO_REGIME",
    "DISRUPTION",
    "VALUATION_TRAP",
    "ACCOUNTING_FRAUD",
    "GEOPOLITICAL",
    "LIQUIDITY_CRISIS",
    "DEAD_MONEY",
    UNRESOLVED_PRICE_ONLY,
}

LESSON_TYPES = {
    "missed_risk",
    "false_positive",
    "missed_opportunity",
    "correct_call",
    "prior_rejection",
}

# `prior_rejection` is a *screening artifact* written by save_rejection_record at
# analysis time — not an outcome. `_lesson_already_processed` used to treat any
# record for a (ticker, analysis_date) as proof the snapshot had been evaluated,
# so a rejection record permanently suppressed the outcome lesson for that
# analysis. Measured on the corpus: 1,698 of 1,790 tickers carried a surviving
# rejection record on a date >= 30 days old, i.e. a genuinely evaluable snapshot
# that could never produce a lesson. Dedup must consider outcome lessons only.
OUTCOME_LESSON_TYPES = LESSON_TYPES - {"prior_rejection"}

# A snapshot that did not trip a threshold at 40 days may trip it at 200, so a
# below-threshold evaluation is provisional rather than final. Re-price once the
# holding period has moved materially.
RETROSPECTIVE_REEVALUATION_INTERVAL_DAYS = 30

# Hard ceiling on yfinance round-trips per run. Without it, every non-triggering
# snapshot is re-fetched on every run forever: 7,690 snapshots are >= 30 days old
# across the live and archived corpora, each costing a stock fetch plus a
# benchmark fetch, sequentially, at a 15 s timeout apiece.
RETROSPECTIVE_MAX_EVALUATIONS_PER_RUN = 400

# Centre of the 1.0-weight TEMPORAL_WEIGHTS band (91-270 days). Budgeted
# evaluation spends its allowance on the highest-confidence window first.
_EVALUATION_BAND_CENTRE_DAYS = 180

# Where the evaluation memo lives. Deliberately a file rather than a ChromaDB
# record: a non-triggering evaluation written to Chroma would be embedded and
# returned by get_relevant_lessons(), polluting retrieval with non-lessons.
DEFAULT_EVALUATION_MEMO_PATH = Path("runtime/retrospective_evaluations.json")

# Memo outcome tokens.
MEMO_OUTCOME_TRIGGERED = "TRIGGERED"
MEMO_OUTCOME_BELOW_THRESHOLD = "BELOW_THRESHOLD"
MEMO_OUTCOME_NO_DATA = "NO_DATA"
# A snapshot that triggered but was capped by MAX_LESSONS_PER_TICKER. Recorded so
# it is not re-priced every run — it was withheld by policy, not lost to failure.
MEMO_OUTCOME_CAPPED = "CAPPED"

# Rides a triggered comparison from phase 2 to phase 3 so the memo can be written
# only once the lesson is durably stored. Stripped before the comparison is
# rendered or persisted.
MEMO_IDENTITY_KEY = "_memo_identity"

# The canonical per-run identity of the snapshot a comparison came from, carried
# so that both dedup sites and the stored record agree on one value.
#
# Deliberately NOT `analysis_id`: persistence.py mints that as the run id (the
# Langfuse trace id when tracing is on), while `snapshot_identity()` falls back to
# the source filename for snapshots written before that field existed. Writing a
# surrogate into a field with an established meaning would give it two writers and
# two semantics. Nothing outside this module reads the stored `analysis_id`
# today — inert, which is not a reason to introduce the conflation.
SNAPSHOT_IDENTITY_KEY = "snapshot_identity"

# How much larger one leg must be than the other to be called the driver. A
# module constant beside THRESHOLDS, tunable once a real corpus exists.
ATTRIBUTION_DOMINANCE_RATIO = 1.5

DRIVER_MARKET = "MARKET"
DRIVER_RESIDUAL = "RESIDUAL"
DRIVER_MIXED = "MIXED"
DRIVER_UNKNOWN = "UNKNOWN"

# Was FX measured, inapplicable, or simply not determinable?
#
# `fx_delta_pct` was initialized to 0.0 and left there on three distinct paths —
# no recorded decision-time rate, a failed live fetch (`except: pass`), and a
# USD-denominated security — so "we could not tell" was reported to the lesson
# model as "the currency did not move". Only the third is a real zero. This is
# the same defect the comment above `benchmark_return_pct` describes as already
# fixed for the market leg; the FX sibling was missed. Measured 2026-08-17: 6 of
# 7,952 snapshots are non-USD with no recorded rate, 987 are USD, and the
# live-fetch failure path is unbounded during a sweep.
# How much of the recorded bear case the lesson prompt gets. `_extract_bear_risks`
# already stores ~500 chars, and the prompt was clipping to 300 — so 55% of
# snapshots (measured over 7,952) had grounding material that existed on disk and
# was never shown to the model. Since the anti-invention rule tells it to name
# only checks that follow from its inputs, withholding inputs it already has is
# the worst of both: the rule binds and the material to satisfy it is missing.
BEAR_EXCERPT_PROMPT_CHARS = 500

FX_OBSERVED = "FX_OBSERVED"
FX_NOT_APPLICABLE = "FX_NOT_APPLICABLE"  # USD-denominated: there is no FX leg
FX_UNAVAILABLE = "FX_UNAVAILABLE"  # could not be determined; NOT a flat rate
FX_OBSERVATIONS = {FX_OBSERVED, FX_NOT_APPLICABLE, FX_UNAVAILABLE}

# Marker on a comparison that could not be assessed. It rides the comparison dict
# rather than a new return type so that `compare_to_reality` keeps its
# `dict | None` contract; the orchestrator branches on it before anything can
# mistake it for a triggered outcome.
UNASSESSED_REASON_KEY = "_unassessed_reason"
UNASSESSED_BENCHMARK = "UNASSESSED_BENCHMARK"

# ── What a lesson may claim ───────────────────────────────────────────────────
#
# Residual dominance does NOT establish that the analysis was wrong. A residual
# can be an earnings surprise, a fraud disclosure, a takeover rumour, sector
# rotation, a data error, or a genuinely mistaken thesis — and price alone cannot
# tell them apart. Stamping such a lesson "general" is the overfitting failure
# this system is most exposed to, given that it learns from a handful of outcomes.
SCOPE_CONTEXTUAL = "CONTEXTUAL"  # market- or regime-dominated
SCOPE_UNRESOLVED = "UNRESOLVED"  # stock-specific, but the cause is unknown
SCOPE_VALIDATED = "VALIDATED"  # cause established by company evidence
LESSON_SCOPES = {SCOPE_CONTEXTUAL, SCOPE_UNRESOLVED, SCOPE_VALIDATED}

# VALIDATED is reserved, not dead-by-accident. Only an evidence-backed company
# post-mortem (fetch filings/news published after the decision and test the kill
# criteria against them) may set it, and no such producer exists yet. Recorded
# here in the `RESERVED_UNOBSERVED` idiom so the next reader knows it is
# deliberate — this repository has been bitten three times by a token that
# silently gated a decision while nothing emitted it (`CMIC_LISTED`,
# `other_legal_risks`, `COVERAGE_COMPLETE_NO_MATCH`). **No gate may key on
# VALIDATED while it is unemitted.**
RESERVED_UNOBSERVED_SCOPES = {SCOPE_VALIDATED}

# ── Whether a lesson may be handed to a live analysis ─────────────────────────
#
# Separate from scope, and the separation is the point. Scope answers "how far
# does this outcome generalize"; eligibility additionally asks whether the record
# carries the evidence needed to *apply* it. Measured 2026-08-16: 67 records were
# stamped CONTEXTUAL while carrying no regime metadata at all, so `_regime_matches`
# could never fire for them — injectable by label, inert in fact. Leaving that
# implicit meant the store could not be audited by reading it.
#
# A positive marker, never a negative one: a record written before this policy
# existed cannot vouch for itself, so absence withholds authority.
LESSON_ELIGIBILITY_INJECTABLE = "INJECTABLE"
LESSON_ELIGIBILITY_REVIEW_ONLY = "REVIEW_ONLY"
LESSON_ELIGIBILITIES = {LESSON_ELIGIBILITY_INJECTABLE, LESSON_ELIGIBILITY_REVIEW_ONLY}

# The retrospective holds price and macro data only. It has no post-decision
# company evidence, so it can never establish whether a pre-registered
# thesis-break trigger fired. This is the *only* value it may write.
THESIS_NOT_EVALUATED = "NOT_EVALUATED"

# A cached macro brief older than this is not "now". The cache refreshes only
# when some analysis happens to run in that region, so staleness is routine.
# 14 days mirrors the reconciler's existing analysis-age default.
REGIME_STALENESS_MAX_DAYS = 14

# Regime fields that bear on a decision. Deliberately not the full enum set:
# `equity_transmission` and `dip_posture` are downstream *descriptions* of the
# same shift, so including them would report the same change several times.
REGIME_COMPARED_FIELDS = ("risk_appetite", "shock_type")

# A LOW-confidence classification is the model saying it could not tell. Comparing
# two such labels produces noise, not a delta.
REGIME_UNUSABLE_CONFIDENCE = "LOW"

# Vector candidates fetched before ranking. The old value of 5, against a top-3
# cut, left the boost/floor machinery almost nothing to rank — it was a fetch
# limit wearing a ranking's clothes.
LESSON_QUERY_CANDIDATES = 20

# UNRESOLVED lessons are STORED but never injected into a live analysis.
#
# An earlier revision injected them with this marker appended, on the reasoning
# that this repo labels unknowns rather than deleting them. Measured against a
# real batch, that reasoning does not survive: **32 of 47 stored lessons (68%)
# were UNRESOLVED**. The stored text is an LLM-authored *imperative* ("avoid
# early margin-recovery stories"), and a caveat rendered beside it does not
# neutralize the instruction — while the underlying move may have been an
# earnings surprise, a takeover rumour, sector rotation or a data error. On a
# corpus this small that is precisely the backtest-overfitting failure (Bailey
# et al.): unexplained returns becoming authoritative rules.
#
# The precedent this repo actually sets is narrower than "label everything":
# `*_SCORE_UNRELIABLE` makes a gate *indeterminate* — it withholds authority. So
# does this. The records accumulate for review and become injectable the moment
# an evidence-backed post-mortem can promote them to VALIDATED.
UNRESOLVED_LESSON_MARKER = (
    "cause unresolved: stock-specific move, no post-decision company evidence"
)

LESSONS_COLLECTION_NAME = "lessons_learned"
_LESSONS_MEMORY_INSTANCE: Any | None = None


# ══════════════════════════════════════════════════════════════════════════════
# Component 1: Prediction Snapshot Extraction
# ══════════════════════════════════════════════════════════════════════════════
def _bear_text(result: Mapping[str, Any]) -> str:
    """Resolve the bear researcher's output from the debate state.

    Single source for the two consumers below — the prose excerpt and the
    pre-committed kill criteria — so they cannot end up reading different rounds
    of the same debate.
    """
    debate = result.get("investment_debate_state", {})
    if not isinstance(debate, Mapping):
        return ""
    return str(debate.get("bear_history") or debate.get("bear_round1") or "")


def _extract_bear_risks(result: dict) -> str:
    """Extract first ~500 chars of bear thesis key risks from debate history."""
    bear_history = _bear_text(result)

    if not bear_history:
        return ""

    # Try to find a KEY RISKS or FAILURE MODE section
    for pattern in [
        r"(?:KEY RISKS|FAILURE MODE|KILL CRITERIA|BEAR CASE).*?(?=\n\n|\Z)",
        r"(?:risk|bear|downside).*",
    ]:
        match = re.search(pattern, bear_history, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(0)[:500]

    # Fallback: first 500 chars of bear history
    return bear_history[:500]


def _extract_data_block_field(fundamentals_report: str, field_name: str) -> str | None:
    """Compatibility wrapper over the shared DATA_BLOCK accessor layer."""
    return extract_data_block_field(fundamentals_report, field_name)


def _extract_data_block_float(
    fundamentals_report: str, field_name: str
) -> float | None:
    """Compatibility wrapper over the shared DATA_BLOCK numeric accessor."""
    return extract_data_block_number(fundamentals_report, field_name)


def _extract_trade_block_text(trader_plan: str, field: str) -> str | None:
    """Extract a text value from a TRADE_BLOCK field (non-numeric)."""
    pattern = rf"{field}:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, trader_plan, re.IGNORECASE)
    if not match:
        return None
    raw = match.group(1).strip()
    return raw if raw and raw.upper() not in ("N/A", "NA", "-", "") else None


def _extract_trade_block_fields(trader_plan: str) -> dict[str, Any]:
    """
    Extract structured TRADE_BLOCK fields from trader output.

    Zero LLM cost — pure regex. Backward-compatible: older JSONs
    without these fields will have None values (handled by reconciler).

    Returns dict with keys: entry_price, stop_price, target_1_price,
    target_2_price, conviction, investment_horizon.
    """
    empty: dict[str, Any] = {
        "entry_price": None,
        "stop_price": None,
        "target_1_price": None,
        "target_2_price": None,
        "conviction": None,
        "investment_horizon": None,
    }
    if not trader_plan:
        return empty

    # Reuse the canonical TRADE_BLOCK parser rather than a second set of field regexes.
    # The decisive difference is that `parse_trade_block` scopes its reads to the block
    # while the copy this replaced searched the whole document: measured over 1,120
    # persisted trader plans the two agreed on 5,581 field reads and disagreed on 19,
    # every one of them the document-wide copy being wrong -- a prose "CONVICTION: N/A"
    # above the block captured as `"N"` (16x), and a non-numeric prose `ENTRY:` line
    # shadowing the real entry price (3x).
    block = parse_trade_block(trader_plan)
    if block is None:
        return empty

    return {
        "entry_price": block.entry_price,
        "stop_price": block.stop_price,
        "target_1_price": block.target_1_price,
        "target_2_price": block.target_2_price,
        "conviction": block.conviction or None,
        # HORIZON is the one field TradeBlockData does not model, so it keeps a local
        # read. Widening the shared model for a single consumer is not worth it.
        "investment_horizon": _extract_trade_block_text(trader_plan, "HORIZON"),
    }


def _provider_quote_currency(result: dict) -> str | None:
    """The currency code the fetcher stamped on the merged quote, if present.

    This is the *decided* denomination, not a guess: ``_normalize_quote_unit_mismatch``
    sets it to the major code when it could corroborate a minor-unit conversion
    and leaves the provider's minor code (``GBp``) when it could not. Returns
    ``None`` for artifacts without a structured-ingress payload, so callers fall
    back to suffix resolution unchanged.
    """
    ingress = (result.get("structured_inputs") or {}).get("raw_financial_metrics")
    payload = ingress.get("payload") if isinstance(ingress, dict) else None
    if not isinstance(payload, dict):
        return None
    currency = payload.get("currency")
    return currency if isinstance(currency, str) and currency.strip() else None


def extract_snapshot(
    result: dict,
    ticker: str,
    is_quick_mode: bool = False,
    *,
    trace_id: str | None = None,
    is_strict_mode: bool = False,
    analysis_id: str | None = None,
    run_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Extract a compact prediction snapshot from an analysis result.

    This is called from save_results_to_file() on every analysis.
    Zero LLM cost — pure regex and dict construction.

    Args:
        result: The full analysis result dict (from graph.ainvoke)
        ticker: Stock ticker symbol

    Returns:
        Compact dict (~20 fields) suitable for JSON serialization
    """
    from src.charts.extractors.pm_block import (
        extract_pm_block,
        extract_verdict_from_text,
    )
    from src.llm_runtime.bindings import active_models_or_legacy

    # Provenance must never cost us a snapshot; degrade to the legacy fields.
    active = active_models_or_legacy(config, quick_mode=is_quick_mode, logger=logger)

    # PM_BLOCK extraction (reuse existing extractor)
    pm_output = result.get("final_trade_decision", "") or ""
    pm_data = extract_pm_block(pm_output)

    # Fallback verdict extraction
    verdict = pm_data.verdict
    if not verdict:
        verdict = extract_verdict_from_text(pm_output)

    # DATA_BLOCK extraction from fundamentals report
    fundamentals = result.get("fundamentals_report", "") or ""

    # Exchange/currency/benchmark mapping
    from src.currency_resolver import resolve_local_trading_currency
    from src.fx_normalization import get_fx_rate_fallback

    suffix = get_ticker_suffix(ticker)

    # The fetcher owns the denomination decision: it converts a minor-unit quote
    # (GBp -> GBP) only when marketCap/PE corroborate, and stamps the resulting
    # code on the payload either way. Passing it through is what lets a pence
    # price stay labelled pence instead of being relabelled pounds by the
    # suffix map — the GAMA.L false-review bug. Absent payload => suffix only,
    # exactly as before.
    resolution = resolve_local_trading_currency(
        ticker=ticker,
        provider_currency=_provider_quote_currency(result),
    )
    currency_source: str
    if resolution.code:
        currency = resolution.code
        currency_source = resolution.source
    elif suffix:
        logger.warning(
            "snapshot_currency_unresolved",
            ticker=ticker,
            suffix=suffix,
            msg=(
                "Currency resolver could not determine a canonical local currency "
                "for a suffixed ticker"
            ),
        )
        currency = None
        currency_source = "unresolved"
    else:
        currency = "USD"
        currency_source = "fallback_bare_ticker"
        logger.debug(
            "snapshot_currency_fallback_bare_ticker",
            ticker=ticker,
            msg="Bare ticker has no canonical exchange suffix; using USD fallback",
        )

    benchmark = EXCHANGE_BENCHMARK.get(suffix, FALLBACK_BENCHMARK)

    # FX rate at analysis time (synchronous fallback only — no async in snapshot).
    # Saved here so the reconciler has an at-analysis-time rate to use for cost
    # calculations without needing a live FX fetch.
    # get_fx_rate_fallback is the canonical fallback-table conversion (minor-unit
    # scaling + USD anchoring); deliberately the table, not live/cache, because
    # this reconstructs the at-analysis-time rate (see the FX section in
    # CLAUDE.md). Do not switch this to a live fetch.
    fx_rate = get_fx_rate_fallback(currency, "USD") if currency else None
    if currency and fx_rate is None:
        logger.warning(
            "snapshot_fx_rate_unknown",
            ticker=ticker,
            currency=currency,
            msg=(
                "Currency not in fallback FX table — saving fx_rate_to_usd=None. "
                "Add to src/fx_normalization.py to fix cost calculations."
            ),
        )

    # TRADE_BLOCK extraction from trader plan (zero LLM cost — pure regex)
    trader_plan = result.get("investment_analysis", {}).get("trader_plan", "") or ""
    trade_block_fields = _extract_trade_block_fields(trader_plan)
    macro_regime_raw = result.get("macro_regime_block") or {}
    macro_regime = macro_regime_raw if isinstance(macro_regime_raw, dict) else {}
    regime_at_decision = macro_regime if macro_regime.get("present") else None

    snapshot = {
        # Core verdict
        "verdict": verdict,
        "health_adj": pm_data.health_adj,
        "growth_adj": pm_data.growth_adj,
        "risk_tally": pm_data.risk_tally,
        "zone": pm_data.zone,
        "position_size": pm_data.position_size,
        # DATA_BLOCK fields
        "current_price": _extract_data_block_float(fundamentals, "CURRENT_PRICE"),
        "currency_source": currency_source,
        "sector": _extract_data_block_field(fundamentals, "SECTOR"),
        "pe_ratio": _extract_data_block_float(fundamentals, "PE_RATIO_TTM"),
        "peg_ratio": _extract_data_block_float(fundamentals, "PEG_RATIO"),
        "pb_ratio": _extract_data_block_float(fundamentals, "PB_RATIO"),
        # ENGLISH = Refinitiv/FactSet (global aggregator, English-bias). TOTAL_EST =
        # Senior's synthesis including FLA local-language estimates (may be int or tier).
        "analyst_coverage": _extract_data_block_float(
            fundamentals, "ANALYST_COVERAGE_ENGLISH"
        ),
        "analyst_coverage_total_est": _extract_data_block_field(
            fundamentals, "ANALYST_COVERAGE_TOTAL_EST"
        ),
        "profitability_trend": _extract_data_block_field(
            fundamentals, "PROFITABILITY_TREND"
        ),
        "52w_high": _extract_data_block_float(fundamentals, "52W_HIGH"),
        "52w_low": _extract_data_block_float(fundamentals, "52W_LOW"),
        # Special-situation routing: Senior promotes the FLA M&A EVENT section
        # into M_AND_A_STATUS so the IBKR reconciler can route held positions
        # as M&A EXIT rather than FUNDAMENTAL FAILURE. Empty/missing when no
        # active deal; values are ACTIVE_TENDER, RUMORED, or NONE.
        "m_and_a_status": _extract_data_block_field(fundamentals, "M_AND_A_STATUS"),
        # TRADE_BLOCK fields (structured for portfolio reconciliation)
        **trade_block_fields,
        # Bear thesis excerpt
        "bear_risks_excerpt": _extract_bear_risks(result),
        # The pipeline's only *falsifiable, pre-registered* statement of what
        # would break the thesis — the decision-journal remedy for judging a
        # past call. Recorded so a future evidence-backed post-mortem can test
        # it; the price-only retrospective may not adjudicate it (see
        # THESIS_NOT_EVALUATED).
        "kill_criteria": extract_kill_criteria(_bear_text(result)),
        # Hazards the analysis actually recorded, as bare type tokens.
        #
        # Noted, never scored: the retrospective has price and macro data only and
        # cannot test whether a flag came true. But the lesson prompt forbids
        # naming a mechanism absent from its inputs, and without this the *only*
        # hazards in scope were the bear case's — so a run flagged CMIC, PFIC or
        # VIE at decision time could not be referred to at all, while an invented
        # "governance bleed" read as equally grounded. Regulatory and currency
        # exposure are exactly the classes a benchmark cannot net out.
        "red_flags_at_decision": sorted(
            {
                str(flag.get("type"))
                for flag in (result.get("red_flags") or [])
                if isinstance(flag, Mapping) and flag.get("type")
            }
        ),
        # Exchange/currency/benchmark
        "exchange": suffix.lstrip(".") if suffix else "US",
        "currency": currency,
        "benchmark_index": benchmark,
        "fx_rate_to_usd": fx_rate,
        # Metadata (from existing save_data structure)
        "ticker": ticker,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        # Per-run identity. `analysis_date` alone collapses two analyses of one
        # ticker on one day into a single record — and that pair is exactly the
        # model/prompt-change comparison the retrospective exists to support.
        # None on paths that mint no run id; snapshot_identity() then falls back
        # to the (already unique) source filename.
        "analysis_id": analysis_id,
        # What the machine was: code commit, effective prompts, seat bindings,
        # thesis thresholds. Lives *in the snapshot* because compare_to_reality
        # loads only the snapshot — a sibling key in save_data would be invisible
        # to the consumer that needs it.
        "run_fingerprint": run_fingerprint,
        # Model names are provenance — they make runs comparable over time
        # ("did the verdict move because the model moved?"). The confidence
        # weighting deliberately does NOT key on them; see compute_confidence.
        "deep_model": active.reasoning,
        "quick_model": active.fast,
        "decision_model": active.decision,
        "decision_intent": active.decision_intent,
        "is_quick_mode": is_quick_mode,
        # `is_strict_mode` records whether `--strict` was active during
        # analysis. Strict gates auto-reject some valid candidates (REIT/ETF,
        # earnings quality) at the screening layer, so a non-BUY in strict mode
        # carries different signal than the same verdict in normal mode —
        # downstream lesson weighting can use this to discount strict-mode
        # rejections.
        "is_strict_mode": is_strict_mode,
        "trace_id": trace_id,
        "regime_at_decision": regime_at_decision,
        "regime_confidence": macro_regime.get("confidence")
        if regime_at_decision
        else None,
        # Macro provenance for the T0-vs-T1 comparison. Without the region a
        # later run cannot find the right cache file, and without the summarizer
        # fingerprint a changed macro *prompt* would read as a changed *world*.
        "macro_region": result.get("macro_context_region"),
        "macro_fingerprint": result.get("macro_context_fingerprint"),
        "macro_generated_at": result.get("macro_context_generated_at"),
    }

    logger.info(
        "prediction_snapshot_extracted",
        ticker=ticker,
        verdict=snapshot["verdict"],
        price=snapshot["current_price"],
        sector=snapshot["sector"],
    )

    return snapshot


# ══════════════════════════════════════════════════════════════════════════════
# Component 2: Retrospective Comparison
# ══════════════════════════════════════════════════════════════════════════════


def _should_emit_snapshot_progress(processed_files: int, total_files: int) -> bool:
    """Return True when snapshot-scan progress should be surfaced."""
    if total_files <= 0:
        return False
    if total_files > 20 and processed_files in {1, 5, 10, 25, 50, 100}:
        return True
    if total_files <= 20:
        return True
    if total_files <= 200:
        step = 25
    elif total_files <= 1000:
        step = 100
    else:
        step = 250
    return processed_files == total_files or processed_files % step == 0


@dataclass(frozen=True, slots=True)
class ReturnAttribution:
    """Two internally-exact views of a realized move; never a three-way split.

    **Local relative (additive).** ``market_return_pct + residual_return_pct ==
    price_return_pct`` by construction, since ``excess = price - benchmark``.
    This is the view the trigger thresholds already act on, and the *only* one in
    which "which leg dominates" is a meaningful question.

    **USD investor (multiplicative).** ``(1 + usd) = (1 + local) x (1 + fx)`` —
    the idiom ``fx_return_split`` already documents in
    ``ibkr/portfolio_presentation.py``. FX is reported here but never competes
    for ``dominant_driver``: it is a *conversion* effect and explains none of the
    local excess return. An earlier design treated market/FX/residual as three
    additive legs, which double-counts.

    ``residual_return_pct`` is deliberately **not** named alpha. With no sector
    benchmark it is "what the country index does not explain", which still
    contains sector-wide rotation; calling it alpha would licence exactly the
    overclaim this decomposition exists to prevent.
    """

    market_return_pct: float | None
    residual_return_pct: float | None
    fx_return_pct: float | None
    usd_investor_return_pct: float | None
    dominant_driver: str
    benchmark_available: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_return_pct": self.market_return_pct,
            "residual_return_pct": self.residual_return_pct,
            "fx_return_pct": self.fx_return_pct,
            "usd_investor_return_pct": self.usd_investor_return_pct,
            "dominant_driver": self.dominant_driver,
            "benchmark_available": self.benchmark_available,
        }


@dataclass(frozen=True, slots=True)
class CachedRegimeDelta:
    """Whether the macro regime moved between a decision and now.

    Named for what it actually is. A macro brief is an advisory LLM
    classification, and the on-disk cache refreshes only when some analysis
    happens to run in that region — so "now" may be days old. This is a *cached*
    comparison, not a measurement of the world.

    ``shifted`` is tri-state and ``None`` is load-bearing: this repository has
    been bitten before by an unknown masquerading as a negative (the
    ``is_quick_mode`` tri-state in ``ibkr/reconciliation_rules``). Unknown carries
    no authority — it neither scopes a lesson nor clears one.
    """

    shifted: bool | None
    shift_reason: str
    regime_now: dict[str, str] | None = None
    staleness_days: int | None = None
    t1_generated_at: str | None = None
    t1_fingerprint: str | None = None
    region: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shifted": self.shifted,
            "shift_reason": self.shift_reason,
            "regime_now": self.regime_now,
            "staleness_days": self.staleness_days,
            "t1_generated_at": self.t1_generated_at,
            "t1_fingerprint": self.t1_fingerprint,
            "region": self.region,
        }


def _unknown_regime_delta(reason: str, **fields: Any) -> CachedRegimeDelta:
    return CachedRegimeDelta(shifted=None, shift_reason=reason, **fields)


def _regime_confidence_usable(confidence: Any) -> bool:
    value = str(confidence or "").strip().upper()
    return bool(value) and value != REGIME_UNUSABLE_CONFIDENCE


def resolve_cached_regime_delta(
    snapshot: Mapping[str, Any],
    cache_dir: Path | None = None,
) -> CachedRegimeDelta:
    """Compare the decision-time regime (T0) with the cached current one (T1).

    Reads the *existing* on-disk macro cache: zero LLM cost, zero new network
    calls, and the cache is keyed by region so two tickers in one region share a
    T1 — which is correct, and cheaper than resolving per ticker.

    Fails to ``None`` (unknown) rather than ``False`` on every degraded path:
    missing cache, stale cache, corrupt JSON, absent T0, low confidence on either
    side, or a macro-prompt fingerprint mismatch. The last is the subtle one — if
    the summarizer prompt changed between the two briefs, a differing label says
    the classifier moved, not the world.
    """
    regime_at_decision = snapshot.get("regime_at_decision")
    if not isinstance(regime_at_decision, Mapping) or not regime_at_decision:
        return _unknown_regime_delta("no regime recorded at decision time")

    # Prefer the region the decision actually used. Re-deriving it from the
    # ticker would follow a *later* mapping change and read a different region's
    # brief than the one that informed the analysis.
    region = str(snapshot.get("macro_region") or "").strip()
    ticker = str(snapshot.get("ticker") or "")
    if not region:
        if not ticker:
            return _unknown_regime_delta("no ticker or recorded macro region")
        try:
            from src.macro_regions import infer_macro_region

            region = infer_macro_region(ticker)
        except Exception:
            return _unknown_regime_delta("macro region could not be inferred")

    if cache_dir is None:
        try:
            from src.macro_context import get_macro_context_cache_dir

            cache_dir = get_macro_context_cache_dir()
        except Exception:
            return _unknown_regime_delta("macro cache directory unavailable")

    cache_path = Path(cache_dir) / f"{region}.json"
    try:
        with open(cache_path) as handle:
            cached = json.load(handle)
    except FileNotFoundError:
        return _unknown_regime_delta(
            f"no cached macro brief for {region}", region=region
        )
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return _unknown_regime_delta(
            f"cached macro brief for {region} is unreadable", region=region
        )
    if not isinstance(cached, dict):
        return _unknown_regime_delta(
            f"cached macro brief for {region} is malformed", region=region
        )

    t1_fingerprint = cached.get("fingerprint")
    t1_generated_at = cached.get("generated_at")
    staleness_days = _cache_staleness_days(cached)
    common: dict[str, Any] = {
        "region": region,
        "staleness_days": staleness_days,
        "t1_generated_at": t1_generated_at,
        "t1_fingerprint": t1_fingerprint,
    }

    if staleness_days is None:
        return _unknown_regime_delta("cached macro brief has no usable date", **common)
    if staleness_days > REGIME_STALENESS_MAX_DAYS:
        return _unknown_regime_delta(
            f"cached macro brief is {staleness_days}d old "
            f"(> {REGIME_STALENESS_MAX_DAYS}d); not 'now'",
            **common,
        )

    # Fail closed on comparability. An absent fingerprint on either side means we
    # cannot tell whether the summarizer prompt moved, and a classifier change is
    # indistinguishable from a world change without it — which is the specific
    # error this guard exists to prevent. An earlier revision compared anyway
    # when either side was missing, to keep the legacy corpus usable; that
    # optimizes coverage over correctness, and "unknown" is the honest answer for
    # a snapshot written before the provenance existed.
    t0_fingerprint = snapshot.get("macro_fingerprint")
    if not t0_fingerprint:
        return _unknown_regime_delta(
            "no macro summarizer fingerprint recorded at decision time", **common
        )
    if not t1_fingerprint:
        return _unknown_regime_delta(
            "cached macro brief carries no summarizer fingerprint", **common
        )
    if t0_fingerprint != t1_fingerprint:
        # The classifier changed, so a differing label is not evidence about the
        # world. Deliberately not treated as a shift.
        return _unknown_regime_delta(
            "macro summarizer prompt changed between the two briefs", **common
        )

    try:
        from src.macro_regime import parse_macro_regime

        regime_now = parse_macro_regime(str(cached.get("report") or ""))
    except Exception:
        return _unknown_regime_delta("cached macro brief could not be parsed", **common)

    if not regime_now.present:
        return _unknown_regime_delta(
            "cached macro brief carries no regime block", **common
        )

    now_dict = regime_now.to_dict()
    common["regime_now"] = {k: str(v) for k, v in now_dict.items()}

    if not _regime_confidence_usable(
        snapshot.get("regime_confidence") or regime_at_decision.get("confidence")
    ):
        return _unknown_regime_delta("decision-time regime confidence is LOW", **common)
    if not _regime_confidence_usable(regime_now.confidence):
        return _unknown_regime_delta("current regime confidence is LOW", **common)

    changes = []
    for field_name in REGIME_COMPARED_FIELDS:
        before = str(regime_at_decision.get(field_name) or "").upper()
        after = str(now_dict.get(field_name) or "").upper()
        if not before or not after:
            return _unknown_regime_delta(
                f"{field_name} missing on one side of the comparison", **common
            )
        if before != after:
            label = field_name.replace("_", " ")
            changes.append(f"{label}: {before} -> {after}")

    if not changes:
        return CachedRegimeDelta(
            shifted=False, shift_reason="no change in risk appetite or shock", **common
        )
    return CachedRegimeDelta(shifted=True, shift_reason="; ".join(changes), **common)


def _cache_staleness_days(cached: Mapping[str, Any]) -> int | None:
    """Age of the cached brief in days, preferring its trade date."""
    raw_date = cached.get("trade_date")
    if raw_date:
        try:
            parsed = datetime.strptime(str(raw_date), "%Y-%m-%d")
        except ValueError:
            parsed = None
        if parsed is not None:
            return max((datetime.now() - parsed).days, 0)

    generated_at = cached.get("generated_at")
    if generated_at:
        try:
            parsed_dt = datetime.fromisoformat(str(generated_at))
        except ValueError:
            return None
        if parsed_dt.tzinfo is not None:
            parsed_dt = parsed_dt.replace(tzinfo=None)
        return max((datetime.now() - parsed_dt).days, 0)
    return None


def lesson_scope_for(dominant_driver: str) -> str:
    """How far a lesson drawn from this outcome may generalize.

    Only a MARKET-dominated move earns ``CONTEXTUAL``. ``MIXED`` means neither
    leg dominates by ``ATTRIBUTION_DOMINANCE_RATIO`` — the move could not be
    attributed — and an unattributed move is not a regime observation. An
    unrecognized driver falls here too: authority defaults to withheld.

    The earlier ``else CONTEXTUAL`` did more damage than a mislabelled record.
    ``_render_attribution`` prints the scope into the lesson prompt, and the
    prompt's "if the scope is UNRESOLVED, the residual is unexplained, not
    diagnosed" rule keys on it — so every MIXED outcome was generated with that
    rule switched off. Measured 2026-08-16: 95 RESIDUAL, 67 MIXED, **zero
    MARKET**, and all 67 CONTEXTUAL lessons asserted a cause that was never
    established ("verify cash flow reconciliations" from a price move alone).

    ``UNRESOLVED`` concedes that the move was stock-specific and the cause
    unknown. It is emphatically not ``VALIDATED`` — establishing *why* needs
    company evidence published after the decision, which a price-only
    retrospective does not have. UNRESOLVED lessons are stored for review and
    never injected; see the eligibility gate in ``get_relevant_lessons``.
    """
    return SCOPE_CONTEXTUAL if dominant_driver == DRIVER_MARKET else SCOPE_UNRESOLVED


def lesson_eligibility(comparison: Mapping[str, Any]) -> tuple[str, str]:
    """May this lesson be handed to a live analysis? Returns ``(eligibility, reason)``.

    Separate from ``lesson_scope_for`` because attribution alone is not enough. A
    CONTEXTUAL lesson is retrieved only when its stored regime matches the present
    one, so a record with no recorded regime is unreachable no matter what its
    text says. That gap is what let 67 records look injectable while being inert.

    The reason is returned and persisted because ``REVIEW_ONLY`` is otherwise
    ambiguous on its face: "the driver was MIXED" is a finding about the snapshot,
    while "the macro cache was stale when this ran" is a finding about the run.
    Only one of them says anything about the lesson.
    """
    # Checked first, and deliberately so: a live deal prices the stock against its
    # terms, not against its market. The benchmark decomposition still computes
    # and means nothing, so whichever leg "dominates" is an artefact of deal
    # mechanics — reporting `driver MIXED` here would name a symptom and hide the
    # cause. `m_and_a_status` is already carried in the snapshot for the IBKR
    # reconciler; RUMORED is not disqualifying, since a rumour does not pin price.
    if str(comparison.get("m_and_a_status") or "").strip().upper() == "ACTIVE_TENDER":
        return (
            LESSON_ELIGIBILITY_REVIEW_ONLY,
            "an active tender priced the stock against deal terms, not the market",
        )

    attribution = comparison.get("attribution")
    attribution = attribution if isinstance(attribution, Mapping) else {}
    driver = str(attribution.get("dominant_driver") or DRIVER_UNKNOWN)
    if driver != DRIVER_MARKET:
        return (
            LESSON_ELIGIBILITY_REVIEW_ONLY,
            f"driver {driver}: the move was not attributed to the market",
        )

    regime = comparison.get("regime_at_decision")
    regime = regime if isinstance(regime, Mapping) else {}
    # Exactly the fields `_regime_matches` compares. Without one of them the
    # CONTEXTUAL stamp is decorative: retrieval can never match it.
    if not any(
        str(regime.get(field) or "").strip() for field in REGIME_COMPARED_FIELDS
    ):
        return (
            LESSON_ELIGIBILITY_REVIEW_ONLY,
            "no decision-time regime recorded, so no regime can ever match it",
        )

    delta = comparison.get("cached_regime_delta")
    delta = delta if isinstance(delta, Mapping) else {}
    shifted = delta.get("shifted")
    if shifted is not False:
        # UNKNOWN is withheld, not permitted. `shifted` is False only when both
        # regimes were usable, comparable and equal; every degraded path returns
        # None. "We could not establish that the regime held still" is not a
        # basis for authorizing guidance about that regime.
        detail = str(delta.get("shift_reason") or "").strip()
        state = "the regime shifted" if shifted is True else "regime shift unknown"
        return (
            LESSON_ELIGIBILITY_REVIEW_ONLY,
            f"{state}{f': {detail}' if detail else ''}",
        )

    return (
        LESSON_ELIGIBILITY_INJECTABLE,
        "market-dominated outcome in a stable regime",
    )


def _dominant_local_driver(market_pct: float, residual_pct: float) -> str:
    """Which of the two *local* legs explains the move, if either."""
    market_magnitude = abs(market_pct)
    residual_magnitude = abs(residual_pct)
    if market_magnitude > residual_magnitude * ATTRIBUTION_DOMINANCE_RATIO:
        return DRIVER_MARKET
    if residual_magnitude > market_magnitude * ATTRIBUTION_DOMINANCE_RATIO:
        return DRIVER_RESIDUAL
    return DRIVER_MIXED


def attribute_return(
    *,
    price_return_pct: float,
    benchmark_return_pct: float | None,
    fx_delta_pct: float | None,
) -> ReturnAttribution:
    """Deterministic split. The lesson LLM is told the answer; it never infers it.

    ``benchmark_return_pct is None`` means the index could not be fetched — which
    is emphatically not the same as an index that did not move. Reporting it as
    ``0.0`` (the pre-existing behaviour) makes ``excess`` equal the raw stock
    return, so a stock down 35% in a market down 30% reads as a company-specific
    collapse. That is the overclaim this whole decomposition exists to prevent, so
    the market leg stays ``None`` and the driver is ``UNKNOWN``.
    """
    fx_pct = None if fx_delta_pct is None else float(fx_delta_pct)
    usd_investor_pct: float | None = None
    if fx_pct is not None:
        # Multiplicative, not additive: a 10% local gain in a currency that fell
        # 10% is not a flat USD outcome.
        usd_investor_pct = (
            (1.0 + price_return_pct / 100.0) * (1.0 + fx_pct / 100.0) - 1.0
        ) * 100.0

    if benchmark_return_pct is None:
        return ReturnAttribution(
            market_return_pct=None,
            residual_return_pct=None,
            fx_return_pct=fx_pct,
            usd_investor_return_pct=usd_investor_pct,
            dominant_driver=DRIVER_UNKNOWN,
            benchmark_available=False,
        )

    market_pct = float(benchmark_return_pct)
    residual_pct = price_return_pct - market_pct
    return ReturnAttribution(
        market_return_pct=round(market_pct, 2),
        residual_return_pct=round(residual_pct, 2),
        fx_return_pct=fx_pct,
        usd_investor_return_pct=(
            None if usd_investor_pct is None else round(usd_investor_pct, 2)
        ),
        dominant_driver=_dominant_local_driver(market_pct, residual_pct),
        benchmark_available=True,
    )


def snapshot_identity(snapshot: Mapping[str, Any]) -> str:
    """Return a stable per-*run* identity for a prediction snapshot.

    Two analyses of one ticker on one day are two analyses, not one — that pair is
    precisely the model/prompt-change comparison this system exists to support, so
    identity may not collapse to ``(ticker, analysis_date)``.

    Modern snapshots carry ``analysis_id`` (the run id minted in
    ``persistence.save_results_to_file``, which is the Langfuse trace id when
    tracing is on). Legacy snapshots fall back to the source filename, which is
    ``TICKER_YYYYMMDD_HHMMSS_analysis.json`` and therefore already unique per run.
    The final fallback is the old composite key, used only when a snapshot reaches
    us with no source attached at all.
    """
    analysis_id = snapshot.get("analysis_id")
    if analysis_id:
        return str(analysis_id)
    source_file = snapshot.get("_source_file")
    if source_file:
        return str(source_file).removesuffix(".json").removesuffix("_analysis")
    return f"{snapshot.get('ticker')}|{snapshot.get('analysis_date')}"


def _snapshot_days_elapsed(snapshot: Mapping[str, Any]) -> int | None:
    """Days between the snapshot's analysis date and now; None if unparseable."""
    raw = snapshot.get("analysis_date")
    if not raw:
        return None
    try:
        analysis_date = datetime.strptime(str(raw), "%Y-%m-%d")
    except ValueError:
        return None
    return (datetime.now() - analysis_date).days


class EvaluationMemo:
    """Records which snapshots have already been priced, and at what age.

    The retrospective's only dedup used to be "does a lesson exist for this
    snapshot", which says nothing about the far more common case: a snapshot that
    *was* evaluated and simply did not clear its trigger threshold. Those left no
    trace, so each one paid a fresh pair of yfinance round-trips on every
    subsequent run, forever.

    Fail-open by construction: a missing, unreadable or corrupt memo means
    "evaluate everything" (the pre-existing behaviour), never "evaluate nothing".
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_EVALUATION_MEMO_PATH
        self._entries: dict[str, dict[str, Any]] = self._load()
        self._dirty = False

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self.path) as handle:
                loaded = json.load(handle)
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            logger.warning(
                "retrospective_memo_unreadable",
                path=str(self.path),
                reason=type(exc).__name__,
                msg="Evaluating every snapshot this run; memo will be rewritten",
            )
            return {}
        if not isinstance(loaded, dict):
            logger.warning(
                "retrospective_memo_malformed",
                path=str(self.path),
                reason=f"expected an object, found {type(loaded).__name__}",
            )
            return {}
        return {key: value for key, value in loaded.items() if isinstance(value, dict)}

    def should_evaluate(self, identity: str, days_elapsed: int | None) -> bool:
        """True when this snapshot has not been priced recently enough."""
        seen = self._entries.get(identity)
        if seen is None:
            return True
        # A benchmark that could not be fetched is a transient outage, not a
        # verdict on the snapshot — retry on the next run rather than in 30 days.
        if seen.get("outcome") == MEMO_OUTCOME_NO_DATA:
            return True
        if days_elapsed is None:
            return True
        try:
            evaluated_at = int(seen.get("evaluated_at_days", 0))
        except (TypeError, ValueError):
            return True
        return days_elapsed - evaluated_at >= RETROSPECTIVE_REEVALUATION_INTERVAL_DAYS

    def record(
        self,
        identity: str,
        *,
        ticker: str,
        analysis_date: str,
        days_elapsed: int | None,
        outcome: str,
    ) -> None:
        """Note that this snapshot was priced. Held in memory until ``flush()``."""
        self._entries[identity] = {
            "ticker": ticker,
            "analysis_date": analysis_date,
            "evaluated_at_days": int(days_elapsed or 0),
            "outcome": outcome,
            "evaluated_on": datetime.now().strftime("%Y-%m-%d"),
        }
        self._dirty = True

    def flush(self) -> None:
        """Persist the memo. Never raises — a lost memo costs time, not results."""
        if not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
            with open(tmp_path, "w") as handle:
                json.dump(self._entries, handle, indent=2, sort_keys=True)
            tmp_path.replace(self.path)
            self._dirty = False
        except Exception as exc:
            logger.warning(
                "retrospective_memo_write_failed",
                path=str(self.path),
                **summarize_exception(exc, operation="retrospective memo write"),
            )


def load_past_snapshots(
    ticker: str | None,
    results_dir: Path,
    *,
    archive_dirs: Sequence[Path] = (),
    progress: Callable[[SnapshotLoadProgress], None] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Load prediction snapshots from saved analysis JSON files.

    Args:
        ticker: If provided, only load snapshots for this ticker.
                If None, load all tickers found.
        results_dir: Directory containing analysis JSON files.
        archive_dirs: Read-only directories of archived artifacts, scanned after
            ``results_dir``. Local retention moves artifacts out at ~120 days,
            inside the 90-270 day band ``TEMPORAL_WEIGHTS`` scores highest, so
            without these the best evidence is unreachable. A directory that does
            not exist is skipped rather than raising, mirroring
            ``scripts/eval_longitudinal_compare.py --archive-dir``.

    Returns:
        Dict mapping ticker -> list of snapshots (sorted by date descending)
    """
    snapshots: dict[str, list[dict[str, Any]]] = {}

    if not results_dir.exists():
        logger.warning("results_dir_not_found", path=str(results_dir))
        # An archive may still be readable, so this is no longer fatal — but with
        # nothing configured it degrades to exactly the old behaviour.
        if not archive_dirs:
            return snapshots

    # Build pattern based on ticker filter
    if ticker:
        safe_ticker = ticker.replace(".", "_").replace("/", "_")
        pattern = f"{safe_ticker}_*_analysis.json"
        # Also try with dot notation for older files
        pattern2 = f"{ticker}_*_analysis.json"
    else:
        pattern = "*_analysis.json"
        pattern2 = None

    # Live results first: the ordering *is* the precedence rule, since the first
    # artifact seen for a given identity wins.
    search_dirs = [results_dir]
    for archive_dir in archive_dirs:
        archive_path = Path(archive_dir)
        if archive_path in search_dirs:
            continue
        if not archive_path.is_dir():
            logger.debug("archive_dir_skipped", path=str(archive_path))
            continue
        search_dirs.append(archive_path)

    files: list[Path] = []
    seen: set[str] = set()
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        found = sorted(search_dir.glob(pattern), reverse=True)
        if pattern2:
            found.extend(sorted(search_dir.glob(pattern2), reverse=True))
        for candidate in found:
            # Deduplicate by filename: the same artifact copied into an archive
            # keeps its name, and the live tree is scanned first.
            if candidate.name in seen:
                continue
            seen.add(candidate.name)
            files.append(candidate)

    total_files = len(files)
    seen_identities: set[str] = set()
    if progress is not None:
        progress(
            SnapshotLoadProgress(
                phase="discovered",
                total_files=total_files,
                processed_files=0,
                loaded_tickers=0,
                loaded_snapshots=0,
                current_file=None,
            )
        )

    for processed_files, filepath in enumerate(files, start=1):

        def emit_progress(
            processed_files_: int = processed_files,
            current_file: str = filepath.name,
        ) -> None:
            if progress is None or not _should_emit_snapshot_progress(
                processed_files_, total_files
            ):
                return
            loaded_snapshots = sum(len(items) for items in snapshots.values())
            progress(
                SnapshotLoadProgress(
                    phase="parsing",
                    total_files=total_files,
                    processed_files=processed_files_,
                    loaded_tickers=len(snapshots),
                    loaded_snapshots=loaded_snapshots,
                    current_file=current_file,
                )
            )

        try:
            with open(filepath) as handle:
                data = json.load(handle)

            snapshot = data.get("prediction_snapshot")
            if not snapshot:
                logger.debug(
                    "no_snapshot_in_file",
                    file=filepath.name,
                    reason="predates retrospective feature",
                )
                emit_progress()
                continue

            snap_ticker = snapshot.get("ticker", "UNKNOWN")
            if snap_ticker not in snapshots:
                snapshots[snap_ticker] = []
            # Attach source file before deriving identity — it is the legacy
            # fallback for snapshots written before analysis_id existed.
            snapshot["_source_file"] = filepath.name
            identity = snapshot_identity(snapshot)
            if identity in seen_identities:
                # Same run re-saved under a different filename (a live artifact
                # and its archived copy). Two *distinct* same-day runs carry
                # different identities and both survive — that pair is the whole
                # point of tracking identity rather than date.
                logger.debug(
                    "duplicate_snapshot_identity_skipped",
                    file=filepath.name,
                    identity=identity,
                )
                emit_progress()
                continue
            seen_identities.add(identity)
            snapshots[snap_ticker].append(snapshot)
            emit_progress()

        except json.JSONDecodeError:
            logger.warning("malformed_json", file=filepath.name)
            emit_progress()
        except Exception as e:
            logger.warning(
                "snapshot_load_error",
                file=filepath.name,
                exc_info=True,
                **summarize_exception(e, operation="snapshot_load"),
            )
            emit_progress()

    if progress is not None:
        progress(
            SnapshotLoadProgress(
                phase="complete",
                total_files=total_files,
                processed_files=total_files,
                loaded_tickers=len(snapshots),
                loaded_snapshots=sum(len(items) for items in snapshots.values()),
                current_file=files[-1].name if files else None,
            )
        )

    return snapshots


async def compare_to_reality(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    """
    Compare a past prediction snapshot to current market reality.

    Fetches current price + benchmark return via yfinance. Computes excess
    return and determines if the delta exceeds lesson-trigger thresholds.

    Args:
        snapshot: Prediction snapshot dict from extract_snapshot()

    Returns:
        Comparison dict if threshold exceeded, None otherwise.
        Also returns None if data fetch fails or elapsed days < 30.

        A third case: when the benchmark could not be fetched, the outcome is
        *unassessable* rather than below threshold. The returned dict carries
        ``UNASSESSED_REASON_KEY`` and must never reach lesson generation — the
        stock's own return alone cannot distinguish a company collapse from a
        market-wide one, and scoring it as if the index were flat is precisely
        how a market crash becomes a lesson about a company.
    """
    import asyncio

    ticker = snapshot.get("ticker")
    analysis_date_str = snapshot.get("analysis_date")
    verdict = snapshot.get("verdict")
    snapshot_price = snapshot.get("current_price")

    if not ticker or not analysis_date_str or not verdict:
        logger.debug("incomplete_snapshot", ticker=ticker)
        return None

    logger.debug(
        "comparison_starting",
        ticker=ticker,
        analysis_date=analysis_date_str,
        verdict=verdict,
        snapshot_price=snapshot_price,
    )

    # Parse analysis date
    try:
        analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d")
    except ValueError:
        logger.debug("invalid_date", date=analysis_date_str)
        return None

    # Shared with the orchestrator, which needs the age before deciding whether
    # this snapshot is worth a fetch at all.
    days_elapsed = _snapshot_days_elapsed(snapshot) or 0
    if days_elapsed < MINIMUM_DAYS_ELAPSED:
        logger.debug(
            "too_recent", ticker=ticker, days=days_elapsed, min=MINIMUM_DAYS_ELAPSED
        )
        return None

    # Fetch current price and benchmark via yfinance
    try:
        import yfinance as yf

        def _fetch_current_data() -> _RetrospectiveMarketData:
            result: _RetrospectiveMarketData = {}

            # Current stock price (adjusted close for total return)
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(
                    start=analysis_date.strftime("%Y-%m-%d"),
                    end=datetime.now().strftime("%Y-%m-%d"),
                )
                if len(hist) >= 2:
                    result["start_adj_close"] = float(hist["Close"].iloc[0])
                    result["end_adj_close"] = float(hist["Close"].iloc[-1])
                else:
                    # Fallback: use info
                    info = stock.info
                    current = info.get("currentPrice") or info.get("regularMarketPrice")
                    if current:
                        result["end_adj_close"] = float(current)
                        if snapshot_price:
                            result["start_adj_close"] = float(snapshot_price)
            except Exception as e:
                logger.debug("stock_fetch_failed", ticker=ticker, error=str(e))

            # Benchmark return over same period
            benchmark = snapshot.get("benchmark_index", FALLBACK_BENCHMARK)
            try:
                bench = yf.Ticker(benchmark)
                bench_hist = bench.history(
                    start=analysis_date.strftime("%Y-%m-%d"),
                    end=datetime.now().strftime("%Y-%m-%d"),
                )
                if len(bench_hist) >= 2:
                    result["bench_start"] = float(bench_hist["Close"].iloc[0])
                    result["bench_end"] = float(bench_hist["Close"].iloc[-1])
            except Exception as e:
                logger.debug(
                    "benchmark_fetch_failed", benchmark=benchmark, error=str(e)
                )
                # Fallback to S&P 500 if primary benchmark fails
                if benchmark != FALLBACK_BENCHMARK:
                    try:
                        bench = yf.Ticker(FALLBACK_BENCHMARK)
                        bench_hist = bench.history(
                            start=analysis_date.strftime("%Y-%m-%d"),
                            end=datetime.now().strftime("%Y-%m-%d"),
                        )
                        if len(bench_hist) >= 2:
                            result["bench_start"] = float(bench_hist["Close"].iloc[0])
                            result["bench_end"] = float(bench_hist["Close"].iloc[-1])
                            result["benchmark_fallback"] = FALLBACK_BENCHMARK
                    except Exception:
                        pass

            return result

        data = await run_with_hard_timeout(
            asyncio.to_thread(_fetch_current_data),
            timeout=15.0,
            label=f"retrospective_yfinance:{ticker}",
        )

    except asyncio.TimeoutError:
        logger.warning("yfinance_timeout", ticker=ticker)
        return None
    except Exception as e:
        details = classify_failure(e, provider="unknown", class_name="yfinance")
        logger.warning(
            "yfinance_error",
            ticker=ticker,
            failure_kind=details.kind,
            host=details.host,
            error_type=details.error_type,
            root_cause_type=details.root_cause_type,
            retryable=details.retryable,
            error_message=details.message,
            exc_info=True,
        )
        return None

    # Calculate returns.
    # Both start_price / end_price are in LOCAL currency (yfinance returns local prices),
    # and both bench_start / bench_end are in the benchmark index's native unit.
    # The percentage formula (end-start)/start cancels the currency unit, so
    # price_return_pct and benchmark_return_pct are currency-neutral percentages
    # and can be compared directly without FX conversion.
    start_price = data.get("start_adj_close")
    end_price = data.get("end_adj_close")
    if not start_price or not end_price or start_price <= 0:
        logger.debug("insufficient_price_data", ticker=ticker)
        return None

    price_return_pct = (
        (end_price - start_price) / start_price
    ) * 100.0  # % in LOCAL ccy

    bench_start = data.get("bench_start")
    bench_end = data.get("bench_end")
    measured_benchmark_return_pct: float | None = None
    if bench_start and bench_end and bench_start > 0:
        measured_benchmark_return_pct = (
            (bench_end - bench_start) / bench_start
        ) * 100.0  # % unitless

    # `benchmark_return_pct` stays 0.0 in the persisted dict for display
    # compatibility with existing consumers. No *decision* reads it: the
    # attribution below carries the truth, and an unavailable benchmark now
    # short-circuits the whole comparison rather than being scored as a flat
    # market.
    benchmark_return_pct = measured_benchmark_return_pct or 0.0
    excess_return_pct = price_return_pct - benchmark_return_pct

    # FX delta, tri-state. `None` means "not determined" and must never be
    # rendered or attributed as a flat rate — see the FX_* tokens.
    fx_delta_pct: float | None = None
    currency = snapshot.get("currency", FALLBACK_CURRENCY)
    snapshot_fx = snapshot.get("fx_rate_to_usd")
    if currency == "USD":
        # A real zero: a USD-denominated security has no FX leg at all.
        fx_delta_pct, fx_observation = 0.0, FX_NOT_APPLICABLE
    else:
        fx_observation = FX_UNAVAILABLE
        if snapshot_fx and snapshot_fx > 0:
            try:
                from src.fx_normalization import get_fx_rate_yfinance

                current_fx = await get_fx_rate_yfinance(currency, "USD")
                if current_fx:
                    fx_delta_pct = ((current_fx - snapshot_fx) / snapshot_fx) * 100.0
                    fx_observation = FX_OBSERVED
            except Exception:
                # Still informational, still not critical — but an outage is not
                # an observation of stability.
                pass

    attribution = attribute_return(
        price_return_pct=price_return_pct,
        benchmark_return_pct=measured_benchmark_return_pct,
        fx_delta_pct=fx_delta_pct,
    )

    if not attribution.benchmark_available:
        # Unassessable, not below threshold. Without the index there is no way to
        # tell a company-specific collapse from a market-wide one, and the
        # thresholds below would be applied to the raw stock return as though the
        # market had been flat.
        logger.info(
            "retrospective_benchmark_unavailable",
            ticker=ticker,
            benchmark=snapshot.get("benchmark_index"),
            price_return=f"{price_return_pct:.1f}%",
            msg="Outcome unassessable; no lesson will be generated",
        )
        return {
            **snapshot,
            UNASSESSED_REASON_KEY: UNASSESSED_BENCHMARK,
            "price_return_pct": round(price_return_pct, 2),
            "days_elapsed": days_elapsed,
            "attribution": attribution.to_dict(),
        }

    # Check thresholds
    thresholds = THRESHOLDS.get(verdict, THRESHOLDS.get("HOLD", {}))
    wrong_threshold = thresholds.get("wrong", 25.0)
    understated_threshold = thresholds.get("understated", 40.0)

    triggered = False
    if verdict in ("BUY",):
        # Wrong direction: big loss
        if excess_return_pct < -wrong_threshold:
            triggered = True
        # Right but understated: huge gain
        elif excess_return_pct > understated_threshold:
            triggered = True
    elif verdict in ("DO_NOT_INITIATE", "SELL"):
        # Wrong direction: stock went up a lot
        if excess_return_pct > wrong_threshold:
            triggered = True
        # Right but understated: crashed even more
        elif excess_return_pct < -understated_threshold:
            triggered = True
    elif verdict in ("HOLD",):
        if abs(excess_return_pct) > wrong_threshold:
            triggered = True

    if not triggered:
        logger.debug(
            "below_threshold",
            ticker=ticker,
            excess_return=f"{excess_return_pct:.1f}%",
            verdict=verdict,
        )
        return None

    comparison = {
        **snapshot,
        "price_return_pct": round(price_return_pct, 2),
        "benchmark_return_pct": round(benchmark_return_pct, 2),
        "excess_return_pct": round(excess_return_pct, 2),
        # Legacy key, kept float for display compatibility exactly as
        # `benchmark_return_pct` is: 0.0 when undetermined, with the adjacent
        # token carrying the truth. No decision reads the number.
        "fx_delta_pct": round(fx_delta_pct, 2) if fx_delta_pct is not None else 0.0,
        "fx_observation": fx_observation,
        "days_elapsed": days_elapsed,
        "start_price": round(start_price, 4),
        "end_price": round(end_price, 4),
        "benchmark_used": data.get(
            "benchmark_fallback", snapshot.get("benchmark_index")
        ),
        "attribution": attribution.to_dict(),
        "lesson_scope": lesson_scope_for(attribution.dominant_driver),
        # Never anything else from this code path — see THESIS_NOT_EVALUATED.
        "thesis_validation_status": THESIS_NOT_EVALUATED,
        # Read-only, from the existing on-disk macro cache: no LLM call, no fetch.
        "cached_regime_delta": resolve_cached_regime_delta(snapshot).to_dict(),
        # Provenance, not causation: CHANGED says the two runs are not directly
        # comparable. It never asserts the tooling change caused the outcome.
        #
        # Per-axis, not the strict scalar. During a retrospective no analysis
        # prompts are in use, so `_comparison_context` is structurally always
        # UNKNOWN and storing it would waste the field — while the run *can*
        # know that the code, models or thresholds moved.
        "comparison_context": _render_comparison_context(
            {"run_fingerprint": snapshot.get("run_fingerprint")}
        ),
    }

    logger.info(
        "significant_delta_detected",
        ticker=ticker,
        verdict=verdict,
        excess_return=f"{excess_return_pct:.1f}%",
        dominant_driver=attribution.dominant_driver,
        days=days_elapsed,
    )

    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# Component 3: Confidence Weighting
# ══════════════════════════════════════════════════════════════════════════════


def compute_confidence(comparison: dict[str, Any]) -> float:
    """
    Compute composite confidence score for a lesson.

    confidence = temporal × model_quality × mode × signal_strength

    Args:
        comparison: Comparison dict from compare_to_reality()

    Returns:
        Float between 0.0 and 1.0
    """
    days = comparison.get("days_elapsed", 0)

    # Temporal component
    temporal = TEMPORAL_STALE
    for max_days, weight in TEMPORAL_WEIGHTS:
        if days <= max_days:
            temporal = weight
            break

    # Analysis-capability component, from the decision seat's intent tier.
    model_q = INTENT_QUALITY.get(
        str(comparison.get("decision_intent") or ""), UNKNOWN_INTENT_QUALITY
    )

    # Analysis mode: prefer explicit flag (modern snapshots); fall back to
    # model-name heuristic for snapshots predating is_quick_mode field.
    if "is_quick_mode" in comparison:
        mode = 0.7 if comparison["is_quick_mode"] else 1.0
    else:
        quick_model = comparison.get("quick_model", "")
        deep = comparison.get("deep_model", "")
        mode = 0.7 if quick_model == deep else 1.0

    # Strict gates auto-reject some valid candidates (REIT/ETF, earnings quality)
    # at the screening layer, so a strict-mode verdict is partly an artifact of
    # the gates rather than a pure quality signal. `save_rejection_record` has
    # discounted rejection records by 0.7 for this reason since it was written;
    # outcome lessons were simply never given the same term.
    if comparison.get("is_strict_mode"):
        mode *= STRICT_MODE_CONFIDENCE_FACTOR

    # Signal strength component (bigger deltas = clearer lessons)
    excess = abs(comparison.get("excess_return_pct", 0.0))
    signal = min(excess / 30.0, 1.0)

    confidence = temporal * model_q * mode * signal
    final = round(min(max(confidence, 0.0), 1.0), 3)
    logger.debug(
        "confidence_computed",
        ticker=comparison.get("ticker"),
        confidence=final,
        temporal=round(temporal, 2),
        model_q=round(model_q, 2),
        mode_factor=round(mode, 2),
        signal=round(signal, 2),
    )
    return float(final)


def _prediction_is_directionally_correct(comparison: dict[str, Any]) -> bool:
    """Return a simple correctness judgment for deferred retrospective scoring."""
    verdict = (comparison.get("verdict") or "").upper()
    excess_return_pct = float(comparison.get("excess_return_pct") or 0.0)

    if verdict == "BUY":
        return excess_return_pct > 0
    if verdict in {"SELL", "DO_NOT_INITIATE"}:
        return excess_return_pct < 0
    if verdict == "HOLD":
        wrong_threshold = THRESHOLDS.get("HOLD", {}).get("wrong", 25.0)
        return abs(excess_return_pct) <= wrong_threshold
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Component 4: Lesson Generation (single LLM call)
# ══════════════════════════════════════════════════════════════════════════════


def _comparison_context(snapshot: Mapping[str, Any]) -> str:
    """Was the machine the same when this decision was made as it is now?

    Deliberately strict: an unresolvable axis on either side yields ``UNKNOWN``,
    because the unknown axis could be the one that moved. During a retrospective
    no analysis prompts are in use, so this is usually ``UNKNOWN`` — which is the
    honest scalar. ``_render_comparison_context`` gives the model the per-axis
    detail that strictness necessarily hides.
    """
    try:
        from src.run_fingerprint import (
            CONTEXT_UNKNOWN,
            RunFingerprint,
            compute_run_fingerprint,
        )

        recorded = RunFingerprint.from_dict(snapshot.get("run_fingerprint"))
        if recorded is None:
            return CONTEXT_UNKNOWN
        return compute_run_fingerprint().compare(recorded)
    except Exception:
        return "UNKNOWN"


def _render_comparison_context(comparison: Mapping[str, Any]) -> str:
    """Per-axis tooling comparison: code, prompts, bindings, thresholds.

    A single ``UNKNOWN`` token would tell the model nothing, and a run *can*
    know that (say) the code and the model bindings moved even while the prompt
    set is unresolvable. Each axis is reported on its own evidence.
    """
    try:
        from src.run_fingerprint import (
            CONTEXT_CHANGED,
            CONTEXT_SAME,
            CONTEXT_UNKNOWN,
            RunFingerprint,
            compute_run_fingerprint,
        )

        recorded = RunFingerprint.from_dict(comparison.get("run_fingerprint"))
        if recorded is None:
            return "UNKNOWN (this analysis predates tooling provenance)"
        current = compute_run_fingerprint()
        if recorded.code_dirty or current.code_dirty:
            return "UNKNOWN (uncommitted changes on one side)"

        now, then = current.to_dict(), recorded.to_dict()
        labels = {
            "code_commit": "code",
            "prompt_set_digest": "prompts",
            "binding_digest": "models",
            "thesis_digest": "thresholds",
        }
        parts = []
        for axis, label in labels.items():
            if now[axis] is None or then[axis] is None:
                verdict = CONTEXT_UNKNOWN
            elif now[axis] == then[axis]:
                verdict = CONTEXT_SAME
            else:
                verdict = CONTEXT_CHANGED
            parts.append(f"{label} {verdict}")
        return " | ".join(parts)
    except Exception:
        return "UNKNOWN"


def _pct(value: Any) -> str:
    """Render a percentage, or say plainly that it is unknown."""
    if value is None:
        return "unknown"
    try:
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return "unknown"


def _render_attribution(comparison: Mapping[str, Any]) -> str:
    """Both views, with the additive one labelled as such.

    A legacy comparison carrying no attribution renders honestly rather than
    printing `None` — the model would read that as a value.
    """
    attribution = comparison.get("attribution")
    if not isinstance(attribution, Mapping):
        return "unavailable for this analysis"

    benchmark = comparison.get("benchmark_used") or "benchmark"
    lines = [
        f"Local relative — Market ({benchmark}): "
        f"{_pct(attribution.get('market_return_pct'))} | Residual: "
        f"{_pct(attribution.get('residual_return_pct'))}",
        "  (these two sum to the stock's local return; the residual is what the",
        "   country index does not explain and still contains sector-wide",
        "   rotation — it is not stock-specific alpha)",
        f"USD investor — local {_pct(comparison.get('price_return_pct'))}"
        f" x FX {_pct(attribution.get('fx_return_pct'))}"
        f" = {_pct(attribution.get('usd_investor_return_pct'))}"
        f"   [{comparison.get('fx_observation') or FX_UNAVAILABLE}]",
        # A comparison carrying no scope has not been attributed. Defaulting to
        # CONTEXTUAL asserted the opposite *into the prompt*, which is where the
        # anti-diagnosis rule is read.
        f"Dominant driver: {attribution.get('dominant_driver') or DRIVER_UNKNOWN}"
        f"   Lesson scope: {comparison.get('lesson_scope') or SCOPE_UNRESOLVED}",
    ]
    return "\n".join(lines)


def _render_regime(regime: Any) -> str:
    if not isinstance(regime, Mapping) or not regime:
        return "not recorded"
    parts = [
        str(regime.get("risk_appetite") or "UNKNOWN"),
        str(regime.get("shock_type") or "UNKNOWN"),
        str(regime.get("shock_phase") or "UNKNOWN"),
    ]
    return " / ".join(parts)


def _render_regime_now(delta: Any) -> str:
    """The T1 line, including *why* it is unknown when it is.

    A bare "unknown" invites the model to guess; naming the cause (stale cache,
    changed macro prompt) tells it the comparison genuinely cannot be made.
    """
    if not isinstance(delta, Mapping):
        return "not resolved [shifted: unknown]"
    shifted = delta.get("shifted")
    label = {True: "yes", False: "no", None: "unknown"}.get(shifted, "unknown")
    regime_now = delta.get("regime_now")
    rendered = _render_regime(regime_now) if regime_now else "not available"
    reason = str(delta.get("shift_reason") or "").strip()
    suffix = f" — {reason}" if reason else ""
    return f"{rendered} [shifted: {label}]{suffix}"


def _observation_lines(lesson: Mapping[str, Any]) -> list[str]:
    """The measurement a lesson was drawn from, rendered beside its prose.

    Without this the injected text is a bare imperative: the reader cannot see
    the horizon, what the excess was struck against, or whether FX was even
    determined — so a 40-day 12% wobble against an unnamed index reads exactly
    like a 250-day collapse against the country benchmark.

    Emitted only for outcome lessons, and only from fields the record actually
    carries. A `prior_rejection` is a screening fact with no such measurement,
    and legacy records predate these keys; both correctly render nothing rather
    than a row of zeroes, which would be a claim.
    """
    if lesson.get("lesson_type") == "prior_rejection":
        return []

    lines: list[str] = []
    horizon = lesson.get("days_elapsed")
    excess = lesson.get("excess_return_pct")
    if horizon and excess is not None:
        against = lesson.get("benchmark_used") or "benchmark not recorded"
        lines.append(
            f"observed over {int(horizon)}d vs {against}: "
            f"stock {_pct(lesson.get('actual_return_pct'))}, "
            f"index {_pct(lesson.get('benchmark_return_pct'))}, "
            f"excess {_pct(excess)} [FX {lesson.get('fx_observation') or 'unknown'}]"
        )

    # `FAILURE_MODE` is an LLM pick from a *causal* vocabulary (GOVERNANCE_BLEED,
    # REGULATORY_SHIFT, ACCOUNTING_FRAUD…) made from price and macro data alone,
    # and it renders in the header beside the prose where it reads as an
    # established finding. Feeding the model decision-time flag tokens sharpened
    # that risk: it can now see `REGULATORY_*` in the inputs and pick the matching
    # category. So the header is labelled.
    #
    # Unconditional, and that is the honest shape: `thesis_validation_status` is
    # permanently NOT_EVALUATED because no producer of VALIDATED exists, so a
    # per-record field or a condition reading one would encode a constant as data.
    # Make this conditional at the same time a validation producer lands — the
    # SCOPE_VALIDATED / RESERVED_UNOBSERVED_SCOPES machinery is where that arrives.
    mode = lesson.get("failure_mode") or "UNKNOWN"
    if mode == UNRESOLVED_PRICE_ONLY:
        # The model declined to name a mechanism. Repeating "unvalidated
        # price-only classification" after a topic that already says so reads as
        # a double negative rather than a caveat.
        lines.append("the inputs identified no mechanism for this outcome")
    else:
        lines.append(
            f"topic {mode} is an unvalidated price-only classification, "
            f"not an established cause"
        )

    flags = str(lesson.get("red_flags_at_decision") or "").strip()
    if flags:
        lines.append(
            f"hazards recorded at that decision (not evidence they occurred or "
            f"caused this outcome): {flags}"
        )

    # Historical context only: never an eligibility gate and never a scoring term.
    # A rumoured deal or an unresolved currency changes how a benchmark-relative
    # number should be read, and neither is recoverable from the prose.
    context = [
        f"region {lesson['macro_region']}" if lesson.get("macro_region") else "",
        f"currency source {lesson['currency_source']}"
        if lesson.get("currency_source")
        else "",
        f"special situation {lesson['m_and_a_status']}"
        if str(lesson.get("m_and_a_status") or "").upper() not in ("", "NONE")
        else "",
    ]
    context = [part for part in context if part]
    if context:
        lines.append("decision context: " + " | ".join(context))
    return lines


def _render_recorded_flags(flags: Any) -> str:
    """Hazards the analysis recorded, as bare tokens.

    Deliberately unranked and uninterpreted. Their job is to bound what the model
    may reach for: the prompt forbids naming a mechanism absent from the inputs,
    and regulatory or currency exposure recorded at decision time is exactly the
    class a country benchmark cannot net out of a residual. Whether any of them
    came true is unknowable here, which the surrounding line states.

    Legacy snapshots predate the field and say so rather than rendering nothing —
    an empty line reads to a model as "no flags", which is a different claim.
    """
    if flags is None:
        return "  not recorded (snapshot predates this field)"
    if not isinstance(flags, list) or not flags:
        return "  none recorded"
    return "\n".join(
        f"  - {str(flag).strip()}" for flag in flags[:12] if str(flag).strip()
    )


def _render_kill_criteria(criteria: Any) -> str:
    if not isinstance(criteria, list) or not criteria:
        return "none recorded"
    return "\n".join(
        f"{index}. {str(item).strip()}"
        for index, item in enumerate(criteria[:3], start=1)
        if str(item).strip()
    )


async def generate_lesson(
    comparison: dict[str, Any],
) -> tuple[str, str, str] | None:
    """
    Generate a generalizable lesson from a significant prediction delta.

    One Gemini Flash call with a compact prompt (~400 input tokens).
    Returns (lesson_text, lesson_type, failure_mode) or None on failure.

    The deterministic layer owns the attribution and hands the model the answer;
    the model's job narrows to writing prose about what is left unexplained. The
    prompt used to withhold the regime it was asked to reason about, which is how
    a benchmark crash became a lesson to relax valuation discipline.
    """
    prompt = f"""Given this past equity analysis and its actual outcome, generate ONE generalizable lesson.

ANALYSIS ({comparison.get("analysis_date", "unknown")}):
Ticker: {comparison.get("ticker")} | Sector: {comparison.get("sector", "Unknown")} | Exchange: {comparison.get("exchange", "Unknown")} | Currency: {comparison.get("currency", "USD")}
Verdict: {comparison.get("verdict")} (Position: {comparison.get("position_size", "N/A")}%) | Zone: {comparison.get("zone", "N/A")}
Health: {comparison.get("health_adj", "N/A")} | Growth: {comparison.get("growth_adj", "N/A")} | P/E: {comparison.get("pe_ratio", "N/A")} | PEG: {comparison.get("peg_ratio", "N/A")}
Valuation references: Fair entry {comparison.get("entry_price") or "N/A"} | Base {comparison.get("target_1_price") or "N/A"} | Stretch {comparison.get("target_2_price") or "N/A"} | Downside review {comparison.get("stop_price") or "N/A"} | Horizon: {comparison.get("investment_horizon") or "N/A"}
Key bear risks: {comparison.get("bear_risks_excerpt", "N/A")[:BEAR_EXCERPT_PROMPT_CHARS]}
Flags recorded at decision (noted, NOT tested against the outcome):
{_render_recorded_flags(comparison.get("red_flags_at_decision"))}

OUTCOME ({comparison.get("days_elapsed", 0)} days later):
Price: {comparison.get("start_price", "N/A")} → {comparison.get("end_price", "N/A")} ({comparison.get("price_return_pct", 0):+.1f}%)

RETURN ATTRIBUTION (computed, not inferred — do not re-derive):
{_render_attribution(comparison)}

REGIME AT DECISION (T0): {_render_regime(comparison.get("regime_at_decision"))} (confidence {comparison.get("regime_confidence") or "unknown"})
CACHED REGIME NOW (T1):  {_render_regime_now(comparison.get("cached_regime_delta"))}
TOOLING BETWEEN RUNS: {_render_comparison_context(comparison)}

PRE-COMMITTED THESIS-BREAK TRIGGERS (from the bear case):
{_render_kill_criteria(comparison.get("kill_criteria"))}
You have price and macro data only — NO post-decision company evidence. Do NOT
state whether these fired. Status is {THESIS_NOT_EVALUATED}.

Rules:
- Lesson must be GENERAL (applicable to similar stocks), not specific to this ticker
- One sentence, max 40 words
- If the dominant driver is not RESIDUAL, or it is UNKNOWN, or the regime shifted
  between T0 and T1, write about REGIME CONDITIONING — how this setup behaves
  under this regime — NOT about the company's fundamentals or valuation
  discipline. A market-wide move is not evidence the screen was wrong.
- If the lesson scope is {SCOPE_UNRESOLVED}, the residual is unexplained, not
  diagnosed. Write what should be CHECKED next time, not what was wrong.
- Do NOT introduce a failure mechanism that appears nowhere above. You may name
  a check only if it follows from something actually shown to you. Inventing a
  plausible cause ("verify cash flow reconciliations", "check for supply chain
  pressure") reads as a diagnosis you have no evidence for.
- If nothing above supports a specific check — a thin or missing bear excerpt, no
  thesis-break triggers recorded, an unknown regime — then SAY SO. Naming what
  could not be determined, and what evidence would settle it next time, is a
  complete and correct lesson. It is always better than a plausible invention,
  and "the inputs identify no mechanism" is the honest answer far more often than
  it looks. When you do this, FAILURE_MODE must be {UNRESOLVED_PRICE_ONLY} —
  every other value names a mechanism, so picking one would re-assert in that
  field exactly the cause you just declined to claim.
- Do NOT demote the screen by contrast. No "rather than", "instead of" or "not
  merely" clause that subordinates fundamental valuation, growth, or the screen
  itself to regime or momentum. Write what to ADD, never what to trust less. An
  unattributed move is not evidence the screen was wrong.
- TYPE labels the prediction against the price only. It is not a verdict on the
  thesis, whose status is {THESIS_NOT_EVALUATED} and stays there.
- A flag recorded at the decision may NOMINATE a check. It may not establish
  FAILURE_MODE as fact: picking REGULATORY_SHIFT because a REGULATORY_ flag was
  recorded, or GOVERNANCE_BLEED because a governance flag was, asserts a cause
  from a price move plus a pre-existing label. FAILURE_MODE is a topic for
  filing, not a finding.

LESSON: [your lesson]
TYPE: missed_risk | false_positive | missed_opportunity | correct_call
FAILURE_MODE: {UNRESOLVED_PRICE_ONLY} | CYCLICAL_PEAK | FX_DRIVEN | GOVERNANCE_BLEED | OPERATIONAL_MISS | REGULATORY_SHIFT | MACRO_REGIME | DISRUPTION | VALUATION_TRAP | ACCOUNTING_FRAUD | GEOPOLITICAL | LIQUIDITY_CRISIS | DEAD_MONEY"""

    # Bound before the try so the failure path below can report the seat's real
    # vendor. Construction itself can fail (a missing credential), and "unknown"
    # is the honest answer there rather than a guess.
    lesson_provider: ProviderName = "unknown"
    lesson_model: str | None = None
    try:
        from src.llm_runtime.construction import build_required_model_for_seat
        from src.llm_runtime.seats import SeatId
        from src.observability import build_langchain_config

        llm = build_required_model_for_seat(SeatId.RETROSPECTIVE)
        from langchain_core.messages import HumanMessage

        invoke_config = build_langchain_config(
            metadata={
                "workflow": "retrospective_lesson",
                "ticker": comparison.get("ticker"),
            }
        )
        from src.config import config as settings_config
        from src.service_tiers import floor_llm_hard_timeout

        # The flex floor is provider-scoped, so it must follow the seat's actual
        # binding rather than assume Google: SeatId.RETROSPECTIVE is bindable,
        # and hardcoding a vendor here gave a non-Google operational plane the
        # wrong ceiling (and a misleading provider in every diagnostic below).
        lesson_provider = get_runtime_provider(llm)
        lesson_model = get_model_name(llm)
        hard_timeout = floor_llm_hard_timeout(
            float(get_runtime_config(settings_config).llm_call_hard_timeout_seconds),
            provider=lesson_provider,
            label="retrospective_lesson_timeout",
        )
        if invoke_config:
            response = await run_with_hard_timeout(
                llm.ainvoke(
                    [HumanMessage(content=prompt)], config=cast(Any, invoke_config)
                ),
                timeout=hard_timeout,
                label=f"llm:retrospective_lesson:{comparison.get('ticker', '?')}",
            )
        else:
            response = await run_with_hard_timeout(
                llm.ainvoke([HumanMessage(content=prompt)]),
                timeout=hard_timeout,
                label=f"llm:retrospective_lesson:{comparison.get('ticker', '?')}",
            )

        from src.agents import extract_string_content

        content = extract_string_content(response.content).strip()

        # Parse response
        lesson_match = re.search(r"LESSON:\s*(.+?)(?:\n|$)", content)
        type_match = re.search(r"TYPE:\s*(\S+)", content)
        mode_match = re.search(r"FAILURE_MODE:\s*(\S+)", content)

        lesson_text = lesson_match.group(1).strip() if lesson_match else content[:200]
        lesson_type = (
            type_match.group(1).strip().lower() if type_match else "missed_risk"
        )
        # An absent or out-of-vocabulary FAILURE_MODE resolves to the one value
        # that claims nothing. It used to resolve to OPERATIONAL_MISS, which
        # asserts an operational cause on the strength of a formatting failure.
        failure_mode = (
            mode_match.group(1).strip().upper() if mode_match else UNRESOLVED_PRICE_ONLY
        )

        # Validate against known enums
        if lesson_type not in LESSON_TYPES:
            lesson_type = "missed_risk"
        if failure_mode not in FAILURE_MODES:
            failure_mode = UNRESOLVED_PRICE_ONLY

        logger.info(
            "lesson_generated",
            ticker=comparison.get("ticker"),
            lesson_type=lesson_type,
            failure_mode=failure_mode,
        )

        return lesson_text, lesson_type, failure_mode

    except Exception as e:
        details = classify_failure(
            e,
            provider=lesson_provider,
            # The seat's resolved model, not the legacy settings field:
            # provider and model must describe the same binding or the
            # telemetry is internally inconsistent.
            model_name=lesson_model,
            class_name="RetrospectiveLessonLLM",
        )
        logger.error(
            "lesson_generation_failed",
            provider=details.provider,
            failure_kind=details.kind,
            host=details.host,
            error_type=details.error_type,
            root_cause_type=details.root_cause_type,
            retryable=details.retryable,
            error_message=details.message,
            exc_info=True,
        )
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Component 4b: Rejection Record Storage (non-BUY verdicts)
# ══════════════════════════════════════════════════════════════════════════════


async def save_rejection_record(
    snapshot: dict[str, Any],
    lessons_memory: Any,
) -> bool:
    """
    Save a non-BUY screening verdict as a rejection record in the global
    lessons_learned ChromaDB collection.

    Called from main.py after each HOLD/DO_NOT_INITIATE/SELL verdict.
    Zero LLM cost — pure metadata extraction and ChromaDB write.

    Dedup / upsert logic:
    - Exact match (ticker + analysis_date + lesson_type=prior_rejection): skip
    - Existing quick-mode + new full-mode: delete old, insert new (upgrade)
    - Existing full-mode + new quick-mode: skip (don't downgrade)
    - Existing full-mode + new full-mode: delete old, insert new (fresher data)

    Args:
        snapshot: Prediction snapshot dict from extract_snapshot()
        lessons_memory: FinancialSituationMemory for lessons_learned collection

    Returns:
        True if stored, False if skipped/failed
    """
    if not lessons_memory or not lessons_memory.available:
        logger.debug("rejection_record_memory_unavailable")
        return False

    ticker = snapshot.get("ticker", "UNKNOWN")
    analysis_date = snapshot.get("analysis_date", "")
    verdict = snapshot.get("verdict", "")
    is_quick_mode = bool(snapshot.get("is_quick_mode", False))
    is_strict_mode = bool(snapshot.get("is_strict_mode", False))

    if not verdict or verdict == "BUY":
        return False

    # 1. Check for exact match (same ticker + analysis_date → idempotent re-run)
    try:
        exact = lessons_memory.situation_collection.get(
            where={
                "$and": [
                    {"ticker": {"$eq": ticker}},
                    {"analysis_date": {"$eq": analysis_date}},
                    {"lesson_type": {"$eq": "prior_rejection"}},
                ]
            }
        )
        if exact and exact.get("ids") and len(exact["ids"]) > 0:
            logger.debug(
                "rejection_record_already_exists", ticker=ticker, date=analysis_date
            )
            return False
    except Exception as e:
        logger.debug("rejection_dedup_check_failed", error=str(e))

    # 2. Check for any existing rejection record for this ticker (any date)
    try:
        any_existing = lessons_memory.situation_collection.get(
            where={
                "$and": [
                    {"ticker": {"$eq": ticker}},
                    {"lesson_type": {"$eq": "prior_rejection"}},
                ]
            }
        )
        if any_existing and any_existing.get("ids") and len(any_existing["ids"]) > 0:
            existing_ids = any_existing["ids"]
            existing_meta = (any_existing.get("metadatas") or [{}])[0]
            existing_is_full = not existing_meta.get("is_quick_mode", True)

            # Don't downgrade: existing full-mode + new quick-mode → skip
            if existing_is_full and is_quick_mode:
                logger.debug(
                    "rejection_record_skip_downgrade",
                    ticker=ticker,
                    existing_date=existing_meta.get("analysis_date"),
                    new_date=analysis_date,
                )
                return False

            # Delete existing *prior_rejection* record(s) before inserting fresher
            # data. This intentionally compacts that one ticker's screening state;
            # it must never be generalized into cleanup of outcome lessons or the
            # durable lessons_learned corpus. Historical outcome records are
            # forward-only evidence and must remain intact.
            try:
                lessons_memory.situation_collection.delete(ids=existing_ids)
                logger.debug(
                    "rejection_record_deleted_for_upsert",
                    ticker=ticker,
                    count=len(existing_ids),
                )
            except Exception as e:
                logger.debug("rejection_record_delete_failed", error=str(e))
    except Exception as e:
        logger.debug("rejection_existing_check_failed", error=str(e))

    # 3. Build document text (factual only — no agent reasoning)
    sector = snapshot.get("sector") or "Unknown"
    exchange = snapshot.get("exchange") or "US"
    currency = snapshot.get("currency") or FALLBACK_CURRENCY
    health_adj = snapshot.get("health_adj", "N/A")
    growth_adj = snapshot.get("growth_adj", "N/A")
    risk_tally = snapshot.get("risk_tally", "N/A")
    zone = snapshot.get("zone") or "N/A"
    bear_risks = (snapshot.get("bear_risks_excerpt") or "")[:300]

    mode_note = (
        "Quick mode + strict gates"
        if is_quick_mode and is_strict_mode
        else "Quick mode"
        if is_quick_mode
        else "Strict gates"
        if is_strict_mode
        else "Standard mode"
    )
    document = (
        f"PRIOR SCREENING RECORD: {ticker} ({sector} / {exchange}) — "
        f"{verdict} on {analysis_date}. "
        f"Health {health_adj}/100, Growth {growth_adj}/100, risk tally {risk_tally}. "
        f"Risk zone: {zone}. Mode: {mode_note}."
    )
    if bear_risks:
        document += f"\nBear risks excerpt: {bear_risks}"

    # 4. Build metadata (extends existing schema)
    # Strict-mode rejections are softer signal: strict gates auto-reject some
    # valid candidates (REIT/ETF, earnings quality) at the screening layer, so a non-BUY
    # in strict mode is partly an artifact of the gates rather than a pure
    # quality signal. Multiplicatively discount the existing quick-mode
    # weight by 0.7 for strict rejections.
    confidence_weight = 0.3 if is_quick_mode else 0.5
    if is_strict_mode:
        confidence_weight *= STRICT_MODE_CONFIDENCE_FACTOR
    deep_model = (
        snapshot.get("deep_model")
        or get_runtime_config(config).deep_think_llm
        or "unknown"
    )

    metadata = {
        "ticker": ticker,
        "sector": sector,
        "exchange": exchange,
        "currency": currency,
        "verdict": verdict,
        "lesson_type": "prior_rejection",
        "failure_mode": "N/A",
        "actual_return_pct": 0.0,
        "benchmark_return_pct": 0.0,
        "excess_return_pct": 0.0,
        "days_elapsed": 0,
        "confidence_weight": confidence_weight,
        "analysis_date": analysis_date,
        "retrospective_date": analysis_date,
        "timestamp": datetime.now().isoformat(),
        "is_quick_mode": is_quick_mode,
        "is_strict_mode": is_strict_mode,
        "analysis_model": deep_model,
    }

    stored = await lessons_memory.add_situations([document], [metadata])
    if stored:
        logger.info(
            "rejection_record_stored",
            ticker=ticker,
            verdict=verdict,
            is_quick_mode=is_quick_mode,
            confidence_weight=confidence_weight,
        )
    else:
        logger.warning("rejection_record_storage_failed", ticker=ticker)
    return bool(stored)


# ══════════════════════════════════════════════════════════════════════════════
# Component 5: Lesson Storage
# ══════════════════════════════════════════════════════════════════════════════


async def store_lesson(
    lesson: str,
    lesson_type: str,
    failure_mode: str,
    comparison: dict[str, Any],
    confidence: float,
    lessons_memory: Any,
) -> bool:
    """
    Store a lesson in the global lessons_learned ChromaDB collection.

    Deduplicates by checking for existing lesson with matching
    (ticker, analysis_date) metadata.

    Args:
        lesson: Lesson text (what gets embedded)
        lesson_type: directional type (missed_risk, etc.)
        failure_mode: structural type (CYCLICAL_PEAK, etc.)
        comparison: Full comparison dict
        confidence: Computed confidence weight
        lessons_memory: FinancialSituationMemory instance for lessons_learned

    Returns:
        True if stored, False if skipped/failed
    """
    if not lessons_memory or not lessons_memory.available:
        logger.debug("lessons_memory_unavailable")
        return False

    ticker = comparison.get("ticker", "UNKNOWN")
    analysis_date = comparison.get("analysis_date", "")
    analysis_id = comparison.get("analysis_id")
    # The canonical per-run identity, carried from the candidate. Falls back to a
    # real analysis_id for comparisons built outside the orchestrator.
    snapshot_id = comparison.get(SNAPSHOT_IDENTITY_KEY) or analysis_id

    # Deduplication shares one predicate with the orchestrator's early skip, and
    # now shares its *key* too. These were two independent copies of the same
    # (ticker, analysis_date) query, so a prior_rejection record blocked the write
    # here even once the orchestrator let the snapshot through — and once that was
    # fixed the two still disagreed, because only one of them had an identity.
    if _lesson_already_processed(
        lessons_memory,
        ticker,
        analysis_date,
        str(snapshot_id) if snapshot_id else None,
    ):
        logger.info(
            "lesson_already_exists",
            ticker=ticker,
            date=analysis_date,
            snapshot_id=snapshot_id,
        )
        return False

    # Decided once, here, from the comparison itself — never passed in by a
    # caller who could disagree with what is about to be written.
    eligibility, eligibility_reason = lesson_eligibility(comparison)

    metadata = {
        "ticker": ticker,
        "sector": comparison.get("sector", "Unknown") or "Unknown",
        "exchange": comparison.get("exchange", "US") or "US",
        "currency": comparison.get("currency", "USD") or "USD",
        "verdict": comparison.get("verdict", "UNKNOWN") or "UNKNOWN",
        "actual_return_pct": float(comparison.get("price_return_pct", 0.0)),
        "benchmark_return_pct": float(comparison.get("benchmark_return_pct", 0.0)),
        "excess_return_pct": float(comparison.get("excess_return_pct", 0.0)),
        "fx_delta_pct": float(comparison.get("fx_delta_pct") or 0.0),
        # Disambiguates the line above: a 0.0 delta means "no FX leg" only when
        # this reads FX_NOT_APPLICABLE, and means nothing at all when it reads
        # FX_UNAVAILABLE.
        "fx_observation": str(comparison.get("fx_observation") or FX_UNAVAILABLE),
        "days_elapsed": int(comparison.get("days_elapsed", 0)),
        "lesson_type": lesson_type,
        "failure_mode": failure_mode,
        # What the excess return was measured against. The prompt has always
        # named it; the stored record did not, so a reader could not tell whether
        # a lesson was struck against ^N225, ^KS11 or nothing in particular —
        # and "residual" means nothing without its benchmark.
        "benchmark_used": str(comparison.get("benchmark_used") or ""),
        # Context a benchmark cannot net out. All flat scalars — ChromaDB
        # metadata takes no lists — so the flag tokens are joined, and only
        # tokens: the `detail`/`rationale` prose is deliberately not carried,
        # since a hazard is a recorded fact while its explanation is not one.
        "red_flags_at_decision": ",".join(
            str(flag)
            for flag in (comparison.get("red_flags_at_decision") or [])
            if str(flag).strip()
        ),
        "macro_region": str(comparison.get("macro_region") or ""),
        "m_and_a_status": str(comparison.get("m_and_a_status") or ""),
        # How the currency was determined. `unresolved` / `fallback_bare_ticker`
        # mean the benchmark and FX legs rest on a guess about the listing, which
        # a later reader cannot recover from the currency code alone.
        "currency_source": str(comparison.get("currency_source") or ""),
        "analysis_model": comparison.get("deep_model", "unknown") or "unknown",
        "analysis_date": analysis_date,
        # Per-run identity, so dedup can tell two same-day analyses apart.
        # Empty string rather than None: ChromaDB metadata rejects null values.
        # `analysis_id` keeps its established meaning (the run/trace id minted in
        # persistence.save_results_to_file) and is empty when the snapshot had
        # none; SNAPSHOT_IDENTITY_KEY carries the surrogate for legacy snapshots.
        "analysis_id": str(analysis_id) if analysis_id else "",
        SNAPSHOT_IDENTITY_KEY: str(snapshot_id) if snapshot_id else "",
        "retrospective_date": datetime.now().strftime("%Y-%m-%d"),
        "confidence_weight": float(confidence),
        "timestamp": datetime.now().isoformat(),
    }
    regime = comparison.get("regime_at_decision") or {}
    if isinstance(regime, dict):
        metadata.update(
            {
                "regime_risk_appetite": regime.get("risk_appetite", ""),
                "regime_shock_type": regime.get("shock_type", ""),
                "regime_shock_phase": regime.get("shock_phase", ""),
                "regime_dip_posture": regime.get("dip_posture", ""),
                "regime_equity_transmission": regime.get("equity_transmission", ""),
                "regime_confidence": comparison.get("regime_confidence", "") or "",
            }
        )

    # Everything a reader needs to reproduce `confidence_weight` and to decide
    # whether this lesson applies to the situation in front of them. All flat
    # scalars: ChromaDB metadata takes no nested values and no nulls.
    attribution = comparison.get("attribution")
    attribution = attribution if isinstance(attribution, dict) else {}
    delta = comparison.get("cached_regime_delta")
    delta = delta if isinstance(delta, dict) else {}
    shifted = delta.get("shifted")
    metadata.update(
        {
            # Same fail-closed default as the prompt renderer: an absent scope is
            # an unattributed outcome, not a regime observation.
            "lesson_scope": str(comparison.get("lesson_scope") or SCOPE_UNRESOLVED),
            "lesson_eligibility": eligibility,
            "lesson_eligibility_reason": eligibility_reason,
            "dominant_driver": str(
                attribution.get("dominant_driver") or DRIVER_UNKNOWN
            ),
            "market_return_pct": float(attribution.get("market_return_pct") or 0.0),
            "residual_return_pct": float(attribution.get("residual_return_pct") or 0.0),
            "benchmark_available": bool(attribution.get("benchmark_available", False)),
            # Tri-state flattened to a token, not a bool: `False` would claim the
            # regime demonstrably held still, which an unresolved delta does not.
            "regime_shifted": (
                "YES" if shifted is True else "NO" if shifted is False else "UNKNOWN"
            ),
            "comparison_context": str(
                comparison.get("comparison_context") or "UNKNOWN"
            ),
            "decision_intent": str(comparison.get("decision_intent") or ""),
            "is_quick_mode": bool(comparison.get("is_quick_mode", False)),
            "is_strict_mode": bool(comparison.get("is_strict_mode", False)),
            "thesis_validation_status": str(
                comparison.get("thesis_validation_status") or THESIS_NOT_EVALUATED
            ),
        }
    )

    stored = await lessons_memory.add_situations([lesson], [metadata])
    if stored:
        logger.info(
            "lesson_stored",
            ticker=ticker,
            lesson_type=lesson_type,
            failure_mode=failure_mode,
            confidence=confidence,
            excess_return=comparison.get("excess_return_pct"),
        )
    else:
        logger.warning(
            "lesson_storage_failed",
            ticker=ticker,
            lesson_type=lesson_type,
        )
    return bool(stored)


# ══════════════════════════════════════════════════════════════════════════════
# Component 6: Lesson Retrieval & Injection
# ══════════════════════════════════════════════════════════════════════════════


def _regime_matches(
    current_regime: Mapping[str, Any] | None, meta: Mapping[str, Any]
) -> bool:
    """Does a regime-conditional lesson apply to the regime in front of us?

    Matches on risk appetite **or** shock type — either alone is enough for the
    lesson to be about a recognizably similar world.

    Returns ``False`` when the current regime is unknown. That is deliberate: a
    lesson stamped CONTEXTUAL asserts something about a *particular* regime, and
    with no regime to compare against there is no basis to apply it. Treating
    unknown as a match would make the scope stamp decorative.
    """
    if not isinstance(current_regime, Mapping) or not current_regime:
        return False
    for field_name in REGIME_COMPARED_FIELDS:
        current = str(current_regime.get(field_name) or "").strip().upper()
        stored = str(meta.get(f"regime_{field_name}") or "").strip().upper()
        if current and stored and current == stored:
            return True
    return False


def _same_ticker_rejections(lessons_memory: Any, ticker: str) -> list[dict[str, Any]]:
    """Fetch this ticker's prior screening record by metadata, not similarity.

    "Have I screened this ticker out before?" is an exact-match fact. Leaving it
    to embedding similarity means the single most decision-relevant record in the
    store competes for a slot on semantic distance and can simply lose.
    """
    if not ticker:
        return []
    try:
        found = lessons_memory.situation_collection.get(
            where={
                "$and": [
                    {"ticker": {"$eq": ticker}},
                    {"lesson_type": {"$eq": "prior_rejection"}},
                ]
            }
        )
    except Exception:
        return []
    if not found:
        return []
    documents = found.get("documents") or []
    metadatas = found.get("metadatas") or []
    return [
        {"document": document, "metadata": dict(metadata), "distance": 0.0}
        for document, metadata in zip(documents, metadatas, strict=False)
        if isinstance(metadata, Mapping)
    ]


async def get_relevant_lessons(
    lessons_memory: Any,
    sector: str,
    ticker: str,
    n_results: int = LESSON_QUERY_CANDIDATES,
) -> list[dict[str, Any]]:
    """
    Query lessons_learned collection for relevant past lessons.

    Two candidate sources, merged: an exact metadata fetch of this ticker's prior
    screening record, then a vector query. The former is deterministic by design —
    see :func:`_same_ticker_rejections`.

    Args:
        lessons_memory: FinancialSituationMemory for lessons_learned collection
        sector: Sector of current analysis (for query relevance)
        ticker: Current ticker (for exchange/currency matching)
        n_results: Max results to fetch from ChromaDB

    Returns:
        List of lesson dicts with 'document', 'metadata', 'distance' keys
    """
    if not lessons_memory or not lessons_memory.available:
        _record_capture_memory_event(
            {
                "event": "lessons_query_skipped",
                "sector": sector,
                "ticker": ticker,
                "n_results": n_results,
                "available": False,
            }
        )
        return []

    try:
        # The ticker was accepted by this function and never used. Including it
        # matters: the store is cross-sector, and a sector-only query cannot
        # distinguish a lesson about this listing from one about its neighbours.
        query = f"Investment lessons for {ticker} and similar {sector} sector stocks"
        results = await lessons_memory.query_similar_situations(
            query_text=query,
            n_results=n_results,
        )
        results = list(results or [])

        # Merge the deterministic fetch ahead of the vector candidates, skipping
        # any it already returned.
        exact = _same_ticker_rejections(lessons_memory, ticker)
        if exact:
            seen_documents = {r.get("document") for r in results}
            prepend = [r for r in exact if r["document"] not in seen_documents]
            results = prepend + results
        _record_capture_memory_event(
            {
                "event": "lessons_query",
                "sector": sector,
                "ticker": ticker,
                "n_results": n_results,
                "query_text": query,
                "available": True,
                "results": results,
            }
        )
        return cast(list[dict[str, Any]], results)
    except Exception as e:
        logger.debug("lesson_query_failed", error=str(e))
        _record_capture_memory_event(
            {
                "event": "lessons_query_failed",
                "sector": sector,
                "ticker": ticker,
                "n_results": n_results,
                "available": True,
                "error": str(e),
            }
        )
        return []


async def format_lessons_for_injection(
    lessons_memory: Any,
    ticker: str,
    sector: str,
    current_regime: Mapping[str, Any] | None = None,
) -> str:
    """
    Query global lessons collection, rank by confidence + geographic/regime boost,
    return formatted text for injection into researcher prompts.

    Called from agents.py researcher_node (2-line integration).

    Args:
        lessons_memory: FinancialSituationMemory for lessons_learned collection
        ticker: Current ticker being analyzed
        sector: Sector of current ticker
        current_regime: Optional parsed MACRO_REGIME_BLOCK for relevance boosting

    Returns:
        Formatted string for prompt injection, or "" if no lessons available
    """
    if not lessons_memory or not lessons_memory.available:
        _record_capture_memory_event(
            {
                "event": "lessons_injection_skipped",
                "ticker": ticker,
                "sector": sector,
                "available": False,
                "reason": "memory_unavailable",
            }
        )
        return ""

    # Fast-path: no lessons exist yet — skip embedding API call (~1-2ms check vs ~200ms)
    try:
        if lessons_memory.situation_collection.count() == 0:
            _record_capture_memory_event(
                {
                    "event": "lessons_injection_skipped",
                    "ticker": ticker,
                    "sector": sector,
                    "available": True,
                    "reason": "collection_empty",
                }
            )
            return ""
    except Exception:
        pass  # Fall through to normal query

    try:
        results = await get_relevant_lessons(lessons_memory, sector, ticker)
    except Exception:
        _record_capture_memory_event(
            {
                "event": "lessons_injection_skipped",
                "ticker": ticker,
                "sector": sector,
                "available": True,
                "reason": "query_exception",
            }
        )
        return ""

    if not results:
        _record_capture_memory_event(
            {
                "event": "lessons_injection_skipped",
                "ticker": ticker,
                "sector": sector,
                "available": True,
                "reason": "no_results",
            }
        )
        return ""

    # Apply geographic boost and confidence filtering
    suffix = get_ticker_suffix(ticker)
    current_exchange = suffix.lstrip(".") if suffix else "US"
    current_currency = EXCHANGE_CURRENCY.get(suffix, FALLBACK_CURRENCY)

    scored_lessons = []
    skipped_off_regime = 0
    skipped_unresolved = 0
    skipped_ineligible = 0
    for r in results:
        meta = r.get("metadata", {})
        base_confidence = meta.get("confidence_weight", 0.5)

        # Outcome lessons must claim eligibility positively. A record written
        # before this policy existed carries no marker and is withheld rather
        # than assumed safe: the store cannot vouch for itself. This is what
        # quarantines the pre-policy corpus without deleting or rewriting it.
        if meta.get("lesson_type") in OUTCOME_LESSON_TYPES and (
            meta.get("lesson_eligibility") != LESSON_ELIGIBILITY_INJECTABLE
        ):
            skipped_ineligible += 1
            continue

        # An UNRESOLVED lesson has no established cause. It is retained in the
        # store for review and for later promotion, but never handed to a live
        # analysis as guidance — see UNRESOLVED_LESSON_MARKER. Redundant for
        # records written under the eligibility policy above, and retained as
        # defense in depth for anything that reaches here without a lesson_type.
        if meta.get("lesson_scope") == SCOPE_UNRESOLVED:
            skipped_unresolved += 1
            continue

        # A CONTEXTUAL lesson was learned under a particular regime and says
        # nothing about a different one. Skipped before scoring, so it cannot be
        # promoted by a geographic boost into a world it does not describe.
        # Legacy records carry no scope and are unaffected.
        if meta.get("lesson_scope") == SCOPE_CONTEXTUAL and not _regime_matches(
            current_regime, meta
        ):
            skipped_off_regime += 1
            continue

        # Boost priority: same-ticker rejection record > geographic proximity
        boost = 0.0
        if (
            meta.get("lesson_type") == "prior_rejection"
            and meta.get("ticker") == ticker
        ):
            boost += 0.35  # Same ticker was previously screened out — very relevant
        else:
            # Geographic boost for regular lessons
            if meta.get("exchange") == current_exchange:
                boost += 0.15
            if meta.get("currency") == current_currency:
                boost += 0.10

            if current_regime and current_regime.get("confidence") in {
                "HIGH",
                "MEDIUM",
            }:
                appetite = current_regime.get("risk_appetite")
                if appetite and meta.get("regime_risk_appetite") == appetite:
                    boost += 0.06
                shock_type = current_regime.get("shock_type")
                if shock_type and meta.get("regime_shock_type") == shock_type:
                    boost += 0.06
                dip_posture = current_regime.get("dip_posture")
                if dip_posture and meta.get("regime_dip_posture") == dip_posture:
                    boost += 0.03

        effective_score = base_confidence + boost

        # Filter low-confidence lessons
        if effective_score < 0.4:
            continue

        scored_lessons.append(
            {
                "lesson": r["document"],
                "failure_mode": meta.get("failure_mode", "UNKNOWN"),
                "sector": meta.get("sector", "Unknown"),
                "exchange": meta.get("exchange", "??"),
                "confidence": round(effective_score, 2),
                "lesson_type": meta.get("lesson_type"),
                "lesson_scope": meta.get("lesson_scope"),
                "ticker": meta.get("ticker"),
                "benchmark_used": meta.get("benchmark_used", ""),
                "days_elapsed": meta.get("days_elapsed"),
                "actual_return_pct": meta.get("actual_return_pct"),
                "benchmark_return_pct": meta.get("benchmark_return_pct"),
                "excess_return_pct": meta.get("excess_return_pct"),
                "fx_observation": meta.get("fx_observation", ""),
                "red_flags_at_decision": meta.get("red_flags_at_decision", ""),
                "macro_region": meta.get("macro_region", ""),
                "m_and_a_status": meta.get("m_and_a_status", ""),
                "currency_source": meta.get("currency_source", ""),
            }
        )

    # Sort by effective score descending, take top 3. UNRESOLVED records never
    # reach this point, so no scope tiebreak is needed.
    scored_lessons.sort(key=lambda x: x["confidence"], reverse=True)
    top_lessons = scored_lessons[:3]

    filtered_count = len(results) - len(scored_lessons) if results else 0
    logger.debug(
        "lesson_retrieval_stats",
        ticker=ticker,
        sector=sector,
        candidates=len(results) if results else 0,
        passed_filter=len(scored_lessons),
        filtered_out=filtered_count,
        skipped_off_regime=skipped_off_regime,
        skipped_unresolved=skipped_unresolved,
        skipped_ineligible=skipped_ineligible,
        top_n=len(top_lessons),
    )

    if not top_lessons:
        _record_capture_memory_event(
            {
                "event": "lessons_injection_skipped",
                "ticker": ticker,
                "sector": sector,
                "available": True,
                "reason": "filtered_out",
                "candidates": len(results) if results else 0,
                "passed_filter": len(scored_lessons),
            }
        )
        return ""

    lines = ["LESSONS FROM PAST ANALYSES (cross-market):"]
    for lesson in top_lessons:
        # A `prior_rejection` record is a screening artifact, not a learned market
        # lesson — label it distinctly so it is not read as a generalizable lesson.
        if lesson.get("lesson_type") == "prior_rejection":
            prefix = f"PRIOR REJECTION ({lesson.get('ticker') or '?'})"
        else:
            prefix = "LESSON"
        lines.append(
            f"- {prefix}: {lesson['lesson']} "
            f"({lesson['failure_mode']} | {lesson['sector']}/{lesson['exchange']} "
            f"| conf: {lesson['confidence']})"
        )
        for detail in _observation_lines(lesson):
            lines.append(f"    {detail}")

    formatted = "\n".join(lines)

    # Inspect lessons text for injection before returning.
    formatted = await get_current_inspection_service().check(
        InspectionEnvelope(
            content_text=formatted,
            raw_content=formatted,
            source_kind=SourceKind.memory_retrieval,
            source_name="lessons_learned",
            metadata={"ticker": ticker, "sector": sector},
        )
    )

    _record_capture_memory_event(
        {
            "event": "lessons_injected",
            "ticker": ticker,
            "sector": sector,
            "available": True,
            "candidates": len(results) if results else 0,
            "passed_filter": len(scored_lessons),
            "selected_lessons": top_lessons,
            "injected_text": formatted,
        }
    )
    return str(formatted)


# ══════════════════════════════════════════════════════════════════════════════
# Component 7: Early Dedup Helper
# ══════════════════════════════════════════════════════════════════════════════


def _lesson_already_processed(
    lessons_memory: Any,
    ticker: str,
    analysis_date: str,
    snapshot_id: str | None = None,
) -> bool:
    """Check whether an *outcome* lesson already exists for this snapshot.

    Three corrections over the original, all load-bearing:

    1. The query is scoped to ``OUTCOME_LESSON_TYPES``. A ``prior_rejection``
       screening record shares the ticker and analysis date with the analysis it
       describes, so counting it here suppressed the outcome lesson entirely.
    2. Matching prefers the canonical per-run identity so two same-day runs stay
       distinct, and falls back to ``analysis_date`` for records stored before
       that field existed. The comparison is done in Python rather than in the
       ``where`` clause because the cases need different keys, and the candidate
       set is tiny by construction (``MAX_LESSONS_PER_TICKER`` caps it per ticker).
    3. Both call sites pass the *same* value. They did not: the orchestrator
       passed the candidate's ``snapshot_identity`` while ``store_lesson`` passed
       ``comparison["analysis_id"]``, which nothing set — so the pre-check
       compared identities and the write fell through to the date. Four lessons
       per run were generated and then rejected, indefinitely, because a snapshot
       is memoized only once its lesson is durably stored.

    Stored records carry the identity under ``snapshot_identity``; ``analysis_id``
    is read as a fallback so records written before that key existed still dedup.

    Uses ChromaDB metadata query — no embedding needed. ~50ms.
    """
    if not lessons_memory or not lessons_memory.available:
        return False
    try:
        existing = lessons_memory.situation_collection.get(
            where={
                "$and": [
                    {"ticker": {"$eq": ticker}},
                    {"lesson_type": {"$in": sorted(OUTCOME_LESSON_TYPES)}},
                ]
            }
        )
    except Exception:
        return False

    if not existing:
        return False
    for meta in existing.get("metadatas") or []:
        if not isinstance(meta, Mapping):
            continue
        stored_id = meta.get(SNAPSHOT_IDENTITY_KEY) or meta.get("analysis_id")
        if stored_id and snapshot_id:
            if str(stored_id) == str(snapshot_id):
                return True
            continue
        # Legacy record (no analysis_id), or a snapshot that carries none:
        # the date is the only identity available on one side or the other.
        if analysis_date and meta.get("analysis_date") == analysis_date:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Component 8: Orchestrator
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class _RunCounters:
    """Mutable accumulator for one retrospective run.

    Every scanned snapshot lands in exactly one of the disposition buckets, which
    is what makes the totals auditable rather than merely suggestive. Frozen into
    a :class:`RetrospectiveRunSummary` once the run finishes.
    """

    scanned: int = 0
    skipped_existing_lesson: int = 0
    skipped_memo: int = 0
    skipped_too_recent: int = 0
    deferred_over_budget: int = 0
    evaluated: int = 0
    unassessed_benchmark: int = 0
    triggered: int = 0
    generated: int = 0
    stored: int = 0
    failed: int = 0

    def freeze(self, *, dry_run: bool) -> RetrospectiveRunSummary:
        return RetrospectiveRunSummary(
            scanned=self.scanned,
            skipped_existing_lesson=self.skipped_existing_lesson,
            skipped_memo=self.skipped_memo,
            skipped_too_recent=self.skipped_too_recent,
            deferred_over_budget=self.deferred_over_budget,
            evaluated=self.evaluated,
            unassessed_benchmark=self.unassessed_benchmark,
            triggered=self.triggered,
            generated=self.generated,
            stored=self.stored,
            failed=self.failed,
            dry_run=dry_run,
        )


@dataclass(frozen=True, slots=True)
class RetrospectiveRunSummary:
    """What one retrospective run actually did.

    The orchestrator previously reported a start line and a stored count, which
    cannot answer the only operational question that matters before widening its
    inputs: how many network round-trips will this cost, and where did the rest of
    the corpus go? Every scanned snapshot lands in exactly one disposition bucket
    (see :attr:`reconciles`), so the totals are auditable rather than indicative.
    """

    scanned: int = 0
    skipped_existing_lesson: int = 0
    skipped_memo: int = 0
    skipped_too_recent: int = 0
    deferred_over_budget: int = 0
    evaluated: int = 0
    unassessed_benchmark: int = 0
    triggered: int = 0
    generated: int = 0
    stored: int = 0
    failed: int = 0
    # A dry run reports what *would* be evaluated, so a reader must be able to
    # tell the projection from the record of work performed.
    dry_run: bool = False

    @property
    def reconciles(self) -> bool:
        """Every scanned snapshot must be accounted for exactly once."""
        return self.scanned == (
            self.skipped_existing_lesson
            + self.skipped_memo
            + self.skipped_too_recent
            + self.deferred_over_budget
            + self.evaluated
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "scanned": self.scanned,
            "skipped_existing_lesson": self.skipped_existing_lesson,
            "skipped_memo": self.skipped_memo,
            "skipped_too_recent": self.skipped_too_recent,
            "deferred_over_budget": self.deferred_over_budget,
            "evaluated": self.evaluated,
            "unassessed_benchmark": self.unassessed_benchmark,
            "triggered": self.triggered,
            "generated": self.generated,
            "stored": self.stored,
            "failed": self.failed,
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True, slots=True)
class _EvaluationCandidate:
    """A snapshot that survived dedup and is eligible to be priced."""

    ticker: str
    identity: str
    days_elapsed: int
    snapshot: dict[str, Any]

    @property
    def analysis_date(self) -> str:
        return str(self.snapshot.get("analysis_date") or "")

    @property
    def band_distance(self) -> int:
        """Distance from the centre of the 1.0-weight confidence band."""
        return abs(self.days_elapsed - _EVALUATION_BAND_CENTRE_DAYS)


def _select_within_budget(
    candidates: list[_EvaluationCandidate], budget: int
) -> tuple[list[_EvaluationCandidate], list[_EvaluationCandidate]]:
    """Split candidates into (selected, deferred), deterministically.

    Ordering is by proximity to the highest-confidence temporal band, then by
    ticker/date/identity so that two runs over an unchanged corpus select exactly
    the same set — a budget that reshuffles would starve the same snapshots
    forever while re-pricing others.
    """
    ordered = sorted(
        candidates,
        key=lambda c: (c.band_distance, c.ticker, c.analysis_date, c.identity),
    )
    if budget is None or budget <= 0:
        return ordered, []
    return ordered[:budget], ordered[budget:]


def _memoize_stored(
    memo: EvaluationMemo,
    pending: Mapping[str, _EvaluationCandidate],
    comparison: Mapping[str, Any],
    *,
    outcome: str = MEMO_OUTCOME_TRIGGERED,
) -> None:
    """Record a triggered snapshot only once its lesson is durably stored.

    "Processed" must mean the work finished. Recording at pricing time meant a
    timed-out lesson call, a refused content inspection or a failed Chroma write
    cost the snapshot its lesson for the whole re-evaluation interval, with a
    manual memo deletion as the only recovery.
    """
    identity = comparison.get(MEMO_IDENTITY_KEY)
    candidate = pending.get(str(identity)) if identity else None
    if candidate is None:
        return
    memo.record(
        candidate.identity,
        ticker=candidate.ticker,
        analysis_date=candidate.analysis_date,
        days_elapsed=candidate.days_elapsed,
        outcome=outcome,
    )


async def run_retrospective(
    ticker: str | None,
    results_dir: Path,
    lessons_memory: Any = None,
    progress: Callable[[SnapshotLoadProgress], None] | None = None,
    *,
    archive_dirs: Sequence[Path] = (),
    memo: EvaluationMemo | None = None,
    memo_path: Path | None = None,
    max_evaluations: int | None = None,
    dry_run: bool = False,
    on_summary: Callable[[RetrospectiveRunSummary], None] | None = None,
) -> list[dict[str, Any]]:
    """
    Orchestrate retrospective: load snapshots → compare → generate → store.

    Evaluation runs in phases so that the expensive middle one can be bounded:
    build the candidate set (cheap, local), select within budget
    (deterministic), then price and generate. The alternative — pricing inline
    per ticker — cannot express a run-wide ceiling.

    Args:
        ticker: If provided, process only this ticker. If None, all tickers.
        results_dir: Directory containing analysis JSONs.
        lessons_memory: FinancialSituationMemory for lessons_learned.
                       If None, creates one.
        archive_dirs: Read-only archived-results directories, scanned after
            ``results_dir`` (see :func:`load_past_snapshots`).
        memo: Evaluation memo; constructed from ``memo_path`` when omitted.
        max_evaluations: Ceiling on snapshots priced this run. ``None`` reads
            ``RETROSPECTIVE_MAX_EVALUATIONS_PER_RUN`` from config, so an operator
            can probe a policy change on a small batch without a source edit.
            Resolved at call time rather than as a signature default, which would
            bind the value once at import.
        dry_run: Report what *would* be evaluated without pricing anything —
            no market fetch, no LLM call, no memo write, no lesson stored.
        on_summary: Receives the run summary. The lesson list stays the return
            value so existing callers are unaffected.

    Returns:
        List of generated lesson dicts (for display/logging)
    """

    def _emit(counters: _RunCounters) -> None:
        """Every exit path reports, including the ones that do no work."""
        summary = counters.freeze(dry_run=dry_run)
        if not summary.reconciles:
            # A disposition bucket was missed. Loud rather than silent: the
            # totals are the only evidence that the budget and memo are sane.
            logger.warning(
                "retrospective_summary_does_not_reconcile", **summary.to_dict()
            )
        if on_summary is not None:
            try:
                on_summary(summary)
            except Exception as exc:
                logger.debug(
                    "retrospective_summary_callback_failed",
                    **summarize_exception(exc, operation="retrospective summary"),
                )

    if max_evaluations is None:
        max_evaluations = int(
            getattr(
                config,
                "retrospective_max_evaluations_per_run",
                RETROSPECTIVE_MAX_EVALUATIONS_PER_RUN,
            )
        )

    # Create lessons memory if not provided
    if lessons_memory is None:
        try:
            from src.memory import FinancialSituationMemory

            lessons_memory = FinancialSituationMemory(LESSONS_COLLECTION_NAME)
        except Exception as e:
            logger.error(
                "lessons_memory_init_failed",
                exc_info=True,
                **summarize_exception(e, operation="lessons_memory_init"),
            )
            _emit(_RunCounters())
            return []

    # Load snapshots
    all_snapshots = load_past_snapshots(
        ticker, results_dir, archive_dirs=archive_dirs, progress=progress
    )

    if not all_snapshots:
        msg = f"for {ticker}" if ticker else "in results directory"
        logger.info("retrospective_no_snapshots", ticker=ticker or "all", detail=msg)
        _emit(_RunCounters())
        return []

    total_snapshots = sum(len(s) for s in all_snapshots.values())
    logger.info(
        "retrospective_starting",
        tickers=len(all_snapshots),
        total_snapshots=total_snapshots,
        filter_ticker=ticker or "all",
    )

    if memo is None:
        memo = EvaluationMemo(memo_path)
    counters = _RunCounters()

    # ── Phase 1: candidate selection (no network, no LLM) ────────────────────
    candidates: list[_EvaluationCandidate] = []
    for snap_ticker, snapshots in all_snapshots.items():
        for snapshot in snapshots:
            counters.scanned += 1
            snap_ticker_value = str(snapshot.get("ticker") or snap_ticker)
            snap_date = str(snapshot.get("analysis_date") or "")
            identity = snapshot_identity(snapshot)

            if _lesson_already_processed(
                lessons_memory, snap_ticker_value, snap_date, identity
            ):
                counters.skipped_existing_lesson += 1
                logger.debug(
                    "snapshot_already_processed",
                    ticker=snap_ticker_value,
                    date=snap_date,
                    identity=identity,
                )
                continue

            days_elapsed = _snapshot_days_elapsed(snapshot)
            if days_elapsed is None or days_elapsed < MINIMUM_DAYS_ELAPSED:
                # Cannot clear the trigger yet; spending budget here would starve
                # snapshots that can.
                counters.skipped_too_recent += 1
                continue

            if not memo.should_evaluate(identity, days_elapsed):
                counters.skipped_memo += 1
                continue

            candidates.append(
                _EvaluationCandidate(
                    ticker=snap_ticker_value,
                    identity=identity,
                    days_elapsed=days_elapsed,
                    snapshot=snapshot,
                )
            )

    selected, deferred = _select_within_budget(candidates, max_evaluations)
    counters.deferred_over_budget = len(deferred)
    if deferred:
        logger.info(
            "retrospective_budget_reached",
            budget=max_evaluations,
            deferred=len(deferred),
            msg="Remaining snapshots will be evaluated on a later run",
        )

    if dry_run:
        # `evaluated` reports the *projected* cost — the whole point of the mode
        # is to answer "how many round-trips would this run buy?" before paying
        # for them. The dry_run flag on the summary keeps that legible.
        counters.evaluated = len(selected)
        logger.info(
            "retrospective_dry_run_complete",
            **counters.freeze(dry_run=True).to_dict(),
        )
        _emit(counters)
        return []

    # ── Phase 2: price the selected snapshots ────────────────────────────────
    comparisons_by_ticker: dict[str, list[dict[str, Any]]] = {}
    for candidate in selected:
        counters.evaluated += 1
        try:
            comparison = await compare_to_reality(candidate.snapshot)
        except Exception as exc:
            counters.failed += 1
            logger.warning(
                "retrospective_comparison_failed",
                ticker=candidate.ticker,
                identity=candidate.identity,
                **summarize_exception(exc, operation="retrospective comparison"),
            )
            continue

        # Checked first, and deliberately: an unassessed outcome is a truthy dict,
        # so any branch that tests `if comparison` would otherwise generate a
        # lesson from a return whose market component is unknown.
        if comparison and comparison.get(UNASSESSED_REASON_KEY):
            counters.unassessed_benchmark += 1
            # NO_DATA re-evaluates on the next run rather than in 30 days — a
            # failed index fetch is a transient outage, not a verdict.
            outcome = MEMO_OUTCOME_NO_DATA
        elif comparison:
            counters.triggered += 1
            comparison["_confidence"] = compute_confidence(comparison)
            # Carry the identity so phase 3 can memoize this snapshot only once
            # its lesson is durably stored. Recording TRIGGERED here would mean a
            # timed-out LLM call or a refused Chroma write silently costs the
            # snapshot its lesson for the whole re-evaluation interval.
            comparison[MEMO_IDENTITY_KEY] = candidate.identity
            # The same value the pre-check above used, so the write-time dedup
            # cannot reach a different conclusion from the early skip.
            comparison[SNAPSHOT_IDENTITY_KEY] = candidate.identity
            comparisons_by_ticker.setdefault(candidate.ticker, []).append(comparison)
            outcome = None
        else:
            outcome = MEMO_OUTCOME_BELOW_THRESHOLD

        if outcome is not None:
            memo.record(
                candidate.identity,
                ticker=candidate.ticker,
                analysis_date=candidate.analysis_date,
                days_elapsed=candidate.days_elapsed,
                outcome=outcome,
            )

    # Flushed before generation so a run interrupted during phase 3 keeps the
    # pricing it already paid for; triggered snapshots are added below.
    memo.flush()
    pending_memo = {c.identity: c for c in selected}

    # ── Phase 3: generate and store, capped per ticker ───────────────────────
    generated_lessons = []
    for snap_ticker, comparisons in comparisons_by_ticker.items():
        logger.info(
            "retrospective_processing_ticker",
            ticker=snap_ticker,
            snapshot_count=len(comparisons),
        )
        ticker_lessons = 0
        # Sort by significance (largest excess return first)
        comparisons.sort(key=lambda c: abs(c.get("excess_return_pct", 0)), reverse=True)

        for index, comparison in enumerate(comparisons):
            if ticker_lessons >= MAX_LESSONS_PER_TICKER:
                logger.info(
                    "max_lessons_reached",
                    ticker=snap_ticker,
                    max=MAX_LESSONS_PER_TICKER,
                    withheld=len(comparisons) - index,
                )
                # Withheld by policy, not lost to failure — memoize so the run
                # does not re-price them every time only to cap them again.
                for capped in comparisons[index:]:
                    _memoize_stored(
                        memo, pending_memo, capped, outcome=MEMO_OUTCOME_CAPPED
                    )
                break

            confidence = comparison["_confidence"]
            result = await generate_lesson(comparison)
            if not result:
                continue

            lesson_text, lesson_type, failure_mode = result

            stored = await store_lesson(
                lesson_text,
                lesson_type,
                failure_mode,
                comparison,
                confidence,
                lessons_memory,
            )

            lesson_record = {
                "ticker": snap_ticker,
                "lesson": lesson_text,
                "lesson_type": lesson_type,
                "failure_mode": failure_mode,
                "excess_return_pct": comparison.get("excess_return_pct"),
                "confidence": confidence,
                "stored": stored,
            }
            generated_lessons.append(lesson_record)
            counters.generated += 1

            if stored:
                ticker_lessons += 1
                counters.stored += 1
                _memoize_stored(memo, pending_memo, comparison)

            trace_id = comparison.get("trace_id")
            if trace_id:
                from src.observability import create_deferred_score

                create_deferred_score(
                    trace_id=trace_id,
                    name="excess_return_6m",
                    value=float(comparison.get("excess_return_pct") or 0.0),
                    data_type="NUMERIC",
                    comment=f"benchmark={comparison.get('benchmark_used', 'UNKNOWN')}",
                    metadata={"ticker": snap_ticker},
                )
                create_deferred_score(
                    trace_id=trace_id,
                    name="prediction_correct",
                    value=1.0
                    if _prediction_is_directionally_correct(comparison)
                    else 0.0,
                    data_type="BOOLEAN",
                    metadata={"ticker": snap_ticker},
                )

    # Second flush: triggered snapshots are memoized only after their lesson is
    # durably stored, so this is what makes a *successful* generation stick.
    memo.flush()

    logger.info(
        "retrospective_complete",
        lessons_generated=len(generated_lessons),
        tickers_evaluated=len(all_snapshots),
        **counters.freeze(dry_run=False).to_dict(),
    )
    _emit(counters)
    return generated_lessons


def create_lessons_memory() -> Any:
    """
    Create a FinancialSituationMemory instance for the global lessons_learned
    collection. This is a factory function for use in memory.py or main.py.

    Returns:
        FinancialSituationMemory instance (may have available=False if ChromaDB
        is not configured)
    """
    global _LESSONS_MEMORY_INSTANCE

    if _LESSONS_MEMORY_INSTANCE is None:
        from src.memory import FinancialSituationMemory

        _LESSONS_MEMORY_INSTANCE = FinancialSituationMemory(LESSONS_COLLECTION_NAME)
    return _LESSONS_MEMORY_INSTANCE


def _reset_lessons_memory_cache_for_tests() -> None:
    """Reset the cached lessons memory instance to keep tests isolated."""
    global _LESSONS_MEMORY_INSTANCE
    _LESSONS_MEMORY_INSTANCE = None
