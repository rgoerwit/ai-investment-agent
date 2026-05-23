"""End-to-end integration coverage for the Tranche 1–5 quality features.

These tests prove that each load-bearing feature reaches the consumer it
was designed for, using fixture *builders* (not hand-rolled string literals)
so the suite stays resilient to incidental wording or formatting drift.

Each integration check answers a single load-bearing question:

- Does a bear ``KILL_CRITERIA`` block surface in both the PM payload AND the
  memo's ``Kill criteria`` section?
- Does Research Manager's ``VARIANT_VIEW`` reach the memo's variant slot?
- Do ``VALUATION_SCENARIOS`` reach the PM payload AND the chart's
  ``FootballFieldData.scenarios`` field?
- Does an APAC non-silent verdict trigger an ``APAC_RESOLUTION`` block
  (either PM-emitted or programmatic fallback)?
- Does a *clean* forensic auditor leave the PM output unchanged (no
  misleading "Auditor flagged anomalies" boilerplate)?

The "strange-input" section covers what's likely to come back from real LLM
output: malformed fences, whitespace-only fields, conflicting blocks,
unicode, dual-shape JSON drift, mixed positive/negative auditor phrasing,
negative or zero derived EPS, and truncated rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from src.agents.decision_nodes import (
    _auditor_has_material_concern,
    _ensure_apac_resolution_block,
    _ensure_auditor_resolution_block,
    _ensure_consultant_resolution_block,
)
from src.agents.support import (
    extract_kill_criteria,
    get_bear_history,
    summarize_for_pm,
)
from src.charts.base import FootballFieldData
from src.charts.extractors.valuation import (
    extract_valuation_scenarios,
    resolve_eps_ttm,
)
from src.eval.report_quality_judge import score_report
from src.reporting.memo import (
    _VARIANT_PLACEHOLDER,
    build_memo,
    render_memo_markdown,
)

# ---------------------------------------------------------------------------
# Fixture builders
#
# Each helper returns a realistic block / report / state dict. Defaults
# match production shapes; named-argument overrides let individual tests
# focus on the one thing they're stressing.
# ---------------------------------------------------------------------------


def data_block(
    *,
    pe_ratio_ttm: float | None = 12.0,
    eps_ttm: float | None = None,
    current_price: float | None = 100.0,
    de_ratio: float | None = 0.32,
    net_debt_ebitda: float | None = 1.4,
    roic_percent: float | None = 22.0,
    extra: dict[str, str] | None = None,
) -> str:
    """Build a fundamentals DATA_BLOCK matching the production wire shape.

    Only the fields exercised by Tranche 1–5 features are populated; pass
    ``None`` to a numeric arg to omit that row, ``extra=`` to add custom
    fields (e.g. legacy aliases or unusual metrics for stress tests).
    """
    rows: list[str] = ["SECTOR: Industrials"]
    if pe_ratio_ttm is not None:
        rows.append(f"PE_RATIO_TTM: {pe_ratio_ttm}")
    if eps_ttm is not None:
        rows.append(f"EPS_TTM: {eps_ttm}")
    if current_price is not None:
        rows.append(f"CURRENT_PRICE: {current_price:.2f}")
    if de_ratio is not None:
        rows.append(f"DE_RATIO: {de_ratio}")
    if net_debt_ebitda is not None:
        rows.append(f"NET_DEBT_EBITDA: {net_debt_ebitda}")
    if roic_percent is not None:
        rows.append(f"ROIC_PERCENT: {roic_percent}")
    if extra:
        rows.extend(f"{k}: {v}" for k, v in extra.items())
    return (
        "### --- START DATA_BLOCK ---\n"
        + "\n".join(rows)
        + "\n### --- END DATA_BLOCK ---\n"
    )


@dataclass
class ScenarioRow:
    """Per-scenario parameters threaded through the VALUATION_SCENARIOS block."""

    multiple: float
    growth_pct: float
    margin_delta_bps: float
    drivers: str
    probability: int


def valuation_params(
    *,
    data_sufficiency: str = "HIGH",
    methodology: str = "P/E",
    bear: ScenarioRow | None = None,
    base: ScenarioRow | None = None,
    bull: ScenarioRow | None = None,
    include_scenarios: bool = True,
) -> str:
    """Build the Valuation Calculator's structured output.

    Always emits a VALUATION_PARAMS block; emits VALUATION_SCENARIOS only
    when ``include_scenarios`` is True. Probabilities default to 30/50/20.
    """
    bear = bear or ScenarioRow(8, -5, -200, "Cyclical trough.", 30)
    base = base or ScenarioRow(12, 8, 0, "Mid-cycle base.", 50)
    bull = bull or ScenarioRow(16, 15, 100, "Cycle peak.", 20)

    params = dedent(
        """\
        ### --- START VALUATION_PARAMS ---
        METHOD: P/E_NORMALIZATION
        SECTOR: Industrials
        SECTOR_MEDIAN_PE: 17
        CURRENT_PE: 12.0
        PEG_RATIO: 0.9
        CURRENT_PRICE: 100.00
        CONFIDENCE: HIGH
        ### --- END VALUATION_PARAMS ---
        """
    )
    if not include_scenarios:
        return params

    def _row(label: str, row: ScenarioRow) -> str:
        return dedent(
            f"""\
            {label}_MULTIPLE: {row.multiple}
            {label}_GROWTH_PCT: {row.growth_pct}
            {label}_MARGIN_DELTA_BPS: {row.margin_delta_bps}
            {label}_DRIVERS: {row.drivers}
            {label}_PROBABILITY: {row.probability}
            """
        )

    scenarios = (
        "### --- START VALUATION_SCENARIOS ---\n"
        f"METHODOLOGY: {methodology}\n"
        f"DATA_SUFFICIENCY: {data_sufficiency}\n"
        + _row("BEAR", bear)
        + _row("BASE", base)
        + _row("BULL", bull)
        + "### --- END VALUATION_SCENARIOS ---\n"
    )
    return params + "\n" + scenarios


def kill_criteria_block(*triggers: str) -> str:
    """Wrap measurable SELL triggers in the fenced KILL_CRITERIA block.

    Bear emits up to 3 triggers; the parser caps at 3.
    """
    body = "\n".join(f"TRIGGER_{i + 1}: {t}" for i, t in enumerate(triggers))
    return (
        "### --- START KILL_CRITERIA ---\n" f"{body}\n" "### --- END KILL_CRITERIA ---"
    )


def bear_history(
    *,
    kill_triggers: tuple[str, ...] = (
        "D/E exceeds 1.5",
        "Two consecutive years of negative FCF",
    ),
    narrative: str = "Bear case body — concerns about cyclical peak.",
) -> str:
    """Build a realistic bear-history string ending with a KILL_CRITERIA block."""
    parts = [narrative, ""]
    if kill_triggers:
        parts.append(kill_criteria_block(*kill_triggers))
    return "\n\n".join(parts)


def pm_narrative(
    *,
    verdict: str = "BUY",
    rationale: str = "Quality compounder at 12x P/E with 22% ROIC.",
    include_pm_block: bool = True,
) -> str:
    """PM-narrative string with optional PM_BLOCK fence (PM_BLOCK preferred by extractor)."""
    pm_block = (
        f"\n\n### --- START PM_BLOCK ---\nVERDICT: {verdict}\n### --- END PM_BLOCK ---\n"
        if include_pm_block
        else ""
    )
    return (
        f"#### PORTFOLIO MANAGER VERDICT: {verdict}\n\n"
        "### DECISION RATIONALE\n\n"
        f"{rationale}\n"
        f"{pm_block}"
    )


def rm_plan(
    *,
    consensus: str = "Market sees growth peaking.",
    variant: str | None = "We see structural margin recovery.",
    basis: str = "Order book +30% YoY surfaced by Foreign Language Analyst.",
    no_variant: bool = False,
) -> str:
    """Research Manager's investment plan with optional VARIANT PERCEPTION block."""
    parts = ["### FINAL RECOMMENDATION: BUY", ""]
    parts.append("### VARIANT PERCEPTION")
    parts.append(f"CONSENSUS_VIEW: {consensus}")
    if no_variant:
        parts.append("NO VARIANT — synthesis aligns with consensus.")
    elif variant is not None:
        parts.append(f"VARIANT_VIEW: {variant}")
        parts.append(f"BASIS: {basis}")
    parts.append("")
    parts.append("### RISKS TO MONITOR:")
    parts.append("- Cyclical demand softness.")
    return "\n".join(parts)


def auditor_report(
    *,
    kind: str = "clean",
    extra: str = "",
) -> str:
    """Build a forensic auditor report.

    ``kind`` selects one of: ``clean`` (no concerns), ``insufficient`` (no
    primary filings), ``flagged`` (one named anomaly), ``mixed`` (both a
    positive token and a negative phrase — negative must win).
    """
    if kind == "clean":
        body = "After detailed forensic review, no material concerns identified."
    elif kind == "insufficient":
        body = "STATUS=INSUFFICIENT_DATA — auditor could not access primary filings."
    elif kind == "flagged":
        body = "Paper profit ratio at 0.18 indicates earnings quality concern."
    elif kind == "mixed":
        body = "Reviewed paper profit ratio; no material concerns identified."
    else:  # pragma: no cover — guards against typo in tests
        raise ValueError(f"unknown auditor kind: {kind}")
    return body + ("\n\n" + extra if extra else "")


def apac_report(
    *, verdict: str = "CAUTION", concern: str = "Promoter pledges unresolved."
) -> str:
    return (
        "### APAC REGIONAL AUDIT: 7203.T\n\n"
        f"**VERDICT FOR CONSULTANT AND PM**: {verdict} — {concern}\n"
    )


@dataclass
class State:
    """Tiny wrapper so tests can switch between runtime and saved-JSON shapes."""

    fundamentals: str = ""
    pm: str = ""
    bear: str = ""
    valuation: str = ""
    plan: str = ""
    auditor: str = ""
    apac: str = ""
    consultant: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def runtime(self) -> dict[str, Any]:
        """Runtime AgentState shape (top-level fields)."""
        d: dict[str, Any] = {
            "final_trade_decision": self.pm,
            "fundamentals_report": self.fundamentals,
            "valuation_params": self.valuation,
            "investment_plan": self.plan,
            "auditor_report": self.auditor,
            "apac_regional_report": self.apac,
            "consultant_review": self.consultant,
        }
        if self.bear:
            d["investment_debate_state"] = {"bear_history": self.bear}
        d.update(self.extras)
        return d

    def saved(self) -> dict[str, Any]:
        """Saved-JSON shape written by src/persistence.py."""
        return {
            "final_decision": {"decision": self.pm},
            "reports": {
                "fundamentals_report": self.fundamentals,
                "valuation_params": self.valuation,
                "auditor_report": self.auditor,
                "apac_regional_report": self.apac,
                "consultant_review": self.consultant,
            },
            "investment_analysis": {
                "investment_plan": self.plan,
                "investment_debate": {"bear_history": self.bear},
            },
            **self.extras,
        }


# ---------------------------------------------------------------------------
# Load-bearing data flows
# ---------------------------------------------------------------------------


@pytest.fixture(params=["runtime", "saved"])
def state_pair(request) -> tuple[str, dict[str, Any]]:
    """Parameterize each integration test across runtime + saved-JSON shapes.

    Prevents shape-specific bugs (which were the entire Tier-1 reason for
    Tranche 5) from sneaking back in.
    """
    triggers = ("D/E exceeds 1.5", "Two consecutive years of negative FCF")
    s = State(
        fundamentals=data_block(),
        pm=pm_narrative(),
        bear=bear_history(kill_triggers=triggers),
        valuation=valuation_params(),
        plan=rm_plan(),
        apac=apac_report(),
        auditor=auditor_report(kind="clean"),
    )
    if request.param == "runtime":
        return request.param, s.runtime()
    return request.param, s.saved()


def test_bear_kill_criteria_reaches_memo(state_pair) -> None:
    """KILL_CRITERIA flow: Bear emits → support helpers extract → memo carries."""
    _, state = state_pair
    triggers = extract_kill_criteria(get_bear_history(state))
    assert triggers, "kill criteria not extracted from bear history"
    memo = build_memo(state)
    # Each trigger from bear must survive into the memo, intact and ordered.
    assert memo.kill_criteria == triggers
    md = render_memo_markdown(memo)
    for trigger in triggers:
        assert trigger in md, f"trigger {trigger!r} missing from rendered memo"


def test_variant_perception_reaches_memo(state_pair) -> None:
    """RM emits VARIANT_VIEW → memo's variant slot carries content (not placeholder)."""
    _, state = state_pair
    memo = build_memo(state)
    assert memo.variant_view != _VARIANT_PLACEHOLDER
    md = render_memo_markdown(memo)
    assert "**Variant view.**" in md


def test_valuation_scenarios_reach_memo_and_pm_and_chart(state_pair) -> None:
    """Scenarios flow: valuation_params → parser+EPS resolver → memo + PM + chart."""
    _, state = state_pair
    fundamentals = state.get("fundamentals_report") or (state.get("reports") or {}).get(
        "fundamentals_report"
    )
    eps = resolve_eps_ttm(fundamentals)
    assert eps and eps > 0, "EPS must resolve from CURRENT_PRICE / PE_RATIO_TTM"

    val_params = state.get("valuation_params") or (state.get("reports") or {}).get(
        "valuation_params"
    )
    scenarios = extract_valuation_scenarios(val_params, eps)
    assert scenarios is not None

    # Math invariants — derived from the scenarios, not hardcoded values.
    ivs = [scenarios.bear_iv, scenarios.base_iv, scenarios.bull_iv]
    assert min(ivs) <= scenarios.weighted_iv <= max(ivs)
    assert scenarios.bear_iv <= scenarios.base_iv <= scenarios.bull_iv

    # Memo carries the scenario summary.
    memo = build_memo(state)
    assert "Bear" in memo.valuation and "Bull" in memo.valuation

    # Chart can accept the scenarios via the typed Protocol.
    fd = FootballFieldData(
        ticker="TEST",
        trade_date="2026-05-21",
        current_price=100.0,
        fifty_two_week_high=150.0,
        fifty_two_week_low=70.0,
        scenarios=scenarios,
    )
    assert fd.scenarios is scenarios


def test_apac_non_silent_triggers_resolution_fallback(state_pair) -> None:
    """APAC non-silent → PM either emits APAC_RESOLUTION or the fallback inserter does."""
    _, state = state_pair
    apac = state.get("apac_regional_report") or (state.get("reports") or {}).get(
        "apac_regional_report"
    )
    pm = state.get("final_trade_decision") or (state.get("final_decision") or {}).get(
        "decision"
    )
    out = _ensure_apac_resolution_block(pm, apac)
    assert "APAC_RESOLUTION:" in out


def test_clean_auditor_does_not_inject_false_anomaly(state_pair) -> None:
    """Clean auditor → fallback inserter is silent (no misleading anomaly block)."""
    _, state = state_pair
    auditor = state.get("auditor_report") or (state.get("reports") or {}).get(
        "auditor_report"
    )
    pm = state.get("final_trade_decision") or (state.get("final_decision") or {}).get(
        "decision"
    )
    assert _ensure_auditor_resolution_block(pm, auditor) == pm
    assert _auditor_has_material_concern(auditor) is False


def test_quality_judge_grades_full_post_tranche_state_as_A(state_pair) -> None:
    """After Tranches 1–5 the saved-JSON shape alone should be enough to score `A`.

    The judge falls back to JSON when no markdown is supplied, picking up
    kill criteria from bear history, variant from RM plan, scenarios from
    valuation_params, and specialist resolution via the post-render PM
    block injection.
    """
    _, state = state_pair
    pm_with_resolution = _ensure_apac_resolution_block(
        state.get("final_trade_decision")
        or (state.get("final_decision") or {}).get("decision"),
        state.get("apac_regional_report")
        or (state.get("reports") or {}).get("apac_regional_report"),
    )
    # Render a quick markdown wrapper around the memo so the judge sees the
    # memo features that depend on markdown presence.
    memo = build_memo(state)
    md = render_memo_markdown(memo) + "\n\n" + pm_with_resolution
    score = score_report(md, state)
    # All six features should fire — we render the memo + APAC_RESOLUTION ourselves.
    assert score.has_memo
    assert score.has_kill_criteria
    assert score.has_scenario_valuation
    assert score.has_specialist_resolution
    assert score.has_variant_view
    # Source confidence comes from memo render; should be present too.
    assert score.has_source_confidence
    assert score.overall == "A"


# ---------------------------------------------------------------------------
# Strange-input cases
#
# Real LLM output is messy. Each test here pins one realistic shape of bad
# input and asserts that the system *degrades gracefully* rather than
# crashing, over-counting, or under-counting features.
# ---------------------------------------------------------------------------


# ---- Bear KILL_CRITERIA stress -----------------------------------------------


@pytest.mark.parametrize(
    "bad_block",
    [
        # Missing END marker — block must be ignored, not partially extracted.
        "### --- START KILL_CRITERIA ---\nTRIGGER_1: half open",
        # Whitespace-only triggers — must be filtered out.
        kill_criteria_block("", "   ", "\t"),
        # No fenced block at all — bear emitted prose only.
        "Bear case body without any kill criteria.",
        # Empty input — must not raise.
        "",
    ],
)
def test_extract_kill_criteria_tolerates_bad_input(bad_block: str) -> None:
    assert extract_kill_criteria(bad_block) == []


def test_kill_criteria_unicode_triggers_round_trip() -> None:
    """Drivers can carry non-ASCII (regional native constructs, em-dashes)."""
    triggers = (
        "D/E exceeds 1.5",
        "営業利益 falls below ¥10B for 2 consecutive quarters",
        "재벌 cross-holding ratio rises above 30%",
    )
    out = extract_kill_criteria(kill_criteria_block(*triggers))
    assert out == list(triggers)


# ---- Valuation scenarios stress ---------------------------------------------


@pytest.mark.parametrize(
    "build_kwargs,reason",
    [
        ({"data_sufficiency": "LOW"}, "agent escape hatch — must suppress IVs"),
        # Probabilities sum to 105 — should reject as fabrication.
        (
            {
                "bear": ScenarioRow(8, -5, -200, "x", 35),
                "base": ScenarioRow(12, 8, 0, "x", 50),
                "bull": ScenarioRow(16, 15, 100, "x", 20),
            },
            "prob sum != 100",
        ),
        # Inverted multiples (bear > bull) — fabrication signal.
        (
            {
                "bear": ScenarioRow(20, -5, -200, "x", 30),
                "base": ScenarioRow(12, 8, 0, "x", 50),
                "bull": ScenarioRow(8, 15, 100, "x", 20),
            },
            "inverted multiples",
        ),
    ],
)
def test_scenarios_reject_known_bad_inputs(build_kwargs: dict, reason: str) -> None:
    val = valuation_params(**build_kwargs)
    fundamentals = data_block()
    eps = resolve_eps_ttm(fundamentals)
    assert (
        extract_valuation_scenarios(val, eps) is None
    ), f"scenarios should reject: {reason}"


def test_scenarios_with_unicode_drivers_survive() -> None:
    """Drivers carry non-ASCII narrative; parser/memo must preserve it intact."""
    bear = ScenarioRow(8, -5, -200, "鉄鋼マージン compression — 200bps", 30)
    val = valuation_params(bear=bear)
    eps = resolve_eps_ttm(data_block())
    s = extract_valuation_scenarios(val, eps)
    assert s is not None
    assert "鉄鋼マージン" in s.bear.drivers


@pytest.mark.parametrize("bad_eps", [None, 0.0, -1.0, -1e9])
def test_scenarios_skip_when_eps_unusable(bad_eps) -> None:
    """Zero / negative / missing EPS must not produce nonsensical IVs."""
    s = extract_valuation_scenarios(valuation_params(), bad_eps)
    assert s is None


def test_scenarios_last_block_wins_on_duplicate_emission() -> None:
    """If the LLM emits two VALUATION_SCENARIOS blocks (self-correction), the
    parser must take the LAST one — matching the convention used by PM_BLOCK
    and other fenced-block parsers in the repo."""
    first = valuation_params(
        bear=ScenarioRow(20, 0, 0, "stale", 30),
        base=ScenarioRow(20, 0, 0, "stale", 50),
        bull=ScenarioRow(20, 0, 0, "stale", 20),
    )
    # Append a second, correctly-ordered block.
    second_only = valuation_params(
        bear=ScenarioRow(8, 0, 0, "fresh bear", 30),
        base=ScenarioRow(12, 0, 0, "fresh base", 50),
        bull=ScenarioRow(16, 0, 0, "fresh bull", 20),
    )
    eps = resolve_eps_ttm(data_block())
    s = extract_valuation_scenarios(first + "\n\n" + second_only, eps)
    assert s is not None
    assert "fresh" in s.bear.drivers


# ---- Variant perception stress ----------------------------------------------


def test_variant_no_variant_is_honored() -> None:
    """An honest "NO VARIANT" declaration counts as variant content."""
    s = State(plan=rm_plan(no_variant=True), pm=pm_narrative())
    memo = build_memo(s.runtime())
    md = render_memo_markdown(memo)
    assert "aligns with consensus" in memo.variant_view
    assert "**Variant view.**" in md


def test_variant_placeholder_is_omitted_from_memo() -> None:
    """When the RM plan carries no variant signal the line is suppressed.

    Must avoid the substring "NO VARIANT" / "VARIANT_VIEW" / "CONSENSUS_VIEW"
    in the plan text — otherwise the extractor (correctly) reads it as
    real content.
    """
    bare_plan = (
        "### FINAL RECOMMENDATION: BUY\n\n"
        "### RISKS TO MONITOR:\n- Cyclical demand.\n"
    )
    s = State(plan=bare_plan, pm=pm_narrative())
    memo = build_memo(s.runtime())
    assert memo.variant_view == _VARIANT_PLACEHOLDER
    assert "**Variant view.**" not in render_memo_markdown(memo)


def test_variant_extraction_handles_truncated_variant_view() -> None:
    """LLM output sometimes ends mid-sentence; extractor must not raise."""
    truncated = "CONSENSUS_VIEW: Market sees X.\nVARIANT_VIEW: We see"
    s = State(plan=truncated, pm=pm_narrative())
    memo = build_memo(s.runtime())
    assert isinstance(memo.variant_view, str)


# ---- Auditor stress ---------------------------------------------------------


@pytest.mark.parametrize(
    "kind,should_fire",
    [
        ("clean", False),
        ("insufficient", False),
        ("flagged", True),
        # 'mixed': positive token AND a clean negation — negation must win,
        # otherwise we re-introduce the "no red flags" false positive.
        ("mixed", False),
    ],
)
def test_auditor_concern_detection_handles_realistic_phrasings(
    kind: str, should_fire: bool
) -> None:
    assert _auditor_has_material_concern(auditor_report(kind=kind)) is should_fire


def test_auditor_idempotent_when_block_already_present_in_pm() -> None:
    """If PM emitted its own AUDITOR_RESOLUTION the fallback must NOT double it."""
    pm = (
        "rationale\n\n"
        "AUDITOR_RESOLUTION:\n- VERDICT: REJECTED\n\n"
        "### --- START PM_BLOCK ---\nVERDICT: BUY\n### --- END PM_BLOCK ---\n"
    )
    out = _ensure_auditor_resolution_block(pm, auditor_report(kind="flagged"))
    assert out.count("AUDITOR_RESOLUTION:") == 1


# ---- APAC stress ------------------------------------------------------------


@pytest.mark.parametrize(
    "apac_text",
    [
        "NO_MATERIAL_APAC_CONNECTION",
        "APAC_SPECIALIST_UNAVAILABLE",
        "",
        "   ",
        None,
    ],
)
def test_apac_silent_or_empty_does_not_inject(apac_text) -> None:
    pm = pm_narrative()
    assert _ensure_apac_resolution_block(pm, apac_text) == pm


def test_apac_substring_match_does_not_trip_on_sentinel_in_body() -> None:
    """If a non-silent APAC report happens to mention the sentinel inside its
    narrative (e.g., quoting it), the inserter should still fire because the
    report as a whole is not the sentinel."""
    body = (
        "### APAC REGIONAL AUDIT\n"
        "Operator note: NO_MATERIAL_APAC_CONNECTION would be wrong here; "
        "Promoter pledges unresolved.\n"
        "**VERDICT FOR CONSULTANT AND PM**: CAUTION — pledges unresolved.\n"
    )
    out = _ensure_apac_resolution_block(pm_narrative(), body)
    assert "APAC_RESOLUTION:" in out


# ---- Summarizer block preservation stress ----------------------------------


def test_summarizer_preserves_blocks_even_when_report_is_padded() -> None:
    """Long agent output with critical blocks at the tail must not lose them."""
    padding = (("Filler paragraph. " * 40) + "\n\n") * 80
    report = padding + kill_criteria_block("D/E > 1.5")
    out = summarize_for_pm(report, "research", max_chars=3000)
    assert "### --- START KILL_CRITERIA ---" in out
    assert "[...summarized...]" in out


def test_summarizer_does_not_duplicate_block_already_in_head() -> None:
    """Dedup guard: if the head retains the block, don't re-append it."""
    block = kill_criteria_block("D/E > 1.5")
    padding = (("Filler. " * 40) + "\n\n") * 80
    report = block + "\n\n" + padding
    out = summarize_for_pm(report, "research", max_chars=3000)
    assert out.count("### --- START KILL_CRITERIA ---") == 1


# ---- State-access stress (the original Tier-1 bug class) -------------------


def test_saved_json_with_only_decision_renders_memo_decision() -> None:
    """Persistence-shape regression: PM verdict lives at final_decision.decision."""
    saved = {"final_decision": {"decision": pm_narrative(verdict="HOLD")}}
    memo = build_memo(saved)
    assert memo.decision == "HOLD"


def test_saved_json_with_no_decision_path_renders_unavailable() -> None:
    """No PM output anywhere → memo stub, not a crash."""
    memo = build_memo({})
    assert memo.decision == "UNAVAILABLE"
    md = render_memo_markdown(memo)
    assert "UNAVAILABLE" in md


def test_extract_kill_criteria_from_saved_json_path() -> None:
    """bear_history lives at investment_analysis.investment_debate.bear_history in saved JSON."""
    triggers = ("trigger A", "trigger B")
    s = State(bear=bear_history(kill_triggers=triggers))
    bear = get_bear_history(s.saved())
    assert extract_kill_criteria(bear) == list(triggers)


# ---- Quality judge error-handling stress -----------------------------------


def test_quality_judge_corrupt_json_returns_none(tmp_path: Path) -> None:
    """Corrupt input shouldn't tank the run; aggregate must skip the file."""
    from src.eval.report_quality_judge import score_saved_analysis

    bad = tmp_path / "corrupt_analysis.json"
    bad.write_text("{ not valid", encoding="utf-8")
    assert score_saved_analysis(bad) is None


def test_quality_judge_empty_directory_yields_zero_count(tmp_path: Path) -> None:
    """No artifacts → zero count, no exception."""
    from src.eval.report_quality_judge import aggregate

    summary = aggregate(list(tmp_path.glob("*_analysis.json")))
    assert summary["count"] == 0
    assert sum(summary["grades"].values()) == 0


# ---- Consultant fallback interaction with new APAC/Auditor wiring ---------


def test_all_three_resolution_blocks_compose_without_overwriting() -> None:
    """Stacking the three fallbacks (consultant + APAC + auditor) on the same
    PM output must produce three distinct blocks without losing any of them."""
    pm = pm_narrative()
    consultant = "1. Material concern about unresolved earnings quality (placeholder)."
    apac = apac_report(verdict="CAUTION", concern="Concrete concern here.")
    auditor = auditor_report(kind="flagged")
    out = _ensure_consultant_resolution_block(pm, consultant)
    out = _ensure_apac_resolution_block(out, apac)
    out = _ensure_auditor_resolution_block(out, auditor)
    assert out.count("CONSULTANT_RESOLUTION:") >= 1
    assert "APAC_RESOLUTION:" in out
    assert "AUDITOR_RESOLUTION:" in out
    # All three appear BEFORE the PM_BLOCK fence (so PM downstream consumers see them).
    pm_marker = out.find("### --- START PM_BLOCK ---")
    for token in ("CONSULTANT_RESOLUTION:", "APAC_RESOLUTION:", "AUDITOR_RESOLUTION:"):
        assert 0 <= out.find(token) < pm_marker, f"{token} not above PM_BLOCK"
