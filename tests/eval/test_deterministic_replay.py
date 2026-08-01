"""L2 deterministic replay: frozen LLM outputs → stable pure-consumer results.

The pipeline's LLM nodes are stochastic, but the parsers/validators that consume
their text are pure functions. This pins the structured output of those consumers
against committed real captured artifacts (harvested from a 2330.TW quick run
under ``evals/captures/schema_v4/``), so a prompt/parser change that silently
alters downstream extraction is caught at zero LLM cost.

Two flavours:
- **Golden values** — explicit asserts on the high-value extracted fields.
- **Structural invariants** — properties that must hold regardless of the verdict
  (e.g. extreme D/E ⇒ pre-screening ``REJECT``), which are noise-free.

Regenerate the metrics snapshot after an *intended* parser change with::

    UPDATE_REPLAY_SNAPSHOTS=1 poetry run pytest tests/eval/test_deterministic_replay.py

See ``scratch/general-prompt-checking.md`` (L2) for the design.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.evidence_constraints import downstream_evidence_constraints
from src.analysis_snapshot import (
    build_analysis_snapshot,
    build_pre_senior_snapshot,
    decision_claim_ids,
    reconcile_data_block_projection,
)
from src.article_audit import (
    audit_article_claim_support,
    audit_article_claim_usage,
    extract_source_confidence_context,
)
from src.charts.extractors.pm_block import extract_pm_block
from src.charts.extractors.valuation import _extract_params
from src.claim_policy import MATERIAL_CLAIM_POLICIES, ClaimPolicy
from src.ibkr.order_builder import parse_trade_block
from src.pm_claim_audit import (
    reconcile_final_decision_trace,
    render_decision_trace_instruction,
)
from src.tooling.structured_ingress import build_structured_ingress_record
from src.validators.financial_rules import detect_red_flags
from src.validators.metric_extractor import extract_metrics
from src.validators.sector_classifier import detect_sector
from src.validators.supplemental_extractors import extract_value_trap_score

FROZEN = Path(__file__).resolve().parents[1] / "fixtures" / "frozen"


def _register_metrics(state: dict, payload: dict) -> dict:
    return {
        **state,
        "structured_inputs": {
            "raw_financial_metrics": build_structured_ingress_record(
                payload,
                agent_key="junior_fundamentals_analyst",
                tool_name="get_financial_metrics",
            )
        },
    }


_SNAPSHOT = FROZEN / "2330_TW_metrics_snapshot.json"


@pytest.fixture(autouse=True)
def _replay_no_network(request):
    """Pure parsing only — disable sockets (reuses pytest-socket)."""
    request.getfixturevalue("socket_disabled")


def _read(name: str) -> str:
    return (FROZEN / name).read_text(encoding="utf-8")


# --- golden values --------------------------------------------------------------


def test_data_block_metrics_golden():
    metrics = extract_metrics(_read("2330_TW_data_block.txt"))
    assert metrics["debt_to_equity"] == 20.0
    assert metrics["pe_ratio"] == 27.47
    assert metrics["pb_ratio"] == 8.71
    assert metrics["sector"] == "technology"
    assert metrics["adjusted_health_score"] == 75.0


def test_data_block_red_flags_golden():
    block = _read("2330_TW_data_block.txt")
    flags, pre_screen = detect_red_flags(
        extract_metrics(block), sector=detect_sector(block)
    )
    assert pre_screen == "PASS"
    assert {f["type"] for f in flags} == {
        "LOCAL_COVERAGE_HIGH",
        "OCF_SOURCE_DISCREPANCY",
    }


def test_decision_inputs_path_matches_dict_path():
    """Stage 5 parity gate: the typed DecisionInputs path must produce
    byte-identical red-flag output to the raw metrics-dict path on the frozen
    corpus (both the base fixture and the extreme-leverage mutation)."""
    from src.decision_inputs import DecisionInputs

    for block in (
        _read("2330_TW_data_block.txt"),
        _read("2330_TW_data_block.txt").replace("D/E: 0.20", "D/E: 9.00"),
    ):
        metrics = extract_metrics(block)
        sector = detect_sector(block)
        inputs = DecisionInputs.from_metrics(metrics, sector=sector)
        assert detect_red_flags(inputs, sector=sector) == detect_red_flags(
            metrics, sector=sector
        )


def test_pm_block_golden():
    pm = extract_pm_block(_read("2330_TW_pm_block.txt"))
    assert pm.verdict == "DO_NOT_INITIATE"
    assert pm.zone == "HIGH"
    assert pm.risk_tally == 1.58
    assert (pm.health_adj, pm.growth_adj) == (75, 100)
    assert isinstance(pm.decision_facts, tuple)
    assert isinstance(pm.decision_gates, tuple)


def test_value_trap_golden():
    vt = extract_value_trap_score(_read("2330_TW_value_trap_block.txt"))
    assert vt["score"] == 95
    assert vt["verdict"] == "ALIGNED"
    assert vt["trap_risk"] == "LOW"


def test_valuation_params_golden():
    vp = _extract_params(_read("2330_TW_valuation_params.txt"))
    assert vp.method == "P/E_NORMALIZATION"
    assert vp.current_pe == 27.47
    assert vp.peg_ratio == 1.02
    assert vp.confidence == "HIGH"


def test_trade_block_golden():
    tb = parse_trade_block(_read("2330_TW_trade_block.txt"))
    assert tb is not None
    assert tb.action == "HOLD"


# --- structural invariants (verdict-independent) --------------------------------


def test_extreme_leverage_forces_reject():
    """Invariant: D/E above the standard-sector reject threshold ⇒ REJECT.

    The body's ``- D/E: 0.20`` line is what the parser reads first; swapping it
    for an extreme value must flip pre-screening deterministically.
    """
    block = _read("2330_TW_data_block.txt").replace("D/E: 0.20", "D/E: 9.00")
    flags, pre_screen = detect_red_flags(
        extract_metrics(block), sector=detect_sector(block)
    )
    assert pre_screen == "REJECT"
    assert any(f["type"] == "EXTREME_LEVERAGE" for f in flags)


# --- golden snapshot (catches any unexpected field drift) -----------------------


def _metrics_snapshot(block: str) -> dict:
    metrics = extract_metrics(block)
    metrics.pop("_raw_report", None)  # huge + circular; not part of the contract
    return metrics


def test_metrics_snapshot_stable():
    current = _metrics_snapshot(_read("2330_TW_data_block.txt"))
    if os.environ.get("UPDATE_REPLAY_SNAPSHOTS") == "1":
        _SNAPSHOT.write_text(
            json.dumps(current, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        pytest.skip("snapshot regenerated")
    assert _SNAPSHOT.exists(), (
        "metrics snapshot missing — regenerate with "
        "UPDATE_REPLAY_SNAPSHOTS=1 pytest tests/eval/test_deterministic_replay.py"
    )
    expected = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    # Compare via JSON normalization so tuple/list and number formatting agree.
    assert json.loads(json.dumps(current, sort_keys=True, default=str)) == expected, (
        "extract_metrics output drifted from the committed snapshot — if intended, "
        "regenerate with UPDATE_REPLAY_SNAPSHOTS=1"
    )


# --- error handling -------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["", "no structured block here", "### --- START DATA_BLOCK ---"]
)
def test_consumers_degrade_on_malformed_input(bad: str):
    """Malformed/empty input returns typed empties — never raises."""
    metrics = extract_metrics(bad)
    assert metrics["debt_to_equity"] is None
    pm = extract_pm_block(bad)
    assert pm.verdict is None
    assert parse_trade_block(bad) is None
    assert extract_value_trap_score(bad)["score"] is None


# --- evidence → claim → decision → publication contract -----------------------


def _block(*lines: str) -> str:
    return (
        "### --- START DATA_BLOCK ---\n"
        + "\n".join(lines)
        + "\n### --- END DATA_BLOCK ---"
    )


def test_claim_contract_replay_closes_remint_and_publication_seams():
    """One no-network replay traverses every deterministic contract boundary."""
    payload = {
        "trailingPE": 12.5,
        "revenueGrowth_MRQ": 0.168693,
        "_revenueGrowth_MRQ_source": "calculated_from_quarterly",
        "latest_quarter_date": "2025-12-31",
    }
    state = _register_metrics(
        {
            "raw_fundamentals_data": json.dumps(payload),
            "foreign_language_report": (
                "CAPACITY_UTILIZATION: N/A\n"
                "CAPACITY_UTILIZATION_SOURCE_URL: N/A\n"
                "CAPACITY_UTILIZATION_AS_OF: UNKNOWN\n"
                "CAPACITY_EVIDENCE_STATUS: UNSUPPORTED"
            ),
        },
        payload,
    )
    snapshot = build_pre_senior_snapshot(state)
    senior = _block(
        "PE_RATIO_TTM: 12.5",
        "REVENUE_GROWTH_MRQ: 16.9%",
        "LATEST_QUARTER_DATE: 2026-03-31",
        "CAPACITY_UTILIZATION: 95%",
        "CAPACITY_UTILIZATION_SOURCE_URL: https://search.example/result",
        "CAPACITY_UTILIZATION_AS_OF: UNKNOWN",
        "CAPACITY_EVIDENCE_STATUS: PRIMARY",
    )

    reconciled, conflicts = reconcile_data_block_projection(senior, snapshot)
    assert "LATEST_QUARTER_DATE: 2025-12-31" in reconciled
    assert "CAPACITY_UTILIZATION: N/A" in reconciled
    assert {conflict["field"] for conflict in conflicts} >= {
        "CAPACITY_UTILIZATION",
    }

    support_id = decision_claim_ids(snapshot)[0]
    pm_output = (
        "CONSULTANT_RESOLUTION:\n"
        "- VERDICT: UNVERIFIABLE\n\n"
        "### PORTFOLIO MANAGER VERDICT: HOLD\n"
        "### --- START PM_BLOCK ---\n"
        "VERDICT: HOLD\n"
        f"DECISION_FACTS: {support_id}\n"
        "DECISION_GATES: NONE\n"
        "### --- END PM_BLOCK ---"
    )
    final_pm, trace = reconcile_final_decision_trace(
        pm_output,
        snapshot,
        [{"type": "CAPACITY_EVIDENCE_GAP", "blocks_buy": True}],
    )
    assert "DECISION_GATES: CAPACITY_EVIDENCE_GAP" in final_pm
    assert trace["status"] == "VALID"

    article = "Capacity utilization is confirmed at 95%."
    assert audit_article_claim_support(article, snapshot)


def test_search_listing_cannot_bind_a_source_required_claim():
    url = "https://issuer.example/results"
    search_record = SimpleNamespace(
        sequence=3,
        content_sha256="abc123def456789",
        content=f"Search result points to {url}",
        urls=(url,),
        blocked=False,
        evidence_status="RESULTS_FOUND",
    )
    snapshot = build_analysis_snapshot(
        {
            "fundamentals_report": _block(
                "CAPACITY_UTILIZATION: 95%",
                f"CAPACITY_UTILIZATION_SOURCE_URL: {url}",
                "CAPACITY_UTILIZATION_AS_OF: 2025-12-31",
                "CAPACITY_EVIDENCE_STATUS: PRIMARY",
            )
        },
        [search_record],
    )
    claim = next(iter(snapshot["claims"].values()))

    assert claim["authority"] == "UNSUPPORTED"
    assert claim["decision_eligible"] is False


def test_annual_only_growth_does_not_inherit_newer_period_metadata():
    payload = {
        "earningsGrowth_MRQ": 1.0,
        "_earningsGrowth_MRQ_source": "aggregator",
        "latest_quarter_date": "2025-12-31",
    }
    snapshot = build_pre_senior_snapshot(
        _register_metrics(
            {
                "raw_fundamentals_data": json.dumps(payload),
                "foreign_language_report": "",
            },
            payload,
        )
    )
    claim = next(
        claim
        for claim in snapshot["claims"].values()
        if claim["field"] == "EARNINGS_GROWTH_MRQ"
    )

    assert claim["period"] is None


def test_uncited_acquisition_context_reaches_every_narrative_consumer():
    value_trap = (
        _block()
        .replace(
            "DATA_BLOCK",
            "VALUE_TRAP_BLOCK",
        )
        .replace(
            "\n### --- END",
            "\nM&A_CONTEXT_EVIDENCE: UNKNOWN\n### --- END",
        )
    )
    state = {
        "fundamentals_report": _block("PE_RATIO_TTM: 12.5"),
        "value_trap_report": value_trap,
        "artifact_statuses": {
            "fundamentals_report": {
                "complete": True,
                "ok": True,
                "content": _block("PE_RATIO_TTM: 12.5"),
            },
            "value_trap_report": {
                "complete": True,
                "ok": True,
                "content": value_trap,
            },
        },
    }

    constraints = downstream_evidence_constraints(state)
    assert "Do not name an acquisition" in constraints
    assert "infer acquisition-led growth" in constraints


def test_policy_registration_propagates_without_consumer_literals(monkeypatch):
    field = "SYNTHETIC_MATERIAL_SIGNAL"
    url = "https://issuer.example/signal"
    monkeypatch.setitem(
        MATERIAL_CLAIM_POLICIES,
        field,
        ClaimPolicy(
            source_url_field="SYNTHETIC_SOURCE_URL",
            authority_field="SYNTHETIC_AUTHORITY",
            source_required=True,
            decision_role="SUPPORT",
            aliases=("synthetic signal",),
        ),
    )
    evidence = SimpleNamespace(
        sequence=9,
        content_sha256="987654abcdef000",
        content="Fetched issuer document",
        urls=(url,),
        blocked=False,
        evidence_status="EVIDENCE_FOUND",
    )
    report = _block(
        f"{field}: 42%",
        f"SYNTHETIC_SOURCE_URL: {url}",
        "SYNTHETIC_AUTHORITY: PRIMARY",
    )
    snapshot = build_analysis_snapshot(
        {"fundamentals_report": report},
        [evidence],
        degraded=False,
    )
    claim_id = next(
        claim_id
        for claim_id, claim in snapshot["claims"].items()
        if claim["field"] == field
    )

    assert claim_id in render_decision_trace_instruction(snapshot, [])
    assert field in extract_source_confidence_context(report, None)
    article = (
        "The synthetic signal is 42%.\n"
        f"```CLAIM_USAGE\n- {claim_id} | The synthetic signal is 42%.\n```"
    )
    assert audit_article_claim_usage(article, snapshot) == []
