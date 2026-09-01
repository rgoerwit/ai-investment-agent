"""Finite regression contract for screening, failure, and fast-fail routing."""

from __future__ import annotations

import pytest

from src.runtime_diagnostics import (
    build_analysis_outcome,
    build_analysis_validity,
    get_analysis_outcome,
    has_unreconciled_auditor_resolution,
)


def _complete_sync_state(**updates):
    state = {
        "artifact_statuses": {
            field: {"complete": True, "ok": True, "content": f"{field} content"}
            for field in (
                "market_report",
                "sentiment_report",
                "news_report",
                "value_trap_report",
            )
        },
        "financial_validation_complete": True,
    }
    state.update(updates)
    return state


def test_analysis_outcome_separates_issuer_rejection_from_unassessable_run() -> None:
    issuer_reject = build_analysis_outcome(
        "REJECT",
        [
            {
                "type": "LIQUIDITY_HARD_FAIL",
                "severity": "CRITICAL",
                "action": "AUTO_REJECT",
            }
        ],
    )
    unavailable = build_analysis_outcome(
        "REJECT",
        [
            {
                "type": "DATA_CONTRACT_INVALID",
                "severity": "CRITICAL",
                "action": "AUTO_REJECT",
            }
        ],
    )

    assert issuer_reject == {
        "schema_version": 1,
        "eligibility": "REJECTED",
        "run_status": "COMPLETED",
        "reason_codes": ["LIQUIDITY_HARD_FAIL"],
    }
    assert unavailable == {
        "schema_version": 1,
        "eligibility": "UNASSESSABLE",
        "run_status": "INSUFFICIENT_DATA",
        "reason_codes": ["DATA_CONTRACT_INVALID"],
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_unassessable_failure_dominates_issuer_rejection(reverse: bool) -> None:
    flags = [
        {
            "type": "LIQUIDITY_HARD_FAIL",
            "severity": "CRITICAL",
            "action": "AUTO_REJECT",
        },
        {
            "type": "VALIDATOR_EXECUTION_FAILED",
            "severity": "CRITICAL",
            "action": "AUTO_REJECT",
        },
    ]
    if reverse:
        flags.reverse()

    outcome = build_analysis_outcome("REJECT", flags)

    assert outcome == {
        "schema_version": 1,
        "eligibility": "UNASSESSABLE",
        "run_status": "FAILED_INFRASTRUCTURE",
        "reason_codes": ["VALIDATOR_EXECUTION_FAILED"],
    }


def test_unreconciled_auditor_resolution_has_one_canonical_predicate() -> None:
    assert has_unreconciled_auditor_resolution(
        """AUDITOR_RESOLUTION
DATA_CHECK: NOT_PROVIDED
VERDICT: UNVERIFIABLE"""
    )
    assert not has_unreconciled_auditor_resolution("STATUS: CLEAN")


def test_unknown_outcome_schema_is_not_treated_as_current_authority() -> None:
    outcome = get_analysis_outcome(
        {
            "pre_screening_result": "PASS",
            "red_flags": [],
            "analysis_outcome": {
                "schema_version": 2,
                "eligibility": "REJECTED",
                "run_status": "COMPLETED",
                "reason_codes": ["FUTURE_RULE"],
            },
        }
    )

    assert outcome["eligibility"] == "QUALIFIES"


def test_merged_screening_primitives_override_stale_stored_outcome() -> None:
    outcome = get_analysis_outcome(
        {
            "pre_screening_result": "REJECT",
            "red_flags": [
                {
                    "type": "LIQUIDITY_HARD_FAIL",
                    "severity": "CRITICAL",
                    "action": "AUTO_REJECT",
                }
            ],
            "analysis_outcome": {
                "schema_version": 1,
                "eligibility": "QUALIFIES",
                "run_status": "COMPLETED",
                "reason_codes": [],
            },
        }
    )

    assert outcome == {
        "schema_version": 1,
        "eligibility": "REJECTED",
        "run_status": "COMPLETED",
        "reason_codes": ["LIQUIDITY_HARD_FAIL"],
    }


def test_sync_router_ends_unassessable_run_without_calling_pm() -> None:
    from src.graph.routing import sync_check_router

    state = _complete_sync_state(
        pre_screening_result="REJECT",
        analysis_outcome={
            "schema_version": 1,
            "eligibility": "UNASSESSABLE",
            "run_status": "INSUFFICIENT_DATA",
            "reason_codes": ["DATA_CONTRACT_INVALID"],
        },
    )

    assert sync_check_router(state, {}, auditor_required=False) == "__end__"


def test_sync_router_sends_evidence_backed_rejection_to_fast_fail() -> None:
    from src.graph.routing import sync_check_router

    state = _complete_sync_state(
        pre_screening_result="REJECT",
        analysis_outcome={
            "schema_version": 1,
            "eligibility": "REJECTED",
            "run_status": "COMPLETED",
            "reason_codes": ["LIQUIDITY_HARD_FAIL"],
        },
    )

    assert sync_check_router(state, {}, auditor_required=False) == "PM Fast-Fail"


@pytest.mark.asyncio
async def test_deterministic_fast_fail_emits_valid_trace_without_model() -> None:
    from src.agents.decision_nodes import create_screen_rejection_node

    node = create_screen_rejection_node()
    result = await node(
        {
            "company_of_interest": "2173.T",
            "pre_screening_result": "REJECT",
            "analysis_outcome": {
                "schema_version": 1,
                "eligibility": "REJECTED",
                "run_status": "COMPLETED",
                "reason_codes": ["LIQUIDITY_HARD_FAIL"],
            },
            "analysis_snapshot": {
                "contract_status": "VALID",
                "claims": {},
                "scorecards": {},
            },
            "red_flags": [
                {
                    "type": "LIQUIDITY_HARD_FAIL",
                    "severity": "CRITICAL",
                    "action": "AUTO_REJECT",
                }
            ],
        },
        {},
    )

    assert result["artifact_statuses"]["final_trade_decision"]["ok"] is True
    assert result["decision_trace"]["status"] == "VALID"
    assert result["decision_trace"]["decision_gates"] == ["LIQUIDITY_HARD_FAIL"]
    assert result["decision_policy"]["source"] == "deterministic_screen"
    assert "VERDICT: DO_NOT_INITIATE" in result["final_trade_decision"]


def test_unassessable_outcome_is_never_publishable() -> None:
    validity = build_analysis_validity(
        {
            "pre_screening_result": "REJECT",
            "analysis_outcome": {
                "schema_version": 1,
                "eligibility": "UNASSESSABLE",
                "run_status": "INSUFFICIENT_DATA",
                "reason_codes": ["DATA_CONTRACT_INVALID"],
            },
            "red_flags": [
                {
                    "type": "DATA_CONTRACT_INVALID",
                    "severity": "CRITICAL",
                    "action": "AUTO_REJECT",
                }
            ],
        }
    )

    assert validity["publishable"] is False
    assert validity["analysis_outcome"]["eligibility"] == "UNASSESSABLE"
    assert "analysis_outcome" in validity["required_failures"]


def test_conflicting_stored_outcome_is_never_publishable() -> None:
    validity = build_analysis_validity(
        {
            "pre_screening_result": "REJECT",
            "analysis_outcome": {
                "schema_version": 1,
                "eligibility": "QUALIFIES",
                "run_status": "COMPLETED",
                "reason_codes": [],
            },
            "red_flags": [
                {
                    "type": "LIQUIDITY_HARD_FAIL",
                    "severity": "CRITICAL",
                    "action": "AUTO_REJECT",
                }
            ],
        }
    )

    assert validity["publishable"] is False
    assert validity["analysis_outcome"]["eligibility"] == "REJECTED"
    assert "analysis_outcome_consistency" in validity["required_failures"]


@pytest.mark.parametrize(
    "stored",
    [
        {
            "schema_version": True,
            "eligibility": "QUALIFIES",
            "run_status": "COMPLETED",
            "reason_codes": [],
        },
        {
            "schema_version": 1,
            "eligibility": "QUALIFIES",
            "run_status": "FAILED_INFRASTRUCTURE",
            "reason_codes": [],
        },
        {
            "schema_version": 1,
            "eligibility": "REJECTED",
            "run_status": "COMPLETED",
            "reason_codes": [7],
        },
    ],
)
def test_malformed_stored_outcome_is_never_publishable(stored: dict) -> None:
    validity = build_analysis_validity(
        {
            "pre_screening_result": "PASS",
            "red_flags": [],
            "analysis_outcome": stored,
        }
    )

    assert validity["publishable"] is False
    assert validity["analysis_outcome"]["eligibility"] == "QUALIFIES"
    assert "analysis_outcome_schema" in validity["required_failures"]


def test_outcome_consistency_failure_survives_repeated_enrichment(monkeypatch) -> None:
    from src.persistence import attach_run_summary

    monkeypatch.setattr(
        "src.persistence.build_run_summary",
        lambda result, **_kwargs: {
            "publishable": result["analysis_validity"]["publishable"]
        },
    )
    stale = {
        "schema_version": 1,
        "eligibility": "QUALIFIES",
        "run_status": "COMPLETED",
        "reason_codes": [],
    }
    result = {
        "pre_screening_result": "REJECT",
        "red_flags": [
            {
                "type": "LIQUIDITY_HARD_FAIL",
                "severity": "CRITICAL",
                "action": "AUTO_REJECT",
            }
        ],
        "analysis_outcome": stale.copy(),
    }

    attach_run_summary(result, quick_mode=True)
    first_failure = result["analysis_validity"]["required_failures"][
        "analysis_outcome_consistency"
    ]
    attach_run_summary(result, quick_mode=True)

    assert result["analysis_outcome"] == stale
    assert result["analysis_validity"]["publishable"] is False
    assert (
        result["analysis_validity"]["required_failures"]["analysis_outcome_consistency"]
        == first_failure
    )


@pytest.mark.asyncio
async def test_deterministic_fast_fail_fails_closed_on_invalid_trace(
    monkeypatch,
) -> None:
    from src.agents.decision_nodes import create_screen_rejection_node

    monkeypatch.setattr(
        "src.agents.decision_nodes.reconcile_final_decision_trace",
        lambda *_args, **_kwargs: (
            "VERDICT: DO_NOT_INITIATE",
            {"status": "INVALID", "reason": "MISSING_GATE"},
        ),
    )
    result = await create_screen_rejection_node()(
        {
            "company_of_interest": "2173.T",
            "pre_screening_result": "REJECT",
            "red_flags": [
                {
                    "type": "LIQUIDITY_HARD_FAIL",
                    "severity": "CRITICAL",
                    "action": "AUTO_REJECT",
                }
            ],
            "analysis_snapshot": {"contract_status": "VALID", "claims": {}},
        },
        {},
    )

    assert result["artifact_statuses"]["final_trade_decision"]["ok"] is False
    assert "decision_policy" not in result
