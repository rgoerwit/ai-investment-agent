"""Deterministic PM-claim audit: 2a number-consistency + 2b provenance gate."""

from __future__ import annotations

from src.pm_claim_audit import audit_pm_claims, validate_decision_trace


def _block(lines: str) -> str:
    return "### --- START DATA_BLOCK ---\n" + lines + "\n### --- END DATA_BLOCK ---"


# GTT profile: raw backlog unknown, coverage is an *estimate*.
_GTT_BLOCK = _block(
    "ADJUSTED_HEALTH_SCORE: 83% (based on 12 available points)\n"
    "PE_RATIO_TTM: 16.95\n"
    "REVENUE_BACKLOG: N/A\n"
    "REVENUE_BACKLOG_COVERAGE: 4.0 yrs"
)


class TestProvenanceGate2b:
    def test_gtt_backlog_overclaim_fires_once(self):
        # base N/A + "contractually secured" near a backlog alias => one caveat.
        pm = (
            "GTT's 4.0-year revenue backlog provides highly visible, "
            "contractually secured cash flows that decouple it from cyclicality."
        )
        out, caveats = audit_pm_claims(pm, fundamentals=_GTT_BLOCK, ticker="GTT.PA")
        assert len(caveats) == 1
        assert "REVENUE_BACKLOG_COVERAGE" in caveats[0]["claim"]
        assert "PM CLAIM CAVEAT" in out
        # The 4.0 the PM cites MATCHES the DATA_BLOCK — 2a must NOT also fire.
        assert sum("REVENUE_BACKLOG_COVERAGE" in c["claim"] for c in caveats) == 1

    def test_weak_provenance_without_certainty_term_no_fire(self):
        pm = "GTT has 4.0-year backlog coverage supporting revenue visibility."
        _, caveats = audit_pm_claims(pm, fundamentals=_GTT_BLOCK, ticker="GTT.PA")
        assert caveats == []

    def test_threshold_prose_with_base_present_no_fire(self):
        # Base present => provenance not weak; even a kill-criterion number nearby
        # (and any certainty word) must not fire.
        block = _block(
            "REVENUE_BACKLOG: EUR 2.5B\n"
            "REVENUE_BACKLOG_COVERAGE: 4.0 yrs\n"
            "PE_RATIO_TTM: 16.95"
        )
        pm = "Kill criterion: backlog coverage drops below 2.0 years (confirmed trigger)."
        _, caveats = audit_pm_claims(pm, fundamentals=block, ticker="X")
        assert caveats == []

    def test_reliability_flagged_valuation_fires(self):
        block = _block("PE_RATIO_TTM: 12.0\nVALUATION_INPUT_RELIABILITY: LOW")
        pm = "The P/E of 12 confirms the stock is cheap as ground truth."
        _, caveats = audit_pm_claims(pm, fundamentals=block, ticker="X")
        assert len(caveats) == 1
        assert "PE_RATIO_TTM" in caveats[0]["claim"]

    def test_reliability_clean_valuation_no_fire(self):
        block = _block("PE_RATIO_TTM: 12.0\nVALUATION_INPUT_RELIABILITY: HIGH")
        pm = "The P/E of 12 confirms the stock is cheap as ground truth."
        _, caveats = audit_pm_claims(pm, fundamentals=block, ticker="X")
        assert caveats == []


class TestNumberConsistency2a:
    _BLOCK = _block(
        "ADJUSTED_HEALTH_SCORE: 83% (based on 12 available points)\n"
        "PE_RATIO_TTM: 16.95\nCURRENT_PRICE: 188.10"
    )

    def test_hard_field_mismatch_caveats(self):
        pm = "Fundamentals are strong (ADJUSTED_HEALTH_SCORE: 72%) per our scoring."
        out, caveats = audit_pm_claims(pm, fundamentals=self._BLOCK, ticker="X")
        assert len(caveats) == 1
        assert "ADJUSTED_HEALTH_SCORE" in caveats[0]["claim"]
        assert "83" in caveats[0]["ground_truth"]
        assert "PM CLAIM CAVEAT" in out

    def test_hard_field_match_no_caveat(self):
        pm = "Trades at (PE_RATIO_TTM: 16.95), a reasonable multiple."
        _, caveats = audit_pm_claims(pm, fundamentals=self._BLOCK, ticker="X")
        assert caveats == []

    def test_gate_input_mismatch_is_still_a_hard_claim_error(self):
        block = _block("DE_RATIO: 20%")
        pm = "Leverage is low (DE_RATIO: 80%)."

        _, caveats = audit_pm_claims(pm, fundamentals=block, ticker="X")

        assert len(caveats) == 1
        assert "DE_RATIO" in caveats[0]["claim"]

    def test_derived_prose_number_never_caveats(self):
        pm = "Our weighted intrinsic value is $125, implying 35% upside from here."
        out, caveats = audit_pm_claims(pm, fundamentals=self._BLOCK, ticker="X")
        assert caveats == []
        assert "PM CLAIM CAVEAT" not in out

    def test_nonhard_key_mismatch_no_caveat(self):
        # A non-hard KEY:VALUE mismatch is debug-logged, never caveated.
        pm = "Note (SOME_OTHER_FIELD: 999) in passing."
        _, caveats = audit_pm_claims(pm, fundamentals=self._BLOCK, ticker="X")
        assert caveats == []

    def test_rounded_hard_field_citation_no_longer_caveats(self):
        # Shared-matcher fix: the PM audit imports _citation_values_match, so a
        # legitimately rounded hard-field citation (canonical 13.63 -> 13.6%)
        # must no longer caveat.
        block = _block("DE_RATIO: 13.63%")
        pm = "Leverage is contained (DE_RATIO: 13.6%)."
        _, caveats = audit_pm_claims(pm, fundamentals=block, ticker="X")
        assert caveats == []

    def test_wrong_scale_hard_field_citation_still_caveats(self):
        block = _block("DE_RATIO: 13.63%")
        pm = "Leverage is negligible (DE_RATIO: 1.36%)."
        _, caveats = audit_pm_claims(pm, fundamentals=block, ticker="X")
        assert len(caveats) == 1
        assert "DE_RATIO" in caveats[0]["claim"]


class TestAuditContract:
    def test_idempotent(self):
        pm = "GTT's 4.0-year revenue backlog provides contractually secured cash flows."
        out1, cav1 = audit_pm_claims(pm, fundamentals=_GTT_BLOCK, ticker="X")
        out2, cav2 = audit_pm_claims(out1, fundamentals=_GTT_BLOCK, ticker="X")
        assert out2 == out1
        assert cav2 == []
        assert out2.count("PM CLAIM CAVEAT") == 1

    def test_no_datablock_is_noop(self):
        pm = "Some verdict prose with contractually secured backlog claims."
        out, caveats = audit_pm_claims(pm, fundamentals="no block here", ticker="X")
        assert caveats == []
        assert out == pm

    def test_empty_inputs_noop(self):
        assert audit_pm_claims("", fundamentals=_GTT_BLOCK) == ("", [])
        assert audit_pm_claims("text", fundamentals=None) == ("text", [])


class TestDecisionTraceContract:
    _SNAPSHOT = {
        "contract_status": "VALID",
        "claims": {
            "claim:pe": {
                "field": "PE_RATIO_TTM",
                "value": "12.0",
                "decision_eligible": True,
                "decision_role": "SUPPORT",
            }
        },
    }

    @staticmethod
    def _pm(prose: str) -> str:
        return (
            f"{prose}\n"
            "### --- START PM_BLOCK ---\n"
            "VERDICT: HOLD\n"
            "DECISION_FACTS: claim:pe\n"
            "DECISION_GATES: NONE\n"
            "### --- END PM_BLOCK ---"
        )

    def test_negated_guidance_mention_is_advisory_not_structural_failure(self):
        trace = validate_decision_trace(
            self._pm("The broker forecast is not management guidance."),
            self._SNAPSHOT,
            [],
        )

        assert trace["status"] == "VALID"
        assert trace["advisory_source_families"] == ["GUIDANCE"]

    def test_uncited_source_sensitive_prose_does_not_replace_claim_contract(self):
        trace = validate_decision_trace(
            self._pm("Management guidance calls for revenue growth."),
            self._SNAPSHOT,
            [],
        )

        assert trace["status"] == "VALID"
        assert trace["advisory_source_families"] == ["GUIDANCE"]

    def test_invalid_claim_identifier_remains_a_structural_failure(self):
        pm = self._pm("Valuation is reasonable.").replace(
            "DECISION_FACTS: claim:pe",
            "DECISION_FACTS: claim:not-registered",
        )

        trace = validate_decision_trace(pm, self._SNAPSHOT, [])

        assert trace["status"] == "INVALID"
        assert trace["invalid_facts"] == ["claim:not-registered"]
