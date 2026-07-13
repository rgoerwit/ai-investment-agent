"""Input-stage guard: a hedged/approximate FLA filing OCF must not be promoted
to FILING authority (KTY.WA 2026-06-27)."""

from __future__ import annotations

from src.agents.support import compute_data_conflicts, filing_ocf_is_approximate

_RAW = '{"operatingCashflow": 920000000, "numberOfAnalystOpinions": 7}'


class TestFilingOcfIsApproximate:
    def test_tilde_value_flagged(self):
        assert filing_ocf_is_approximate(
            "Filing Cash Flow: N/A (Operating Cash Flow ~1.148 bln PLN for FY2025)"
        )

    def test_approx_keyword_flagged(self):
        assert filing_ocf_is_approximate("Operating Cash Flow (Filing): approx. 1.1B")

    def test_na_with_parenthetical_number_flagged(self):
        assert filing_ocf_is_approximate("Operating Cash Flow: N/A (about 950m est.)")

    def test_exact_value_not_flagged(self):
        assert not filing_ocf_is_approximate(
            "Operating Cash Flow (Filing): 937,000,000 PLN"
        )

    def test_plain_na_not_flagged(self):
        assert not filing_ocf_is_approximate("Operating Cash Flow (Filing): N/A")

    def test_none_and_empty(self):
        assert not filing_ocf_is_approximate(None)
        assert not filing_ocf_is_approximate("")


class TestComputeDataConflictsEmitsNonAuthoritative:
    def test_hedged_filing_ocf_emits_guidance(self):
        foreign = (
            "Filing Cash Flow: N/A (Operating Cash Flow ~1.148 bln PLN for FY2025)"
        )
        out = compute_data_conflicts(_RAW, foreign)
        assert "OCF_FILING_NON_AUTHORITATIVE" in out
        assert "do NOT set OPERATING_CASH_FLOW_SOURCE: FILING" in out

    def test_exact_filing_ocf_no_non_authoritative(self):
        foreign = "Operating Cash Flow (Filing): 937,000,000 PLN\nPeriod: FY2025"
        out = compute_data_conflicts(_RAW, foreign)
        assert "OCF_FILING_NON_AUTHORITATIVE" not in out

    def test_no_foreign_data_no_crash(self):
        assert "OCF_FILING_NON_AUTHORITATIVE" not in compute_data_conflicts(_RAW, "")
