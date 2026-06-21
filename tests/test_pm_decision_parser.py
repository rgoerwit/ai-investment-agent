"""Tests for the neutral PM decision parser (src.pm_decision_parser)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pm_decision_parser import canonicalize_pm_verdict, parse_final_decision_scores
from tests.import_boundary import assert_no_offenders

_REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCanonicalizePmVerdict:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("BUY", "BUY"),
            ("hold", "HOLD"),
            (" sell ", "SELL"),
            ("DO_NOT_INITIATE", "DO_NOT_INITIATE"),
            ("DO NOT INITIATE", "DO_NOT_INITIATE"),
            ("Do-Not-Initiate", "DO_NOT_INITIATE"),
            ("DONOTINITATE", "DO_NOT_INITIATE"),  # known typo alias
            ("REJECT", "DO_NOT_INITIATE"),
            ("", "UNPARSEABLE"),
            (None, "UNPARSEABLE"),
            ("maybe", "UNPARSEABLE"),
        ],
    )
    def test_canonicalize(self, raw, expected):
        assert canonicalize_pm_verdict(raw) == expected


class TestParseFinalDecisionScores:
    def test_verdict_from_block(self):
        assert parse_final_decision_scores("VERDICT: BUY").get("verdict") == "BUY"

    def test_verdict_from_portfolio_manager_prose(self):
        text = "PORTFOLIO MANAGER VERDICT: DO NOT INITIATE\nRationale: ..."
        assert parse_final_decision_scores(text).get("verdict") == "DO_NOT_INITIATE"

    def test_verdict_from_action_bold(self):
        assert (
            parse_final_decision_scores("**Action**: **BUY**").get("verdict") == "BUY"
        )

    def test_verdict_returned_raw_not_canonicalized(self):
        # REJECT must survive verbatim; the analysis-index call site decides how
        # to normalize. Canonicalizing here would silently rewrite stored verdicts.
        assert parse_final_decision_scores("VERDICT: REJECT").get("verdict") == "REJECT"

    def test_zone_does_not_bleed_into_verdict(self):
        result = parse_final_decision_scores("VERDICT: BUY\nZONE: MODERATE")
        assert result.get("verdict") == "BUY"
        assert result.get("zone") == "MODERATE"

    def test_health_and_growth_adjustments(self):
        result = parse_final_decision_scores("HEALTH_ADJ: 62.5\nGROWTH_ADJ: 55")
        assert result.get("health_adj") == 62.5
        assert result.get("growth_adj") == 55.0

    def test_signed_risk_tally(self):
        assert parse_final_decision_scores("RISK_TALLY: -0.5").get("risk_tally") == -0.5
        assert parse_final_decision_scores("RISK_TALLY: 2.25").get("risk_tally") == 2.25

    def test_malformed_risk_tally_safe(self):
        assert "risk_tally" not in parse_final_decision_scores("RISK_TALLY: --")

    def test_unparseable_text_has_no_verdict(self):
        assert "verdict" not in parse_final_decision_scores("no decision here")


def test_parser_module_imports_no_heavy_deps():
    """The neutral parser must stay dependency-free (stdlib only) so any
    lightweight caller can reuse it without dragging in agents/charts/langchain."""
    assert_no_offenders(
        "import sys\n"
        "import src.pm_decision_parser  # noqa\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'src', 'langchain', 'langchain_core', 'langgraph', 'pydantic'} "
        "and m not in {'src', 'src.pm_decision_parser'})\n"
        "print('HEAVY:' + ','.join(heavy))\n",
        sentinel="HEAVY",
        cwd=str(_REPO_ROOT),
        message="src.pm_decision_parser pulled in non-stdlib deps",
    )
