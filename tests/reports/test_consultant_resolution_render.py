"""Render-layer regressions (KTY.WA 2026-06-27).

1. The PM's CONSULTANT_RESOLUTION reconciliation must survive `_clean_text` as
   reader-facing bullets — previously the whole block was stripped and the PM's
   surrounding fence was left as an empty ``` block under the heading.
2. Unconditional "undiscovered" language is qualified when coverage is not
   confirmed low.
"""

from __future__ import annotations

import re

from src.report_generator import QuietModeReporter

_FENCE = "```"

_PM_WITH_RESOLUTION = f"""#### CONSULTANT DISAGREEMENT RESOLUTION

{_FENCE}
CONSULTANT_RESOLUTION:
- CONCERN: Internal net-income inconsistency (592M vs 568M PLN)
- DATA_CHECK: DATA_BLOCK shows Net Income of 592M PLN, but native-source confirms 568M PLN.
- VERDICT: CONFIRMED_RISK (+0.75, Tier C2 conflict on Net Income)
{_FENCE}

{_FENCE}
APAC_RESOLUTION: NONE
{_FENCE}

{_FENCE}
AUDITOR_RESOLUTION: NONE
{_FENCE}
"""


def _reporter() -> QuietModeReporter:
    return QuietModeReporter(
        ticker="KTY.WA",
        company_name="Grupa Kety",
        quick_mode=False,
        skip_charts=True,
    )


class TestConsultantResolutionRender:
    def test_reconciliation_bullets_survive(self):
        cleaned = _reporter()._clean_text(_PM_WITH_RESOLUTION)
        assert "CONCERN: Internal net-income inconsistency" in cleaned
        assert "VERDICT: CONFIRMED_RISK" in cleaned

    def test_machine_label_removed(self):
        cleaned = _reporter()._clean_text(_PM_WITH_RESOLUTION)
        assert "CONSULTANT_RESOLUTION:" not in cleaned

    def test_no_empty_fence_left_under_heading(self):
        cleaned = _reporter()._clean_text(_PM_WITH_RESOLUTION)
        assert re.search(r"```[ \t]*\n[ \t]*```", cleaned) is None

    def test_apac_and_auditor_none_blocks_still_render(self):
        cleaned = _reporter()._clean_text(_PM_WITH_RESOLUTION)
        assert "APAC_RESOLUTION: NONE" in cleaned
        assert "AUDITOR_RESOLUTION: NONE" in cleaned

    def test_heading_preserved(self):
        cleaned = _reporter()._clean_text(_PM_WITH_RESOLUTION)
        assert "CONSULTANT DISAGREEMENT RESOLUTION" in cleaned

    def test_no_crash_on_malformed_unterminated_fence(self):
        text = f"#### CONSULTANT DISAGREEMENT RESOLUTION\n\n{_FENCE}\nCONSULTANT_RESOLUTION:\n- CONCERN: x\n"
        # Must not raise; bullets either preserved or left intact.
        cleaned = _reporter()._clean_text(text)
        assert "CONCERN: x" in cleaned


class TestSoftenUndiscovered:
    _SENTIMENT = (
        "Status: UNDISCOVERED (Strong positive).\n"
        "Undiscovered Status: PASS (Strongly Undiscovered).\n"
    )

    @staticmethod
    def _fund(total_est: str) -> str:
        return (
            "### --- START DATA_BLOCK ---\n"
            f"ANALYST_COVERAGE_TOTAL_EST: {total_est}\n"
            "### --- END DATA_BLOCK ---"
        )

    def test_softened_when_total_coverage_moderate(self):
        out = QuietModeReporter._soften_undiscovered_language(
            self._SENTIMENT, self._fund("MODERATE")
        )
        assert "undiscovered" not in out.lower()
        assert "low English-language aggregator visibility" in out

    def test_softened_when_data_quality_note_present(self):
        # has_note is a plain substring check; no DATA_BLOCK markers required.
        fund = "ANALYST_COVERAGE_DATA_QUALITY_NOTE: avoid unqualified hidden framing\n"
        out = QuietModeReporter._soften_undiscovered_language(self._SENTIMENT, fund)
        assert "undiscovered" not in out.lower()

    def test_unchanged_when_coverage_confirmed_low(self):
        out = QuietModeReporter._soften_undiscovered_language(
            self._SENTIMENT, self._fund("LOW")
        )
        assert out == self._SENTIMENT

    def test_idempotent(self):
        fund = self._fund("MODERATE")
        once = QuietModeReporter._soften_undiscovered_language(self._SENTIMENT, fund)
        twice = QuietModeReporter._soften_undiscovered_language(once, fund)
        assert once == twice

    def test_empty_and_missing_fields_no_crash(self):
        assert QuietModeReporter._soften_undiscovered_language("", "x") == ""
        # No coverage signal at all -> unchanged.
        assert (
            QuietModeReporter._soften_undiscovered_language(self._SENTIMENT, "")
            == self._SENTIMENT
        )

    def test_preceding_space_preserved_without_modifier(self):
        """Regression (145020.KQ): the old regex consumed the space before a
        bare "undiscovered", yielding "forlow English-language…"."""
        out = QuietModeReporter._soften_undiscovered_language(
            "Positive for undiscovered thesis.\n", self._fund("MODERATE")
        )
        assert "for low English-language aggregator visibility thesis" in out
        assert "forlow" not in out

    def test_preceding_newline_preserved_without_modifier(self):
        out = QuietModeReporter._soften_undiscovered_language(
            "Positive for\nundiscovered thesis.\n", self._fund("MODERATE")
        )
        assert "for\nlow English-language aggregator visibility thesis" in out

    def test_modifier_still_consumed_with_its_whitespace(self):
        out = QuietModeReporter._soften_undiscovered_language(
            "Status: PASS (Strongly Undiscovered).\n", self._fund("MODERATE")
        )
        assert "(low English-language aggregator visibility)" in out
        assert "Strongly low" not in out

    def test_caveat_banner_neutralizes_synonyms(self):
        # Synonym overclaims with no literal "undiscovered" still get the caveat.
        synonym_text = (
            "The stock is effectively invisible to Western retail investors and "
            "entirely absent from the global investor consciousness.\n"
        )
        out = QuietModeReporter._soften_undiscovered_language(
            synonym_text, self._fund("MODERATE")
        )
        assert "Coverage caveat:" in out
        assert out.endswith(synonym_text)  # banner prepended, body intact

    def test_caveat_not_added_when_coverage_low(self):
        synonym_text = "The stock is effectively invisible to Western retail.\n"
        out = QuietModeReporter._soften_undiscovered_language(
            synonym_text, self._fund("LOW")
        )
        assert "Coverage caveat:" not in out
