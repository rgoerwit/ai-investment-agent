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


_AUDITOR_STUB = (
    "AUDITOR_RESOLUTION:\n"
    "- FINDING: Forensic Auditor flagged anomalies not explicitly addressed by PM rationale.\n"
    "- DATA_CHECK: NOT_PROVIDED\n"
    "- VERDICT: UNVERIFIABLE\n"
)


class TestUnresolvedAuditorStub:
    def test_stub_reformatted_to_prose_caveat(self):
        cleaned = _reporter()._clean_text(f"Rationale text.\n\n{_AUDITOR_STUB}")
        assert "> **Auditor note:**" in cleaned
        assert "NOT_PROVIDED" not in cleaned
        assert "UNVERIFIABLE" not in cleaned

    def test_populated_auditor_block_untouched(self):
        populated = (
            "AUDITOR_RESOLUTION:\n"
            "- FINDING: DSO ballooning flagged.\n"
            "- DATA_CHECK: DATA_BLOCK DSO 92 days vs auditor 118 days.\n"
            "- VERDICT: CONFIRMED_RISK (+0.5)\n"
        )
        cleaned = _reporter()._clean_text(populated)
        assert "CONFIRMED_RISK" in cleaned
        assert "> **Auditor note:**" not in cleaned

    def test_auditor_resolution_none_untouched(self):
        cleaned = _reporter()._clean_text(_PM_WITH_RESOLUTION)
        assert "AUDITOR_RESOLUTION: NONE" in cleaned
        assert "> **Auditor note:**" not in cleaned

    def test_no_auditor_block_unchanged(self):
        text = "Just a rationale with no auditor block.\n"
        assert "> **Auditor note:**" not in _reporter()._clean_text(text)

    def test_fenced_stub_reformatted_without_orphan_fences(self):
        """A PM-authored fenced stub must not leave the blockquote trapped
        inside orphan ``` lines (the fence is consumed whole)."""
        fenced = f"Rationale.\n\n{_FENCE}\n{_AUDITOR_STUB}{_FENCE}\n\nMore text.\n"
        cleaned = _reporter()._clean_text(fenced)
        assert "> **Auditor note:**" in cleaned
        assert "NOT_PROVIDED" not in cleaned
        assert _FENCE not in cleaned  # no orphan fences remain
        assert "More text." in cleaned

    def test_fenced_populated_block_untouched(self):
        populated = (
            f"{_FENCE}\nAUDITOR_RESOLUTION:\n"
            "- FINDING: DSO ballooning flagged.\n"
            "- DATA_CHECK: DATA_BLOCK DSO 92 days vs auditor 118 days.\n"
            f"- VERDICT: CONFIRMED_RISK (+0.5)\n{_FENCE}\n"
        )
        cleaned = _reporter()._clean_text(populated)
        assert "CONFIRMED_RISK" in cleaned
        assert "> **Auditor note:**" not in cleaned


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
        # Banner prepended; raw prose left intact (no fragile in-place rewrite).
        assert "Coverage caveat:" in out
        assert out.endswith(self._SENTIMENT)

    def test_softened_when_data_quality_note_present(self):
        # has_note is a plain substring check; no DATA_BLOCK markers required.
        fund = "ANALYST_COVERAGE_DATA_QUALITY_NOTE: avoid unqualified hidden framing\n"
        out = QuietModeReporter._soften_undiscovered_language(self._SENTIMENT, fund)
        assert "Coverage caveat:" in out

    def test_no_ungrammatical_noun_phrase_splice(self):
        """The removed in-place regex spliced a noun phrase into header/label/
        predicate slots. The body must stay verbatim; only the banner is added."""
        sentiment = (
            "#### UNDISCOVERED STATUS ASSESSMENT\n"
            "**Status**: UNDISCOVERED\n"
            "The stock is genuinely undiscovered by retail crowds.\n"
        )
        out = QuietModeReporter._soften_undiscovered_language(
            sentiment, self._fund("MODERATE")
        )
        assert out.endswith(sentiment)  # body verbatim
        assert "#### low English-language aggregator visibility" not in out
        assert "is low English-language aggregator visibility by" not in out

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
