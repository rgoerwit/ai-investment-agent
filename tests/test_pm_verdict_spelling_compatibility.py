"""Every PM-verdict reader must accept every spelling the PM actually emits.

The narrative ``PORTFOLIO MANAGER VERDICT:`` header carries more than one spelling in
production. Measured across 1,493 persisted decisions:

    DO NOT INITIATE   1002
    HOLD               315
    BUY                114
    DO_NOT_INITIATE     20
    DO NOT_INITIATE      2

Four of the six readers accepted only the space-separated form and returned *no match*
at all on the other two -- not a wrong verdict, an absent one, silently falling through
to a default. Each reader was individually correct and individually tested; nothing
compared them, which is precisely the gap this table closes.

Companion to ``tests/test_pm_verdict_consistency.py``, which pins that these readers
import the shared canonicalizer. This one pins that they agree on *inputs*.
"""

from __future__ import annotations

import pytest

from src.charts.extractors.pm_block import extract_verdict_from_text
from src.pm_decision_parser import (
    PM_VERDICT_HEADER_RE,
    canonicalize_pm_verdict,
    parse_final_decision_scores,
)
from src.report_generator import QuietModeReporter
from src.reporting.memo import _VERDICT_NARRATIVE

# Spellings observed in the corpus, plus the bolded form the prompt allows.
HEADER_SPELLINGS = [
    pytest.param("DO NOT INITIATE", "DO_NOT_INITIATE", id="spaced"),
    pytest.param("DO_NOT_INITIATE", "DO_NOT_INITIATE", id="underscored"),
    pytest.param("DO NOT_INITIATE", "DO_NOT_INITIATE", id="mixed_separators"),
    pytest.param("BUY", "BUY", id="buy"),
    pytest.param("HOLD", "HOLD", id="hold"),
    pytest.param("SELL", "SELL", id="legacy_sell"),
    pytest.param("REJECT", "DO_NOT_INITIATE", id="legacy_reject"),
]


def _reporter() -> QuietModeReporter:
    return QuietModeReporter(ticker="TEST.T")


def _decision_text(spelling: str, *, bold: bool = False) -> str:
    header = f"**{spelling}**" if bold else spelling
    return (
        "### Decision Rationale\n"
        "The gates were assessed.\n\n"
        f"### PORTFOLIO MANAGER VERDICT: {header}\n"
    )


class TestHeaderSpellingsResolveEverywhere:
    @pytest.mark.parametrize(("spelling", "expected"), HEADER_SPELLINGS)
    def test_shared_header_pattern(self, spelling: str, expected: str) -> None:
        match = PM_VERDICT_HEADER_RE.search(_decision_text(spelling))
        assert match is not None, f"header pattern missed {spelling!r}"
        assert canonicalize_pm_verdict(match.group(1)) == expected

    @pytest.mark.parametrize(("spelling", "expected"), HEADER_SPELLINGS)
    def test_chart_extractor(self, spelling: str, expected: str) -> None:
        assert extract_verdict_from_text(_decision_text(spelling)) == expected

    @pytest.mark.parametrize(("spelling", "expected"), HEADER_SPELLINGS)
    def test_memo_narrative_reader(self, spelling: str, expected: str) -> None:
        match = _VERDICT_NARRATIVE.search(_decision_text(spelling))
        assert match is not None, f"memo reader missed {spelling!r}"
        assert canonicalize_pm_verdict(match.group(1)) == expected

    @pytest.mark.parametrize(("spelling", "expected"), HEADER_SPELLINGS)
    def test_report_generator_prose_reader(self, spelling: str, expected: str) -> None:
        rendered = _reporter()._extract_prose_decision(_decision_text(spelling))
        assert rendered is not None, f"report generator missed {spelling!r}"
        # _display_verdict may reformat for humans; canonicalizing back must round-trip.
        assert canonicalize_pm_verdict(rendered) == expected

    @pytest.mark.parametrize(("spelling", "expected"), HEADER_SPELLINGS)
    def test_free_text_score_parser(self, spelling: str, expected: str) -> None:
        parsed = parse_final_decision_scores(_decision_text(spelling))
        assert canonicalize_pm_verdict(parsed.get("verdict")) == expected

    @pytest.mark.parametrize(("spelling", "expected"), HEADER_SPELLINGS)
    def test_bolded_header_is_equivalent(self, spelling: str, expected: str) -> None:
        text = _decision_text(spelling, bold=True)
        match = PM_VERDICT_HEADER_RE.search(text)
        assert match is not None, f"bolded header missed {spelling!r}"
        assert canonicalize_pm_verdict(match.group(1)) == expected
        assert extract_verdict_from_text(text) == expected


class TestReadersAgreeWithEachOther:
    """The property that matters: no reader may disagree with another."""

    @pytest.mark.parametrize(("spelling", "expected"), HEADER_SPELLINGS)
    def test_all_readers_return_the_same_canonical_verdict(
        self, spelling: str, expected: str
    ) -> None:
        text = _decision_text(spelling)
        header = PM_VERDICT_HEADER_RE.search(text)
        memo = _VERDICT_NARRATIVE.search(text)
        results = {
            "shared_header": canonicalize_pm_verdict(header.group(1) if header else ""),
            "chart_extractor": extract_verdict_from_text(text),
            "memo": canonicalize_pm_verdict(memo.group(1) if memo else ""),
            "report_generator": canonicalize_pm_verdict(
                _reporter()._extract_prose_decision(text) or ""
            ),
            "score_parser": canonicalize_pm_verdict(
                parse_final_decision_scores(text).get("verdict")
            ),
        }
        assert set(results.values()) == {expected}, (
            f"readers disagree on {spelling!r}: {results}"
        )


class TestMalformedInputDegradesSafely:
    def test_unknown_token_is_unparseable_not_a_guess(self) -> None:
        text = _decision_text("MAYBE LATER")
        assert extract_verdict_from_text(text) is None
        assert (
            canonicalize_pm_verdict(parse_final_decision_scores(text).get("verdict"))
            == "UNPARSEABLE"
        )

    def test_absent_header_falls_back_to_hold(self) -> None:
        """report_generator's documented safe default, not an exception."""
        assert _reporter().extract_decision("No verdict anywhere.") == "HOLD"
