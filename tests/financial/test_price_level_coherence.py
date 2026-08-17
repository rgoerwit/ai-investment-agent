"""Price levels must share a scale with the price they were derived from.

The defect this pins (GAMA.L, 2026-08-15): one artifact carried
``CURRENT_PRICE: 9.76`` with ``ENTRY 900 / STOP 750 / TARGET_1 1150`` and
labelled every one of them ``GBP``. Currency-code agreement cannot catch that —
both sides said GBP — because the disagreement is a *scale* inside one code.

Downstream it rendered as ``REVIEW GAMA.L price drift 98.9% down [SELL]`` on a
stock that had not moved, because five separate consumers compare an analysis
price against a live broker price and none of them checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.fundamentals_reconciler import (
    stamp_price_currency,
    stamp_trade_block_price_currency,
)
from src.fx_normalization import (
    PRICE_LEVEL_BAND,
    PriceLevelCoherence,
    assess_price_level_coherence,
    comparable_prices,
)

# The real GAMA.L_20260815_100957 numbers, kept verbatim so this test keeps
# describing a defect that actually occurred rather than a plausible one.
GAMA_PRICE = 9.76
GAMA_LEVELS = (900.0, 750.0, 1150.0, 1400.0)


class TestCoherenceContract:
    def test_ordinary_levels_are_coherent(self):
        verdict = assess_price_level_coherence(100.0, [110.0, 85.0, 130.0, 150.0])
        assert verdict.status is PriceLevelCoherence.COHERENT
        assert not verdict.is_incoherent

    def test_the_real_gama_block_is_incoherent(self):
        verdict = assess_price_level_coherence(GAMA_PRICE, GAMA_LEVELS)
        assert verdict.status is PriceLevelCoherence.INCOHERENT
        # The hint must survive: the worst ratio is 143x, but the ~100x
        # signature sits on a *different* level (900/9.76 = 92x). Testing only
        # the extreme ratio would silently drop the diagnosis.
        assert "minor-unit" in verdict.reason

    def test_a_generous_stretch_target_is_not_flagged(self):
        """The band must not punish a legitimately wide valuation range."""
        verdict = assess_price_level_coherence(10.0, [12.0, 8.0, 60.0, 150.0])
        assert verdict.status is PriceLevelCoherence.COHERENT

    @pytest.mark.parametrize(
        ("reference", "levels", "why"),
        [
            (None, [900.0], "no reference price"),
            (0.0, [900.0], "zero reference price"),
            (-5.0, [900.0], "negative reference price"),
            (9.76, [None, 0.0, -3.0], "no usable levels"),
            (9.76, [], "empty levels"),
        ],
    )
    def test_unknowns_are_unassessed_never_coherent(self, reference, levels, why):
        """UNASSESSED must not collapse into COHERENT.

        Reading 'I could not tell' as 'this is fine' is the `is_quick_mode`
        tri-state mistake. Callers leave levels alone on UNASSESSED, so folding
        it into COHERENT would be invisible until a bad level slipped through.
        """
        verdict = assess_price_level_coherence(reference, levels)
        assert verdict.status is PriceLevelCoherence.UNASSESSED, why
        assert not verdict.is_incoherent

    def test_band_boundaries_are_inclusive(self):
        low, high = PRICE_LEVEL_BAND
        assert (
            assess_price_level_coherence(100.0, [100.0 * high]).status
            is PriceLevelCoherence.COHERENT
        )
        assert (
            assess_price_level_coherence(100.0, [100.0 * low]).status
            is PriceLevelCoherence.COHERENT
        )
        assert (
            assess_price_level_coherence(100.0, [100.0 * high * 1.01]).status
            is PriceLevelCoherence.INCOHERENT
        )

    def test_a_hundredth_scale_error_is_caught_symmetrically(self):
        """Pence price against pounds levels, the inverse of the GAMA case."""
        verdict = assess_price_level_coherence(976.0, [9.0, 7.5, 11.5])
        assert verdict.status is PriceLevelCoherence.INCOHERENT
        assert "minor-unit" in verdict.reason

    def test_worst_ratio_is_chosen_on_a_log_scale(self):
        """0.01x is as wrong as 100x; neither may be preferred by magnitude."""
        verdict = assess_price_level_coherence(100.0, [0.5, 300.0])
        assert verdict.status is PriceLevelCoherence.INCOHERENT
        assert verdict.worst_ratio == pytest.approx(0.005)


class TestProducerRefusesToAssertAFalseDenomination:
    """`stamp_trade_block_price_currency` must not stamp a contradicted code."""

    @staticmethod
    def _fundamentals(price: str, currency: str = "GBP") -> str:
        return (
            "### --- START DATA_BLOCK ---\n"
            f"CURRENT_PRICE: {price}\n"
            f"PRICE_CURRENCY: {currency}\n"
            "### --- END DATA_BLOCK ---"
        )

    @staticmethod
    def _trade_block(entry, stop, t1, t2) -> str:
        return (
            f"ACTION: BUY\nENTRY: {entry}\nSTOP: {stop}\n"
            f"TARGET_1: {t1}\nTARGET_2: {t2}\n"
        )

    def test_coherent_levels_stamp_the_code_unchanged(self):
        """The non-regression that matters: every non-.L artifact is untouched."""
        content = self._trade_block(95.0, 80.0, 130.0, 150.0)
        stamped = stamp_trade_block_price_currency(
            content, self._fundamentals("100.00", "USD"), ticker="TEST"
        )
        assert "PRICE_CURRENCY: USD" in stamped
        assert "N/A" not in stamped

    def test_the_real_gama_shape_stamps_not_available(self):
        content = self._trade_block(*GAMA_LEVELS)
        stamped = stamp_trade_block_price_currency(
            content, self._fundamentals(str(GAMA_PRICE)), ticker="GAMA.L"
        )
        assert "PRICE_CURRENCY: N/A" in stamped
        assert "PRICE_CURRENCY: GBP" not in stamped

    def test_levels_themselves_are_never_rewritten(self):
        """Refuse the label; never guess a rescale. We cannot tell which side is wrong."""
        content = self._trade_block(*GAMA_LEVELS)
        stamped = stamp_trade_block_price_currency(
            content, self._fundamentals(str(GAMA_PRICE)), ticker="GAMA.L"
        )
        for level in GAMA_LEVELS:
            assert f"{level}" in stamped

    def test_missing_data_block_leaves_content_untouched(self):
        content = self._trade_block(900.0, 750.0, 1150.0, 1400.0)
        assert stamp_trade_block_price_currency(content, "no block here") == content

    def test_absent_levels_still_stamp_the_code(self):
        """UNASSESSED is not INCOHERENT: nothing contradicts the code."""
        stamped = stamp_trade_block_price_currency(
            "ACTION: HOLD\nTARGET_2: N/A\n",
            self._fundamentals("9.76"),
            ticker="GAMA.L",
        )
        assert "PRICE_CURRENCY: GBP" in stamped

    def test_existing_injection_and_duplicate_guards_still_hold(self):
        """A value containing a regex template must not be interpreted."""
        content = (
            "ACTION: BUY\nENTRY: 95.00\nSTOP: 80.00\nTARGET_1: 130.00\n"
            "TARGET_2: 150.00\nPRICE_CURRENCY: \\g<0>\nPRICE_CURRENCY: XXX\n"
        )
        stamped = stamp_trade_block_price_currency(
            content, self._fundamentals("100.00", "USD"), ticker="TEST"
        )
        assert stamped.count("PRICE_CURRENCY: USD") == 2
        assert "\\g<0>" not in stamped


class TestLoaderRefusesIncoherentLevels:
    """The reader is the boundary that protects all five price consumers."""

    @staticmethod
    def _artifact(tmp_path: Path, ticker: str, price: float, levels) -> Path:
        entry, stop, t1, t2 = levels
        path = tmp_path / f"{ticker}_20260815_100957_analysis.json"
        path.write_text(
            json.dumps(
                {
                    "prediction_snapshot": {
                        "ticker": ticker,
                        "analysis_date": "2026-08-15",
                        "verdict": "HOLD",
                        "current_price": price,
                        "currency": "GBP",
                        "entry_price": entry,
                        "stop_price": stop,
                        "target_1_price": t1,
                        "target_2_price": t2,
                        "health_adj": 70.0,
                        "growth_adj": 60.0,
                    },
                    "reports": {},
                }
            )
        )
        return path

    def _load(self, path: Path):
        from src.ibkr.analysis_index import _build_analysis_record_from_file

        return _build_analysis_record_from_file(path)

    def test_incoherent_levels_are_discarded_with_a_reason(self, tmp_path):
        record = self._load(self._artifact(tmp_path, "GAMA.L", GAMA_PRICE, GAMA_LEVELS))
        assert record is not None
        assert record.price_levels_coherent is False
        assert record.price_levels_incoherent_reason
        assert record.entry_price is None
        assert record.stop_price is None
        assert record.target_1_price is None
        assert record.target_2_price is None
        # The reference price is NOT discarded — it is not the value in doubt.
        assert record.current_price == GAMA_PRICE

    def test_coherent_levels_survive_intact(self, tmp_path):
        record = self._load(
            self._artifact(tmp_path, "TEST.L", 100.0, (95.0, 80.0, 130.0, 150.0))
        )
        assert record is not None
        assert record.price_levels_coherent is True
        assert record.price_levels_incoherent_reason is None
        assert record.entry_price == 95.0
        assert record.stop_price == 80.0

    def test_one_bad_level_discards_the_whole_block(self, tmp_path):
        """Scale is a property of the block, not of an individual field."""
        record = self._load(
            self._artifact(tmp_path, "TEST.L", 100.0, (95.0, 80.0, 130.0, 15000.0))
        )
        assert record is not None
        assert record.price_levels_coherent is False
        assert record.entry_price is None


class TestConsumersAreNeutralisedByTheLoader:
    """Nulling at the reader must silence every downstream comparison.

    This is the regression for the live report line. Each consumer already
    guards on None, which is why the fix needed no edit to any of them — and is
    exactly why it needs asserting: the protection is implicit.
    """

    @staticmethod
    def _record(coherent: bool):
        from src.ibkr.models import AnalysisRecord

        levels = (
            {"entry_price": 95.0, "stop_price": 80.0, "target_1_price": 130.0}
            if coherent
            else {}
        )
        return AnalysisRecord(
            ticker="GAMA.L",
            analysis_date="2026-08-15",
            verdict="HOLD",
            current_price=GAMA_PRICE,
            currency="GBP",
            price_levels_coherent=coherent,
            **levels,
        )

    def test_staleness_reports_no_price_drift(self):
        """The exact line that reached the operator: 'price drift 98.9% down'."""
        from src.ibkr.reconciliation_rules import check_staleness

        stale, reason = check_staleness(
            self._record(coherent=False), current_price_local=9.7587366
        )
        assert "price drift" not in reason
        assert not stale

    def test_review_level_breach_does_not_fire(self):
        from src.ibkr.reconciliation_rules import check_review_level_breach

        assert not check_review_level_breach(
            self._record(coherent=False), 9.7587366, "GBP"
        )

    def test_base_case_reference_is_not_reached(self):
        from src.ibkr.reconciliation_rules import check_base_case_reference_reached

        assert not check_base_case_reference_reached(
            self._record(coherent=False), 9.7587366, "GBP"
        )

    def test_a_coherent_record_still_produces_signals(self):
        """The guard must not have silenced everything indiscriminately."""
        from src.ibkr.reconciliation_rules import check_review_level_breach

        assert check_review_level_breach(self._record(coherent=True), 50.0, "GBP")


class TestComparablePricesContract:
    """The cross-denomination half: same economy, different unit."""

    def test_pence_and_pounds_are_pulled_to_a_common_unit(self):
        pair = comparable_prices(975.0, "GBp", 9.80, "GBP")
        assert pair is not None
        assert pair.left == pytest.approx(9.75)
        assert pair.right == pytest.approx(9.80)
        assert pair.left_scale == pytest.approx(0.01)

    def test_gbx_is_the_same_denomination_as_gbp_pence(self):
        assert comparable_prices(975.0, "GBX", 9.80, "GBP") is not None

    def test_major_unit_pairs_pass_through_untouched(self):
        pair = comparable_prices(100.0, "USD", 110.0, "USD")
        assert pair is not None
        assert (pair.left, pair.right) == (100.0, 110.0)
        assert pair.left_scale == 1.0

    @pytest.mark.parametrize(
        ("left_ccy", "right_ccy", "why"),
        [
            ("GBP", "JPY", "different economies"),
            (None, "GBP", "unlabelled left"),
            ("GBP", None, "unlabelled right"),
            ("", "GBP", "empty left"),
        ],
    )
    def test_incomparable_pairs_fail_closed(self, left_ccy, right_ccy, why):
        """Converting one side of a mismatched pair invents a 100x ratio."""
        assert comparable_prices(100.0, left_ccy, 110.0, right_ccy) is None, why

    @pytest.mark.parametrize("bad", [0.0, -1.0, None])
    def test_non_positive_prices_carry_no_scale(self, bad):
        assert comparable_prices(bad, "USD", 110.0, "USD") is None
        assert comparable_prices(100.0, "USD", bad, "USD") is None

    def test_case_sensitivity_is_preserved(self):
        """'GBp' and 'GBP' differ only in case and by a factor of 100."""
        pence = comparable_prices(975.0, "GBp", 975.0, "GBp")
        pounds = comparable_prices(975.0, "GBP", 975.0, "GBP")
        assert pence is not None and pounds is not None
        assert pence.left_scale == pytest.approx(0.01)
        assert pounds.left_scale == 1.0


class TestTheTwoMechanismsAreComplementary:
    """Neither check subsumes the other; the repo needs both.

    This is the architectural claim the whole change rests on, and it is the
    one an external review got wrong in both directions — first proposing a
    code-only helper (which passes GAMA.L), then treating the scale check as
    though it made the code check redundant.
    """

    def test_code_agreement_does_not_catch_a_scale_error(self):
        """GAMA.L: both sides say GBP, and it is still wrong by 100x."""
        pair = comparable_prices(900.0, "GBP", GAMA_PRICE, "GBP")
        assert pair is not None, "codes agree, so this check is satisfied"
        assert pair.left == 900.0 and pair.right == GAMA_PRICE

        verdict = assess_price_level_coherence(GAMA_PRICE, [900.0])
        assert verdict.is_incoherent, "only the scale check catches it"

    def test_scale_agreement_does_not_catch_a_denomination_error(self):
        """A GBp position vs a GBP analysis: consistent scale, wrong unit."""
        verdict = assess_price_level_coherence(9.80, [9.75])
        assert verdict.status is PriceLevelCoherence.COHERENT, (
            "levels agree with their own reference price"
        )

        pair = comparable_prices(9.75, "GBp", 9.80, "GBP")
        assert pair is not None
        assert pair.left == pytest.approx(0.0975), "only the code check rescales it"


class TestTheDivergenceObserverCannotBreakTheArtifact:
    """An observation-only check must never fail the run it observes.

    `inf` is the trap and it reached production code: it passes every naive
    guard (`not inf` is False, `inf > 0` is True), makes `reported / inf`
    exactly 0.0, and then raises ZeroDivisionError on the reciprocal inside the
    minor-unit hint — turning a log line into a failed fundamentals artifact.
    `nan` did not raise but logged `ratio=nan`, which is noise asserting nothing.
    """

    BLOCK = "CURRENT_PRICE: 9.76\nPRICE_CURRENCY: GBP\n"

    @pytest.mark.parametrize(
        "quote",
        [
            pytest.param(float("inf"), id="inf"),
            pytest.param(float("-inf"), id="-inf"),
            pytest.param(float("nan"), id="nan"),
            pytest.param(0.0, id="zero"),
            pytest.param(-5.0, id="negative"),
            pytest.param("9.76", id="string"),
            pytest.param(True, id="bool"),
            pytest.param(None, id="none"),
        ],
    )
    def test_no_malformed_quote_raises(self, quote):
        stamped = stamp_price_currency(
            self.BLOCK, {"currency": "GBP", "currentPrice": quote}
        )
        assert "PRICE_CURRENCY: GBP" in stamped

    def test_an_unusable_current_quote_falls_back_to_the_regular_quote(self, caplog):
        """`a or b` would accept an inf currentPrice and never reach the fallback.

        Asserting only that stamping succeeded is too weak: a version that
        bailed out entirely on the inf would pass. The emitted fields are what
        prove the *second* quote was actually selected and compared.
        """
        import logging

        with caplog.at_level(logging.WARNING):
            stamped = stamp_price_currency(
                self.BLOCK,
                {
                    "currency": "GBP",
                    "currentPrice": float("inf"),
                    "regularMarketPrice": 0.0976,
                },
            )
        assert "PRICE_CURRENCY: GBP" in stamped
        assert "data_block_price_diverges_from_payload" in caplog.text
        assert "payload_price=0.0976" in caplog.text, (
            "the fallback quote must be the one compared, not the inf"
        )
        assert "ratio=100.0" in caplog.text
        assert "near_minor_unit=True" in caplog.text

    def test_a_real_divergence_is_still_observed(self, caplog):
        """The guards must not have silenced the thing this exists to measure."""
        import logging

        with caplog.at_level(logging.WARNING):
            stamp_price_currency(
                self.BLOCK, {"currency": "GBP", "currentPrice": 0.0976}
            )
        assert "data_block_price_diverges_from_payload" in caplog.text

    def test_an_agreeing_quote_is_silent(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            stamp_price_currency(self.BLOCK, {"currency": "GBP", "currentPrice": 9.75})
        assert "data_block_price_diverges_from_payload" not in caplog.text
