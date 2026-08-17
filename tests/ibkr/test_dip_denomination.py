"""Every analysis-price vs live-price calculation in dip scoring converts by code.

`dip_pct` was fixed first and the upside arithmetic three lines below it was
missed — the same comparison, the same function, uncorrected. This is the
cross-denomination case, which the scale check deliberately does *not* cover:
a perfectly coherent GBP analysis against a GBp broker quote. Codes differ,
scales are each internally fine, and the product is a ~100x phantom upside that
awards the full bonus and pushes a name over the ★★★ / concentration bar.
"""

from __future__ import annotations

import pytest

from src.ibkr.dip_watch import (
    DIP_CONCENTRATION_MIN_SCORE,
    build_dip_watch_candidates,
    compute_dip_score,
    dip_pct,
)
from src.ibkr.models import ReconciliationItem
from tests.factories.ibkr import make_analysis, make_position


def _item(*, analysis_currency: str, position_currency: str) -> ReconciliationItem:
    """A held BUY whose analysis and broker quote may disagree on denomination.

    Numbers chosen so the *same economic reality* is described on both sides
    when the codes agree: 146.5p entry, 129.87p now, 175p target.
    """
    analysis = make_analysis(
        ticker="MEGP.L",
        verdict="BUY",
        currency=analysis_currency,
        entry_price=146.5,
        stop_price=125.0,
        target_1=175.0,
        target_2=200.0,
        current_price=146.5,
    )
    analysis.health_adj = 92.0
    analysis.growth_adj = 67.0
    return ReconciliationItem(
        ticker="MEGP.L",
        action="HOLD",
        urgency="LOW",
        reason="No action",
        ibkr_position=make_position(
            ticker="MEGP.L",
            current_price=129.87,
            avg_cost=138.75,
            currency=position_currency,
        ),
        analysis=analysis,
    )


class TestSameDenominationIsUnchanged:
    def test_dip_and_score_behave_normally(self):
        item = _item(analysis_currency="GBX", position_currency="GBX")
        assert dip_pct(item) == pytest.approx(11.35, abs=0.05)
        score = compute_dip_score(item)
        # base 63.6 + price bonus + a real upside bonus on a 175p target
        assert 60.0 < score < 100.0


class TestCrossDenominationProducesNoPhantomSignal:
    """GBp analysis against a GBP quote: same economy, 100x apart."""

    def test_equivalent_money_in_either_denomination_scores_identically(self):
        """The assertion that actually proves conversion happened.

        A pence-quoted position and a pounds-quoted position describing the
        SAME money must produce the same dip, upside and score. Asserting only
        that the score is finite and non-negative — as an earlier version of
        this test did — would pass against raw arithmetic, i.e. against the
        exact regression it exists to catch.
        """
        pence = _item(analysis_currency="GBX", position_currency="GBX")
        # Same money, pounds on both sides: 1.465 entry, 1.2987 now, 1.75 target.
        pounds = _item(analysis_currency="GBP", position_currency="GBP")
        pounds.analysis.entry_price = 1.465
        pounds.analysis.target_1_price = 1.75
        pounds.analysis.current_price = 1.465
        pounds.ibkr_position.current_price_local = 1.2987

        assert dip_pct(pounds) == pytest.approx(dip_pct(pence), abs=0.01)
        assert compute_dip_score(pounds) == pytest.approx(
            compute_dip_score(pence), abs=0.01
        )

    def test_a_mixed_pair_converges_on_the_same_answer(self):
        """GBp analysis vs GBP quote, describing the same money, must agree."""
        matched = _item(analysis_currency="GBX", position_currency="GBX")
        mixed = _item(analysis_currency="GBX", position_currency="GBP")
        # 129.87 pence == 1.2987 pounds: identical money, different label.
        mixed.ibkr_position.current_price_local = 1.2987

        assert dip_pct(mixed) == pytest.approx(dip_pct(matched), abs=0.01)
        assert compute_dip_score(mixed) == pytest.approx(
            compute_dip_score(matched), abs=0.01
        )

    def test_raw_arithmetic_would_not_produce_these_numbers(self):
        """Pins the magnitude, so a revert to raw subtraction cannot pass.

        Raw: (146.5 - 1.2987)/146.5 = 99.1% "dip". Converted: 11.35%.
        """
        mixed = _item(analysis_currency="GBX", position_currency="GBP")
        mixed.ibkr_position.current_price_local = 1.2987
        assert dip_pct(mixed) == pytest.approx(11.35, abs=0.05)
        assert dip_pct(mixed) < 50.0, "a converted pair cannot read as a ~99% dip"

    def test_an_unrelated_economy_yields_no_dip_and_no_upside(self):
        """JPY analysis vs GBX position: not comparable, so no signal at all."""
        item = _item(analysis_currency="JPY", position_currency="GBX")
        assert dip_pct(item) == 0.0
        # Score collapses to the health/growth base: no price bonus, no upside.
        base = (item.analysis.health_adj or 0) * 0.4 + (
            item.analysis.growth_adj or 0
        ) * 0.4
        assert compute_dip_score(item) == pytest.approx(base)

    def test_a_phantom_upside_cannot_push_a_name_over_the_star_bar(self):
        """The consequence that matters: threshold crossing on arithmetic alone."""
        item = _item(analysis_currency="JPY", position_currency="GBX")
        assert compute_dip_score(item) < DIP_CONCENTRATION_MIN_SCORE


class TestDisplayedUpsideObeysTheSameContract:
    def test_incomparable_currencies_render_no_upside(self):
        item = _item(analysis_currency="JPY", position_currency="GBX")
        candidates = build_dip_watch_candidates([item])
        for candidate in candidates:
            assert candidate.upside_pct is None, (
                "an unconverted pair must not render a phantom upside"
            )

    def test_matching_currencies_render_the_exact_upside(self):
        item = _item(analysis_currency="GBX", position_currency="GBX")
        candidates = build_dip_watch_candidates([item])
        assert candidates, "fixture must be dip-eligible or this asserts nothing"
        # (175 - 129.87) / 129.87 = 34.75%
        assert candidates[0].upside_pct == pytest.approx(34.8, abs=0.1)

    def test_a_mixed_pair_renders_the_same_upside_as_the_matched_pair(self):
        """`upside_pct` is implemented separately from the score's upside bonus.

        Pinning the score path does not pin this one, and this is the number the
        operator actually reads. Same money, different labels: 129.87 pence and
        1.2987 pounds must render one answer.
        """
        matched = _item(analysis_currency="GBX", position_currency="GBX")
        mixed = _item(analysis_currency="GBX", position_currency="GBP")
        mixed.ibkr_position.current_price_local = 1.2987

        matched_out = build_dip_watch_candidates([matched])
        mixed_out = build_dip_watch_candidates([mixed])
        assert matched_out and mixed_out, "both fixtures must be dip-eligible"
        assert mixed_out[0].upside_pct == pytest.approx(
            matched_out[0].upside_pct, abs=0.1
        )
        assert mixed_out[0].upside_pct == pytest.approx(34.8, abs=0.1)
        # Raw arithmetic would give (175 - 1.2987)/1.2987 = 13375%.
        assert mixed_out[0].upside_pct < 100.0


class TestRiskRewardRatioIsDeliberatelyUntouched:
    def test_it_has_no_production_callers(self):
        """Recorded, not fixed.

        `risk_reward_ratio` is stop-anchored and production-dead — CLAUDE.md
        records `DipWatchCandidate.risk_reward` as a permanently-None legacy
        field. Routing it through the currency contract would be churn on code
        nothing calls, so it is left as-is; this test exists so the claim stays
        true rather than being assumed.
        """
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parents[2]
        callers = []
        for path in list((root / "src").rglob("*.py")) + list(
            (root / "scripts").rglob("*.py")
        ):
            text = path.read_text()
            for match in re.finditer(r"\brisk_reward_ratio\s*\(", text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                if not text[line_start:].lstrip().startswith("def "):
                    callers.append(str(path.relative_to(root)))
        assert callers == [], (
            f"risk_reward_ratio gained callers and now needs the currency "
            f"contract: {sorted(set(callers))}"
        )
