"""Step 4: two honest views of an outcome, and an outcome that cannot be judged.

The lesson that motivated this is real. ``001060.KS``, DO_NOT_INITIATE on
2026-06-16: the KOSPI fell 20.04%, the stock rose 5.87%, excess +25.91% — scored
"wrong", and the stored lesson told future runs that defensive stocks can be
attractive "even if they appear technically overextended by standard valuation
metrics". A rule to relax valuation discipline, induced from a benchmark crash.

Two structural corrections:

* **Market and residual are the only additive legs.** ``excess = price -
  benchmark`` by construction, so a third "FX leg" double-counts. FX combines
  *multiplicatively* and answers a different question — what the USD investor
  earned — which is why it is reported but never competes for dominance.
* **A missing benchmark makes the outcome unassessable, not flat.** Reporting it
  as 0.0% made ``excess`` equal the raw stock return, so a stock down 35% in a
  market down 30% read as a company-specific collapse.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.retrospective import (
    ATTRIBUTION_DOMINANCE_RATIO,
    DRIVER_MARKET,
    DRIVER_MIXED,
    DRIVER_RESIDUAL,
    DRIVER_UNKNOWN,
    UNASSESSED_BENCHMARK,
    UNASSESSED_REASON_KEY,
    attribute_return,
    run_retrospective,
)
from tests.advanced.retrospective_fakes import FakeLessonsMemory, make_snapshot

# ══════════════════════════════════════════════════════════════════════════════
# Accounting identities
# ══════════════════════════════════════════════════════════════════════════════


class TestLocalViewIsAdditive:
    @pytest.mark.parametrize(
        ("price", "benchmark"),
        [
            (10.0, 4.0),
            (-35.0, -3.0),
            (5.87, -20.04),
            (0.0, 0.0),
            (120.0, -45.5),
            (-0.01, 0.02),
        ],
    )
    def test_market_plus_residual_equals_the_stock_return(self, price, benchmark):
        attribution = attribute_return(
            price_return_pct=price, benchmark_return_pct=benchmark, fx_delta_pct=0.0
        )
        assert attribution.market_return_pct is not None
        assert attribution.residual_return_pct is not None
        assert math.isclose(
            attribution.market_return_pct + attribution.residual_return_pct,
            price,
            abs_tol=0.01,
        )

    def test_residual_is_the_excess_return(self):
        attribution = attribute_return(
            price_return_pct=5.87, benchmark_return_pct=-20.04, fx_delta_pct=0.0
        )
        assert attribution.residual_return_pct == pytest.approx(25.91, abs=0.01)


class TestUsdViewIsMultiplicative:
    def test_identity_holds_for_a_non_usd_position(self):
        attribution = attribute_return(
            price_return_pct=10.0, benchmark_return_pct=0.0, fx_delta_pct=-10.0
        )
        expected = ((1.10) * (0.90) - 1.0) * 100.0
        assert attribution.usd_investor_return_pct == pytest.approx(expected, abs=0.01)

    def test_it_is_not_the_additive_answer(self):
        """A 10% gain against a 10% currency fall is not flat."""
        attribution = attribute_return(
            price_return_pct=10.0, benchmark_return_pct=0.0, fx_delta_pct=-10.0
        )
        assert attribution.usd_investor_return_pct != pytest.approx(0.0, abs=0.1)
        assert attribution.usd_investor_return_pct == pytest.approx(-1.0, abs=0.01)

    def test_a_usd_position_degenerates_cleanly(self):
        attribution = attribute_return(
            price_return_pct=12.0, benchmark_return_pct=3.0, fx_delta_pct=0.0
        )
        assert attribution.fx_return_pct == 0.0
        assert attribution.usd_investor_return_pct == pytest.approx(12.0, abs=0.01)

    def test_an_unknown_fx_rate_yields_no_usd_view(self):
        attribution = attribute_return(
            price_return_pct=12.0, benchmark_return_pct=3.0, fx_delta_pct=None
        )
        assert attribution.fx_return_pct is None
        assert attribution.usd_investor_return_pct is None


class TestFxNeverCompetesForDominance:
    def test_a_large_fx_move_does_not_become_the_driver(self):
        """FX explains none of the local excess — that is the double-count fixed."""
        attribution = attribute_return(
            price_return_pct=2.0, benchmark_return_pct=1.0, fx_delta_pct=-40.0
        )
        assert attribution.dominant_driver in {DRIVER_MARKET, DRIVER_MIXED}
        assert attribution.fx_return_pct == -40.0

    def test_fx_is_absent_from_the_driver_vocabulary(self):
        assert "FX" not in {
            DRIVER_MARKET,
            DRIVER_RESIDUAL,
            DRIVER_MIXED,
            DRIVER_UNKNOWN,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Dominance
# ══════════════════════════════════════════════════════════════════════════════


class TestDominance:
    def test_market_dominates_a_broad_selloff(self):
        attribution = attribute_return(
            price_return_pct=-31.0, benchmark_return_pct=-30.0, fx_delta_pct=0.0
        )
        assert attribution.dominant_driver == DRIVER_MARKET

    def test_residual_dominates_a_company_specific_move(self):
        attribution = attribute_return(
            price_return_pct=-35.0, benchmark_return_pct=-3.0, fx_delta_pct=0.0
        )
        assert attribution.dominant_driver == DRIVER_RESIDUAL

    def test_the_001060_ks_case_is_mixed_not_residual(self):
        """The acceptance case: the harmful lesson must be scoped, not generalized."""
        attribution = attribute_return(
            price_return_pct=5.87, benchmark_return_pct=-20.04, fx_delta_pct=-5.57
        )
        # 25.91 / 20.04 = 1.29, under the 1.5 ratio.
        assert attribution.dominant_driver == DRIVER_MIXED

    def test_comparable_legs_are_mixed(self):
        attribution = attribute_return(
            price_return_pct=20.0, benchmark_return_pct=10.0, fx_delta_pct=0.0
        )
        assert attribution.dominant_driver == DRIVER_MIXED

    def test_exactly_at_the_ratio_is_mixed_not_dominant(self):
        """Strict inequality: the boundary belongs to the humbler answer."""
        market = 10.0
        price = market + market * ATTRIBUTION_DOMINANCE_RATIO
        attribution = attribute_return(
            price_return_pct=price, benchmark_return_pct=market, fx_delta_pct=0.0
        )
        assert attribution.residual_return_pct == pytest.approx(
            market * ATTRIBUTION_DOMINANCE_RATIO
        )
        assert attribution.dominant_driver == DRIVER_MIXED

    def test_just_past_the_ratio_is_dominant(self):
        market = 10.0
        price = market + market * ATTRIBUTION_DOMINANCE_RATIO + 0.1
        attribution = attribute_return(
            price_return_pct=price, benchmark_return_pct=market, fx_delta_pct=0.0
        )
        assert attribution.dominant_driver == DRIVER_RESIDUAL

    def test_a_flat_outcome_is_mixed(self):
        attribution = attribute_return(
            price_return_pct=0.0, benchmark_return_pct=0.0, fx_delta_pct=0.0
        )
        assert attribution.dominant_driver == DRIVER_MIXED

    def test_a_pure_market_move_with_no_residual_is_market(self):
        attribution = attribute_return(
            price_return_pct=-20.0, benchmark_return_pct=-20.0, fx_delta_pct=0.0
        )
        assert attribution.residual_return_pct == pytest.approx(0.0)
        assert attribution.dominant_driver == DRIVER_MARKET


# ══════════════════════════════════════════════════════════════════════════════
# The unassessable case
# ══════════════════════════════════════════════════════════════════════════════


class TestMissingBenchmark:
    def test_the_market_leg_is_none_never_zero(self):
        attribution = attribute_return(
            price_return_pct=-35.0, benchmark_return_pct=None, fx_delta_pct=0.0
        )
        assert attribution.market_return_pct is None
        assert attribution.residual_return_pct is None
        assert attribution.benchmark_available is False

    def test_the_driver_is_unknown_not_residual(self):
        """The exact overclaim: a market crash read as a company collapse."""
        attribution = attribute_return(
            price_return_pct=-35.0, benchmark_return_pct=None, fx_delta_pct=0.0
        )
        assert attribution.dominant_driver == DRIVER_UNKNOWN
        assert attribution.dominant_driver != DRIVER_RESIDUAL

    def test_the_usd_view_survives_a_missing_benchmark(self):
        """FX and the index are independent failures."""
        attribution = attribute_return(
            price_return_pct=10.0, benchmark_return_pct=None, fx_delta_pct=-10.0
        )
        assert attribution.usd_investor_return_pct == pytest.approx(-1.0, abs=0.01)


def _yf_stub(*, stock=(100.0, 65.0), benchmark=(100.0, 70.0) or None):
    """A yfinance double whose benchmark can be withheld."""
    import pandas as pd

    class _T:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **_):
            series = stock if not self.symbol.startswith("^") else benchmark
            if series is None:
                return pd.DataFrame({"Close": []})
            return pd.DataFrame({"Close": list(series)})

        @property
        def info(self):
            return {}

    return _T


class TestUnassessableOutcomesNeverBecomeLessons:
    @pytest.mark.asyncio
    async def test_a_crash_with_no_benchmark_generates_nothing(self, tmp_path):
        """Assert on the *cost*: zero LLM calls, zero writes, for a -35% stock."""
        import yfinance

        memory = FakeLessonsMemory()
        snapshots = {
            "2767.T": [make_snapshot(age_days=180, analysis_id="run-a", verdict="BUY")]
        }

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch.object(
                yfinance, "Ticker", _yf_stub(stock=(100.0, 65.0), benchmark=None)
            ),
            patch(
                "src.retrospective.generate_lesson", new_callable=AsyncMock
            ) as mock_lesson,
        ):
            lessons = await run_retrospective(
                "2767.T",
                Path("/fake"),
                memory,
                memo_path=tmp_path / "m.json",
            )

        mock_lesson.assert_not_awaited()
        assert lessons == []
        assert memory.add_calls == 0

    @pytest.mark.asyncio
    async def test_the_same_crash_with_a_benchmark_does_generate(self, tmp_path):
        """The control: only the missing index suppresses the lesson."""
        import yfinance

        memory = FakeLessonsMemory()
        snapshots = {
            "2767.T": [make_snapshot(age_days=180, analysis_id="run-a", verdict="BUY")]
        }

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch.object(
                yfinance,
                "Ticker",
                _yf_stub(stock=(100.0, 65.0), benchmark=(100.0, 101.0)),
            ),
            patch(
                "src.retrospective.generate_lesson",
                new_callable=AsyncMock,
                return_value=("a lesson", "missed_risk", "OPERATIONAL_MISS"),
            ) as mock_lesson,
        ):
            lessons = await run_retrospective(
                "2767.T",
                Path("/fake"),
                memory,
                memo_path=tmp_path / "m.json",
            )

        mock_lesson.assert_awaited_once()
        assert len(lessons) == 1

    @pytest.mark.asyncio
    async def test_it_is_counted_separately_from_below_threshold(self, tmp_path):
        memory = FakeLessonsMemory()
        snapshots = {"2767.T": [make_snapshot(age_days=180, analysis_id="run-a")]}
        seen = []

        unassessed = {
            **snapshots["2767.T"][0],
            UNASSESSED_REASON_KEY: UNASSESSED_BENCHMARK,
        }

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch(
                "src.retrospective.compare_to_reality",
                new_callable=AsyncMock,
                return_value=unassessed,
            ),
        ):
            await run_retrospective(
                "2767.T",
                Path("/fake"),
                memory,
                memo_path=tmp_path / "m.json",
                on_summary=seen.append,
            )

        assert seen[-1].unassessed_benchmark == 1
        assert seen[-1].triggered == 0
        assert seen[-1].reconciles is True

    @pytest.mark.asyncio
    async def test_it_is_retried_on_the_next_run_not_memoized_for_30_days(
        self, tmp_path
    ):
        """A failed index fetch is an outage, not a verdict."""
        memory = FakeLessonsMemory()
        snapshots = {"2767.T": [make_snapshot(age_days=180, analysis_id="run-a")]}
        memo_path = tmp_path / "m.json"
        unassessed = {
            **snapshots["2767.T"][0],
            UNASSESSED_REASON_KEY: UNASSESSED_BENCHMARK,
        }

        for _ in range(2):
            with (
                patch("src.retrospective.load_past_snapshots", return_value=snapshots),
                patch(
                    "src.retrospective.compare_to_reality",
                    new_callable=AsyncMock,
                    return_value=unassessed,
                ) as mock_compare,
            ):
                await run_retrospective(
                    "2767.T", Path("/fake"), memory, memo_path=memo_path
                )
            mock_compare.assert_awaited_once()


class TestAttributionReachesTheComparison:
    @pytest.mark.asyncio
    async def test_a_triggered_comparison_carries_the_attribution(self, tmp_path):
        import yfinance

        memory = FakeLessonsMemory()
        snapshots = {
            "2767.T": [make_snapshot(age_days=180, analysis_id="run-a", verdict="BUY")]
        }
        captured: list[dict] = []

        async def _capture(comparison):
            captured.append(comparison)
            return ("a lesson", "missed_risk", "OPERATIONAL_MISS")

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch.object(
                yfinance,
                "Ticker",
                _yf_stub(stock=(100.0, 65.0), benchmark=(100.0, 101.0)),
            ),
            patch("src.retrospective.generate_lesson", side_effect=_capture),
        ):
            await run_retrospective(
                "2767.T", Path("/fake"), memory, memo_path=tmp_path / "m.json"
            )

        attribution = captured[0]["attribution"]
        assert attribution["benchmark_available"] is True
        assert attribution["dominant_driver"] == DRIVER_RESIDUAL
        assert attribution["market_return_pct"] == pytest.approx(1.0, abs=0.01)
