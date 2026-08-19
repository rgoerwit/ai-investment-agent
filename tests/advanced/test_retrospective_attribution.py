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
    FX_NOT_APPLICABLE,
    FX_OBSERVATIONS,
    FX_OBSERVED,
    FX_UNAVAILABLE,
    UNASSESSED_BENCHMARK,
    UNASSESSED_REASON_KEY,
    _render_attribution,
    attribute_return,
    compare_to_reality,
    run_retrospective,
)
from tests.advanced.retrospective_fakes import (
    FakeLessonsMemory,
    make_snapshot,
    yfinance_ticker_stub,
)

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

        # Reversed deliberately. This asserted that a usable benchmark *did*
        # reach the lesson LLM, which was the contract until the record became a
        # deterministic rendering. The outcome it exercises is real and still
        # produces a record — it just is not generated.
        mock_lesson.assert_not_awaited()
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

        def _capture(comparison, verdict):
            captured.append(comparison)
            return ("a lesson", "missed_risk", "OPERATIONAL_MISS")

        with (
            patch("src.retrospective.load_past_snapshots", return_value=snapshots),
            patch.object(
                yfinance,
                "Ticker",
                _yf_stub(stock=(100.0, 65.0), benchmark=(100.0, 101.0)),
            ),
            patch("src.retrospective.build_lesson_record", side_effect=_capture),
        ):
            await run_retrospective(
                "2767.T", Path("/fake"), memory, memo_path=tmp_path / "m.json"
            )

        attribution = captured[0]["attribution"]
        assert attribution["benchmark_available"] is True
        assert attribution["dominant_driver"] == DRIVER_RESIDUAL
        assert attribution["market_return_pct"] == pytest.approx(1.0, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# FX is tri-state: unavailable is not flat
# ══════════════════════════════════════════════════════════════════════════════


class TestFxObservationIsTriState:
    """`fx_delta_pct` was initialized to 0.0 and left there on three paths.

    No recorded decision-time rate, a failed live fetch, and a USD-denominated
    security all produced `+0.0%` — so "we could not tell" was reported to the
    lesson model as "the currency did not move". Only the third is a real zero.

    This is the same defect the comment above `benchmark_return_pct` describes as
    already fixed for the market leg; the FX sibling was missed. Measured
    2026-08-17: 6 of 7,952 snapshots are non-USD with no recorded rate, 987 are
    USD, and the live-fetch failure path is unbounded during a sweep.
    """

    def test_the_three_states_are_distinct_tokens(self):
        assert len({FX_OBSERVED, FX_NOT_APPLICABLE, FX_UNAVAILABLE}) == 3
        assert FX_OBSERVATIONS == {FX_OBSERVED, FX_NOT_APPLICABLE, FX_UNAVAILABLE}

    def test_a_missing_fx_leg_is_not_attributed_as_flat(self):
        """`attribute_return` must carry None through, not coerce it to zero."""
        attribution = attribute_return(
            price_return_pct=-30.0, benchmark_return_pct=-10.0, fx_delta_pct=None
        )
        assert attribution.fx_return_pct is None
        assert attribution.usd_investor_return_pct is None

    def test_an_observed_fx_leg_still_composes_multiplicatively(self):
        attribution = attribute_return(
            price_return_pct=10.0, benchmark_return_pct=4.0, fx_delta_pct=-5.0
        )
        assert attribution.fx_return_pct == pytest.approx(-5.0)
        # (1+usd) = (1+local) x (1+fx)
        assert attribution.usd_investor_return_pct == pytest.approx(
            ((1.10 * 0.95) - 1) * 100, abs=0.01
        )

    def test_an_undetermined_leg_renders_as_unknown_not_as_zero(self):
        """The prompt is the consumer that matters; "+0.0%" would be a claim."""
        rendered = _render_attribution(
            {
                "price_return_pct": -30.0,
                "attribution": attribute_return(
                    price_return_pct=-30.0,
                    benchmark_return_pct=-10.0,
                    fx_delta_pct=None,
                ).to_dict(),
                "fx_observation": FX_UNAVAILABLE,
            }
        )
        assert "unknown" in rendered
        assert FX_UNAVAILABLE in rendered, "the prompt must say why it is unknown"
        assert "FX +0.0%" not in rendered


class TestCompareToRealityReportsFxHonestly:
    """The integration the isolated tests above do NOT cover.

    Reverting the initializer to `0.0` left every test in this file green, which
    is the trap: `attribute_return` and `_render_attribution` were exercised
    directly while the site that *decides* the value was not. These drive
    `compare_to_reality` itself.
    """

    async def _compare(self, monkeypatch, *, currency, fx_rate, fx_result):
        # `compare_to_reality` does `import yfinance as yf` *inside* the fetch
        # function, so `src.retrospective.yf` is not a seam. Patching it — with
        # `raising=False` silencing the one signal that would have said so — left
        # these four tests reaching the live network, passing on real 7203.T data
        # and failing anywhere offline.
        import yfinance

        monkeypatch.setattr(
            yfinance,
            "Ticker",
            yfinance_ticker_stub(stock=(1000.0, 700.0), benchmark=(30000.0, 27000.0)),
        )

        async def _fx(*_a, **_k):
            if isinstance(fx_result, Exception):
                raise fx_result
            return fx_result

        monkeypatch.setattr("src.fx_normalization.get_fx_rate_yfinance", _fx)
        snapshot = make_snapshot(age_days=184)
        snapshot.update(
            {
                "ticker": "7203.T",
                "currency": currency,
                "fx_rate_to_usd": fx_rate,
                "benchmark_index": "^N225",
                "verdict": "BUY",
            }
        )
        return await compare_to_reality(snapshot)

    @pytest.mark.asyncio
    async def test_a_failed_fx_fetch_is_unavailable_not_flat(self, monkeypatch):
        comparison = await self._compare(
            monkeypatch,
            currency="JPY",
            fx_rate=0.0067,
            fx_result=RuntimeError("yfinance down"),
        )
        assert comparison is not None
        assert comparison["fx_observation"] == FX_UNAVAILABLE
        assert comparison["attribution"]["fx_return_pct"] is None, (
            "an FX outage must not be attributed as a flat currency"
        )

    @pytest.mark.asyncio
    async def test_a_missing_decision_time_rate_is_unavailable(self, monkeypatch):
        comparison = await self._compare(
            monkeypatch, currency="JPY", fx_rate=None, fx_result=0.0070
        )
        assert comparison is not None
        assert comparison["fx_observation"] == FX_UNAVAILABLE
        assert comparison["attribution"]["fx_return_pct"] is None

    @pytest.mark.asyncio
    async def test_a_usd_security_is_not_applicable_and_genuinely_zero(
        self, monkeypatch
    ):
        comparison = await self._compare(
            monkeypatch, currency="USD", fx_rate=1.0, fx_result=1.0
        )
        assert comparison is not None
        assert comparison["fx_observation"] == FX_NOT_APPLICABLE
        assert comparison["attribution"]["fx_return_pct"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_a_successful_fetch_is_observed(self, monkeypatch):
        comparison = await self._compare(
            monkeypatch, currency="JPY", fx_rate=0.0067, fx_result=0.0060
        )
        assert comparison is not None
        assert comparison["fx_observation"] == FX_OBSERVED
        assert comparison["attribution"]["fx_return_pct"] == pytest.approx(
            -10.45, abs=0.1
        )
