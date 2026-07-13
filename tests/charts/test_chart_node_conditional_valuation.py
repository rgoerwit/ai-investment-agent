from types import SimpleNamespace
from unittest.mock import patch

from src.charts.base import ChartConfig
from src.charts.chart_node import _generate_football_field

_VALUATION_PARAMS = (
    "### --- START VALUATION_PARAMS ---\n"
    "METHOD: P/E_NORMALIZATION\nCURRENT_PRICE: 100.00\nCONFIDENCE: HIGH\n"
    "CURRENT_PE: 10.0\nSECTOR_MEDIAN_PE: 15.0\n"
    "### --- END VALUATION_PARAMS ---\n\n"
    "### --- START VALUATION_SCENARIOS ---\n"
    "METHODOLOGY: P/E\nDATA_SUFFICIENCY: HIGH\n"
    "BEAR_MULTIPLE: 8\nBEAR_GROWTH_PCT: -5\nBEAR_MARGIN_DELTA_BPS: -200\n"
    "BEAR_DRIVERS: Bear.\nBEAR_PROBABILITY: 30\n"
    "BASE_MULTIPLE: 12\nBASE_GROWTH_PCT: 8\nBASE_MARGIN_DELTA_BPS: 0\n"
    "BASE_DRIVERS: Base.\nBASE_PROBABILITY: 50\n"
    "BULL_MULTIPLE: 16\nBULL_GROWTH_PCT: 15\nBULL_MARGIN_DELTA_BPS: 100\n"
    "BULL_DRIVERS: Bull.\nBULL_PROBABILITY: 20\n"
    "### --- END VALUATION_SCENARIOS ---\n"
)


def _data_block() -> SimpleNamespace:
    return SimpleNamespace(
        current_price=100.0,
        fifty_two_week_high=150.0,
        fifty_two_week_low=70.0,
        moving_avg_50=None,
        moving_avg_200=None,
        external_target_high=None,
        external_target_low=None,
        external_target_mean=None,
        analyst_coverage=None,
    )


def _pm_block() -> SimpleNamespace:
    return SimpleNamespace(
        valuation_discount=None,
        zone=None,
    )


def test_conditional_scenarios_suppress_chart_targets_and_overlay(tmp_path) -> None:
    state = {
        "valuation_params": _VALUATION_PARAMS,
        "fundamentals_report": (
            "#### CROSS-CHECK FLAGS\n"
            "- [CYCLICAL PEAK — LOW P/E MAY BE PEAK-DISTORTED]\n"
            "### --- START DATA_BLOCK ---\n"
            "CURRENT_PRICE: 100\n"
            "PE_RATIO_TTM: 10\n"
            "PE_RATIO_FORWARD: 8\n"
            "### --- END DATA_BLOCK ---"
        ),
    }
    captured = {}

    def _capture(data, _config):
        captured["data"] = data
        return tmp_path / "chart.png"

    with patch(
        "src.charts.generators.football_field.generate_football_field", _capture
    ):
        path = _generate_football_field(
            state=state,
            ticker="TEST",
            trade_date="2026-06-14",
            data_block=_data_block(),
            pm_block=_pm_block(),
            chart_config=ChartConfig(output_dir=tmp_path),
        )

    assert path is not None
    data = captured["data"]
    assert data.our_target_low is None
    assert data.our_target_high is None
    assert data.scenarios is None
    assert "Scenario valuation suppressed" in data.footnote


def test_normalized_forward_eps_keeps_chart_scenarios(tmp_path) -> None:
    state = {
        "valuation_params": _VALUATION_PARAMS,
        "fundamentals_report": (
            "#### CROSS-CHECK FLAGS\n"
            "- [NORMALIZE EARNINGS — RECURRING PROFIT LOWER THAN REPORTED]\n"
            "### --- START DATA_BLOCK ---\n"
            "CURRENT_PRICE: 100\n"
            "PE_RATIO_TTM: 10\n"
            "PE_RATIO_FORWARD: 20\n"
            "### --- END DATA_BLOCK ---"
        ),
    }
    captured = {}

    def _capture(data, _config):
        captured["data"] = data
        return tmp_path / "chart.png"

    with patch(
        "src.charts.generators.football_field.generate_football_field", _capture
    ):
        _generate_football_field(
            state=state,
            ticker="TEST",
            trade_date="2026-06-14",
            data_block=_data_block(),
            pm_block=_pm_block(),
            chart_config=ChartConfig(output_dir=tmp_path),
        )

    data = captured["data"]
    assert data.our_target_low is not None
    assert data.our_target_high is not None
    assert data.scenarios is not None


def test_thin_single_source_external_target_warning_prepended(tmp_path) -> None:
    """A degenerate point target wildly off price is suppressed as an anchor AND the
    reason is surfaced in quality_warnings (prepended so it survives the 2-warning cap)."""
    data_block = SimpleNamespace(
        current_price=447.40,
        fifty_two_week_high=602.0,
        fifty_two_week_low=447.40,
        moving_avg_50=None,
        moving_avg_200=None,
        external_target_high=1415.0,
        external_target_low=1415.0,
        external_target_mean=1415.0,
        analyst_coverage=None,
    )
    captured = {}

    def _capture(data, _config):
        captured["data"] = data
        return tmp_path / "chart.png"

    with patch(
        "src.charts.generators.football_field.generate_football_field", _capture
    ):
        _generate_football_field(
            state={},
            ticker="FRAGUAB.MX",
            trade_date="2026-06-21",
            data_block=data_block,
            pm_block=_pm_block(),
            chart_config=ChartConfig(output_dir=tmp_path),
        )

    data = captured["data"]
    assert data.has_external_targets() is False  # not used as a chart anchor
    assert data.quality_warnings is not None
    assert any("suppressed" in w for w in data.quality_warnings)


def test_thin_target_warning_survives_cap_when_red_flags_present(tmp_path) -> None:
    """With the 2-warning budget already full of red-flag warnings, the prepended
    suppression warning must survive the cap (and the list stays capped at 2)."""
    data_block = SimpleNamespace(
        current_price=447.40,
        fifty_two_week_high=602.0,
        fifty_two_week_low=447.40,
        moving_avg_50=None,
        moving_avg_200=None,
        external_target_high=1415.0,
        external_target_low=1415.0,
        external_target_mean=1415.0,
        analyst_coverage=None,
    )
    state = {
        "red_flags": [
            {"severity": "CRITICAL", "detail": "first red flag detail"},
            {"severity": "WARNING", "detail": "second red flag detail"},
        ]
    }
    captured = {}

    def _capture(data, _config):
        captured["data"] = data
        return tmp_path / "chart.png"

    with patch(
        "src.charts.generators.football_field.generate_football_field", _capture
    ):
        _generate_football_field(
            state=state,
            ticker="FRAGUAB.MX",
            trade_date="2026-06-21",
            data_block=data_block,
            pm_block=_pm_block(),
            chart_config=ChartConfig(output_dir=tmp_path),
        )

    warnings = captured["data"].quality_warnings
    assert len(warnings) == 2  # cap respected
    assert "suppressed" in warnings[0]  # the outlier warning survived at the front
