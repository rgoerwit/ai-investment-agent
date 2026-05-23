"""Tests for the scenario overlay on the football-field chart (Tranche 4, Step 9)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.charts.base import FootballFieldData
from src.charts.generators.football_field import (
    _add_scenario_row_label,
    _collect_visible_rows,
    _overlay_scenarios,
)


def _scenarios(
    *,
    bear_iv: float = 80.0,
    base_iv: float = 120.0,
    bull_iv: float = 160.0,
    weighted_iv: float = 116.0,
    bear_prob: float = 30.0,
    base_prob: float = 50.0,
    bull_prob: float = 20.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        bear_iv=bear_iv,
        base_iv=base_iv,
        bull_iv=bull_iv,
        weighted_iv=weighted_iv,
        bear=SimpleNamespace(probability=bear_prob),
        base=SimpleNamespace(probability=base_prob),
        bull=SimpleNamespace(probability=bull_prob),
    )


def _data(scenarios=None, **kwargs) -> FootballFieldData:
    return FootballFieldData(
        ticker="TEST",
        trade_date="2026-05-20",
        current_price=kwargs.pop("current_price", 100.0),
        fifty_two_week_high=kwargs.pop("fifty_two_week_high", 150.0),
        fifty_two_week_low=kwargs.pop("fifty_two_week_low", 70.0),
        scenarios=scenarios,
        **kwargs,
    )


def _mock_ax() -> MagicMock:
    ax = MagicMock()
    ax.get_xlim.return_value = (50.0, 200.0)
    return ax


def test_overlay_returns_none_when_scenarios_absent() -> None:
    ax = _mock_ax()
    assert _overlay_scenarios(ax, _data()) is None
    ax.scatter.assert_not_called()


def test_overlay_returns_none_when_scenarios_malformed() -> None:
    """Missing attributes → scenario overlay silently skipped, no exception."""
    ax = _mock_ax()
    bad = SimpleNamespace(bear_iv=80.0)  # missing base_iv, bull_iv, etc.
    assert _overlay_scenarios(ax, _data(scenarios=bad)) is None
    ax.scatter.assert_not_called()


def test_overlay_returns_none_when_iv_nonpositive() -> None:
    ax = _mock_ax()
    bad = _scenarios(bear_iv=-5.0)
    assert _overlay_scenarios(ax, _data(scenarios=bad)) is None
    ax.scatter.assert_not_called()


def test_overlay_happy_path_plots_four_markers() -> None:
    ax = _mock_ax()
    y = _overlay_scenarios(ax, _data(scenarios=_scenarios()))
    assert y is not None
    # Four scatter calls: bear, base, bull, weighted.
    assert ax.scatter.call_count == 4

    labels = [call.kwargs.get("label", "") for call in ax.scatter.call_args_list]
    assert labels == ["Bear", "Base", "Bull", "Weighted IV"]


def test_overlay_uses_compact_labels_without_probabilities() -> None:
    ax = _mock_ax()
    _overlay_scenarios(ax, _data(scenarios=_scenarios()))
    labels = [call.kwargs.get("label", "") for call in ax.scatter.call_args_list]
    assert not any("(" in lbl or "$" in lbl for lbl in labels)


def test_overlay_markers_placed_at_correct_iv_x_coordinates() -> None:
    ax = _mock_ax()
    _overlay_scenarios(ax, _data(scenarios=_scenarios()))
    x_values = [call.args[0][0] for call in ax.scatter.call_args_list]
    assert 80.0 in x_values
    assert 120.0 in x_values
    assert 160.0 in x_values
    assert 116.0 in x_values  # weighted


def test_overlay_weighted_marker_is_diamond() -> None:
    ax = _mock_ax()
    _overlay_scenarios(ax, _data(scenarios=_scenarios()))
    markers = [call.kwargs.get("marker") for call in ax.scatter.call_args_list]
    assert markers.count("D") == 1
    assert markers.count("o") == 3


def test_overlay_y_position_sits_above_bars() -> None:
    """Scenario row should sit above the topmost bar (52w / external / our)."""
    ax = _mock_ax()
    data = _data(
        scenarios=_scenarios(),
        external_target_low=90.0,
        external_target_high=110.0,
        our_target_low=95.0,
        our_target_high=125.0,
    )
    y = _overlay_scenarios(ax, data)
    assert y is not None
    visible_row_count = len(_collect_visible_rows(data))
    assert y >= visible_row_count


def test_scenario_row_label_uses_final_axis_limit() -> None:
    ax = _mock_ax()
    ax.get_xlim.return_value = (40.0, 180.0)
    _add_scenario_row_label(ax, 3.1, "black")
    ax.text.assert_called_once()
    assert ax.text.call_args.args[:3] == (40.0, 3.1, "Scenarios")


def test_scenario_row_label_skips_when_no_scenarios() -> None:
    ax = _mock_ax()
    _add_scenario_row_label(ax, None, "black")
    ax.text.assert_not_called()


def test_collect_visible_rows_counts_baseline_only() -> None:
    """No external/our targets → only the 52w row."""
    assert _collect_visible_rows(_data()) == ["52w"]


def test_collect_visible_rows_includes_external_when_reasonable() -> None:
    data = _data(external_target_low=85.0, external_target_high=115.0)
    rows = _collect_visible_rows(data)
    assert "ext" in rows


def test_collect_visible_rows_excludes_unreasonable_external_targets() -> None:
    """A target that's 100x the current price triggers `_is_target_reasonable=False`."""
    data = _data(external_target_low=10000.0, external_target_high=20000.0)
    rows = _collect_visible_rows(data)
    assert "ext" not in rows
