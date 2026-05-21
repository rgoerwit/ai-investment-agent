"""Layout regression guard for the football-field chart (Tranche 5, Step 11).

The chart now uses ``layout="constrained"`` instead of ``tight_layout()`` so
the six-row legend (52w / external / our + bear / base / bull / weighted IV)
doesn't clip on long ticker / company names. These tests render worst-case
inputs and check that the resulting image file is non-empty and the saved
PNG dimensions are within a sane bounded range.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

# Force a non-interactive backend before importing pyplot to keep CI happy.
import matplotlib  # noqa: E402
import pytest

matplotlib.use("Agg")

from src.charts.base import ChartConfig, ChartFormat, FootballFieldData  # noqa: E402
from src.charts.generators.football_field import generate_football_field  # noqa: E402


def _scenarios(weighted_iv: float = 116.0) -> SimpleNamespace:
    return SimpleNamespace(
        methodology="P/E",
        data_sufficiency="HIGH",
        bear_iv=80.0,
        base_iv=120.0,
        bull_iv=160.0,
        weighted_iv=weighted_iv,
        bear=SimpleNamespace(probability=30.0, drivers="Bear case driver text."),
        base=SimpleNamespace(probability=50.0, drivers="Base case driver text."),
        bull=SimpleNamespace(probability=20.0, drivers="Bull case driver text."),
    )


def _data(ticker: str = "TEST", **kwargs) -> FootballFieldData:
    return FootballFieldData(
        ticker=ticker,
        trade_date="2026-05-21",
        current_price=kwargs.pop("current_price", 100.0),
        fifty_two_week_high=kwargs.pop("fifty_two_week_high", 150.0),
        fifty_two_week_low=kwargs.pop("fifty_two_week_low", 70.0),
        external_target_high=kwargs.pop("external_target_high", 130.0),
        external_target_low=kwargs.pop("external_target_low", 85.0),
        our_target_high=kwargs.pop("our_target_high", 140.0),
        our_target_low=kwargs.pop("our_target_low", 95.0),
        target_methodology="P/E normalization",
        target_confidence="HIGH",
        scenarios=kwargs.pop("scenarios", _scenarios()),
        **kwargs,
    )


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "charts"
    d.mkdir()
    return d


def _config(out: Path) -> ChartConfig:
    return ChartConfig(output_dir=out, format=ChartFormat.PNG)


def test_chart_renders_with_short_label(out_dir: Path) -> None:
    """Sanity check: short ticker, all bars + scenarios renders to a file."""
    path = generate_football_field(_data(ticker="TEST"), _config(out_dir))
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 1024  # Non-trivial PNG.


def test_chart_renders_with_long_company_name(out_dir: Path) -> None:
    """Worst-case label: long company name + scenarios + all target rows.

    Pre-Step-11 the legend would clip; with ``layout="constrained"`` the
    image still renders to a non-empty PNG. We deliberately don't pixel-
    diff (font metrics drift across systems) — we assert the image file
    is non-trivially sized and within a sane bounded range.
    """
    long_ticker = "VERYLONGTICKERSYMBOL.HK"
    path = generate_football_field(_data(ticker=long_ticker), _config(out_dir))
    assert path is not None and path.exists()
    size = path.stat().st_size
    # Non-trivial PNG, but also not absurd (catches a runaway-image regression).
    assert 1024 < size < 1_500_000, f"unexpected png size {size}"


def test_chart_renders_without_scenarios(out_dir: Path) -> None:
    """Legacy single-range path: no scenarios should still render cleanly."""
    path = generate_football_field(_data(scenarios=None), _config(out_dir))
    assert path is not None and path.exists()
    assert path.stat().st_size > 1024


def test_chart_handles_extreme_weighted_iv_inside_envelope(out_dir: Path) -> None:
    """Weighted IV near the min (heavy bear weight) used to occasionally push the
    marker row's label off the visible axis. With constrained layout it stays in."""
    # 70 sits below the bear IV (80) to stress the x-axis lower bound.
    scenarios = _scenarios(weighted_iv=70.0)
    path = generate_football_field(_data(scenarios=scenarios), _config(out_dir))
    assert path is not None and path.exists()
    assert path.stat().st_size > 1024


def test_chart_uses_constrained_layout_not_tight_layout() -> None:
    """Source-level guard: the renderer should not call ``fig.tight_layout()``.

    Mixing ``layout="constrained"`` with ``tight_layout()`` raises a
    matplotlib UserWarning and disables the constrained layout — exactly the
    regression Step 11 is preventing.
    """
    import inspect

    from src.charts.generators import football_field as ff

    source = inspect.getsource(ff.generate_football_field)
    assert 'layout="constrained"' in source
    assert "fig.tight_layout()" not in source
