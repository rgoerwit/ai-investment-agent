from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.web.ibkr_dashboard.macro_alerts import MacroAlertService


class _DummyStore:
    def __init__(self, events, *, available: bool = True) -> None:
        self._events = events
        self.available = available

    def get_active_events(self):
        return list(self._events)


def test_build_alert_merges_active_event_metadata():
    event = SimpleNamespace(
        event_type="RISK_OFF",
        impact="CYCLICAL",
        news_headline="Rates shock",
    )
    service = MacroAlertService(lambda: _DummyStore([event]))
    payload = service.build_alert(
        [
            "CORRELATED_SELL_EVENT: 8 positions changed verdict within 7d of 2026-03-20"
            " (45% of held positions) — probable macro event."
        ]
    )
    assert payload["detected"] is True
    assert payload["correlation_pct"] == 45
    assert payload["event_type"] == "RISK_OFF"
    assert payload["headline"] == "Rates shock"


class TestEveryTriggerPhrasingParses:
    """The dashboard must read all three phrasings `portfolio_health` emits.

    Regression for the Aug 2026 defect: this reader hardcoded the ``within Nd of``
    wording, so the ``cumulative`` and ``drawdown_breadth`` triggers -- which say
    ``as of DATE`` and carry no lookback window -- produced an alert with
    ``peak_count``/``event_date``/``correlation_pct`` all None. It failed silently and
    stayed publishable; only the `window` wording was ever tested.

    Phrasings mirror ``compute_portfolio_health``; the end-to-end contract against
    genuinely emitted flags lives in
    ``tests/ibkr/test_macro_detection.py::test_every_consumer_reads_a_production_generated_flag``.
    """

    WINDOW = (
        "CORRELATED_SELL_EVENT: 6 positions changed verdict within 14d of 2026-08-01"
        " (30% of held positions) — probable macro event [window]."
    )
    CUMULATIVE = (
        "CORRELATED_SELL_EVENT: 9 positions changed verdict across the held book"
        " as of 2026-08-01 (40% of held positions) — probable macro event [cumulative]."
    )
    DRAWDOWN = (
        "CORRELATED_SELL_EVENT: 9 positions currently trading ≥10% below entry"
        " as of 2026-08-01 (40% of held positions)"
        " — probable macro event [drawdown_breadth]."
    )

    @pytest.mark.parametrize(
        ("flag", "expected_count", "expected_pct", "expected_window"),
        [
            (WINDOW, 6, 30, 14),
            (CUMULATIVE, 9, 40, None),
            (DRAWDOWN, 9, 40, None),
        ],
        ids=["window", "cumulative", "drawdown_breadth"],
    )
    def test_detail_survives_every_phrasing(
        self, flag, expected_count, expected_pct, expected_window
    ):
        payload = MacroAlertService(lambda: _DummyStore([])).build_alert([flag])
        assert payload is not None
        assert payload["peak_count"] == expected_count
        assert payload["event_date"] == "2026-08-01"
        assert payload["correlation_pct"] == expected_pct
        # `window_days` is legitimately absent on the two "as of" phrasings; it must be
        # None rather than raising on int(None).
        assert payload["window_days"] == expected_window

    def test_unparseable_flag_still_yields_a_detected_alert(self):
        """A shape change must degrade to a detail-free alert, never crash."""
        payload = MacroAlertService(lambda: _DummyStore([])).build_alert(
            ["CORRELATED_SELL_EVENT: something entirely unexpected"]
        )
        assert payload is not None
        assert payload["detected"] is True
        assert payload["peak_count"] is None
        assert payload["window_days"] is None


def test_build_alert_caches_store_instance():
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return _DummyStore([])

    service = MacroAlertService(factory)
    service.build_alert(
        [
            "CORRELATED_SELL_EVENT: 8 positions changed verdict within 7d of 2026-03-20"
            " (45% of held positions) — probable macro event."
        ]
    )
    service.build_alert(
        [
            "CORRELATED_SELL_EVENT: 8 positions changed verdict within 7d of 2026-03-20"
            " (45% of held positions) — probable macro event."
        ]
    )
    assert calls == 1
