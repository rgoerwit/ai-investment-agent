from __future__ import annotations

from types import SimpleNamespace

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
