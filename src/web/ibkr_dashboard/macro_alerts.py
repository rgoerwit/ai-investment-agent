from __future__ import annotations

import re
import threading
from typing import Any

from src.ibkr.portfolio_health import CORRELATED_EVENT_EVIDENCE_PATTERN
from src.memory import create_macro_events_store

# Compiled from the emitter's own pattern so this reader cannot drift from the flag it
# parses. It previously hardcoded the "within Nd of DATE" phrasing and so matched only
# the `window` trigger -- `cumulative` and `drawdown_breadth` both say "as of DATE" and
# rendered an alert with no count, date, or percentage.
_CORRELATED_EVENT_RE = re.compile(CORRELATED_EVENT_EVIDENCE_PATTERN, re.IGNORECASE)


class MacroAlertService:
    def __init__(self, store_factory=create_macro_events_store) -> None:
        self._store_factory = store_factory
        self._store = None
        self._store_initialized = False
        self._store_init_failed = False
        self._lock = threading.Lock()

    def build_alert(self, health_flags: list[str]) -> dict[str, Any] | None:
        correlated_flag = next(
            (flag for flag in health_flags if "CORRELATED_SELL_EVENT" in flag),
            None,
        )
        if correlated_flag is None:
            return None

        match = _CORRELATED_EVENT_RE.search(correlated_flag)
        # `window` is absent by design on the cumulative / drawdown_breadth phrasings --
        # those carry no lookback window -- so it stays None even on a successful match.
        window = match.group("window") if match else None
        payload: dict[str, Any] = {
            "detected": True,
            "flag": correlated_flag,
            "peak_count": int(match.group("count")) if match else None,
            "window_days": int(window) if window else None,
            "event_date": match.group("date") if match else None,
            "correlation_pct": int(match.group("pct")) if match else None,
            "event_type": None,
            "impact": None,
            "headline": None,
        }

        event = self._get_active_event()
        if event is None:
            return payload

        payload.update(
            {
                "event_type": event.event_type,
                "impact": event.impact,
                "headline": event.news_headline,
            }
        )
        return payload

    def _get_active_event(self):
        store = self._get_store()
        if store is None or not store.available:
            return None
        try:
            return next(iter(store.get_active_events()), None)
        except Exception:
            return None

    def _get_store(self):
        with self._lock:
            if not self._store_initialized:
                self._store_initialized = True
                try:
                    self._store = self._store_factory()
                except Exception:
                    self._store = None
                    self._store_init_failed = True
            if self._store_init_failed:
                return None
            return self._store
