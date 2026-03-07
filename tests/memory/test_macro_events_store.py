"""Tests for MacroEventsStore in src/memory.py."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.memory import MacroEvent, MacroEventsStore, create_macro_events_store


def _make_event(
    event_date="2026-03-05",
    impact="TRANSIENT",
    event_type="TARIFF_TRADE",
    scope="GLOBAL",
    primary_region="GLOBAL",
    expiry=None,
    forced_reanalysis=False,
) -> MacroEvent:
    if expiry is None:
        expiry = (date.fromisoformat(event_date) + timedelta(days=28)).isoformat()
    return MacroEvent(
        event_date=event_date,
        detected_date="2026-03-07",
        expiry=expiry,
        impact=impact,
        event_type=event_type,
        scope=scope,
        primary_region=primary_region,
        primary_sector="",
        severity="MEDIUM",
        correlation_pct=0.30,
        peak_count=6,
        total_held=20,
        news_headline="Test headline",
        news_detail="Test detail",
        forced_reanalysis=forced_reanalysis,
    )


class TestMacroEventsStoreInit:
    def test_init_with_chromadb_collection_error_sets_available_false(self):
        """ChromaDB raises on get_or_create_collection → available=False."""
        mock_client = MagicMock()
        mock_client.get_or_create_collection.side_effect = Exception("DB error")
        with patch("chromadb.PersistentClient", return_value=mock_client):
            store = MacroEventsStore()
        assert not store.available

    def test_create_macro_events_store_returns_store_instance(self):
        """create_macro_events_store() returns a MacroEventsStore."""
        with patch("chromadb.PersistentClient") as mock_client_cls:
            mock_client_cls.return_value = MagicMock()
            store = create_macro_events_store()
        assert isinstance(store, MacroEventsStore)

    def test_init_chromadb_exception_does_not_raise(self):
        """Any exception during _init() is caught; available stays False."""
        with patch("chromadb.PersistentClient", side_effect=RuntimeError("fail")):
            store = MacroEventsStore()
        assert not store.available


class TestMacroEventsStoreRoundTrip:
    def _mock_store(self, existing_ids=None):
        """Build a MacroEventsStore backed by a mock ChromaDB collection."""
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_col.get.return_value = {"ids": existing_ids or [], "metadatas": []}
        store = MacroEventsStore.__new__(MacroEventsStore)
        store.available = True
        store.collection = mock_col
        return store, mock_col

    def test_store_event_calls_collection_add(self):
        """store_event() calls collection.add() with correct id and metadata."""
        store, col = self._mock_store()
        event = _make_event()
        result = store.store_event(event)
        assert result is True
        col.add.assert_called_once()
        call_kwargs = col.add.call_args[1] if col.add.call_args[1] else {}
        ids_arg = call_kwargs.get(
            "ids", col.add.call_args[0][0] if col.add.call_args[0] else []
        )
        assert "2026-03-05" in ids_arg[0]

    def test_store_event_uses_dummy_embeddings(self):
        """store_event() passes [0.0]*768 as embedding (not Gemini API)."""
        store, col = self._mock_store()
        store.store_event(_make_event())
        kwargs = col.add.call_args[1]
        embedding = kwargs["embeddings"][0]
        assert len(embedding) == 768
        assert all(v == 0.0 for v in embedding)

    def test_store_event_dedup_within_7_days_returns_false(self):
        """A second event within 7-day window of existing → skipped, returns False."""
        store, col = self._mock_store(existing_ids=["macro_2026-03-05_2026-03-07"])
        event = _make_event(event_date="2026-03-06")  # 1 day later
        result = store.store_event(event)
        assert result is False
        col.add.assert_not_called()

    def test_store_event_beyond_7_days_returns_true(self):
        """An event 8+ days after the existing one → stored, returns True."""
        store, col = self._mock_store()
        # No existing events in the window for this date
        col.get.return_value = {"ids": [], "metadatas": []}
        event = _make_event(event_date="2026-03-14")
        result = store.store_event(event)
        assert result is True

    def test_store_event_unavailable_returns_false(self):
        """store_event() on unavailable store returns False without touching collection."""
        store = MacroEventsStore.__new__(MacroEventsStore)
        store.available = False
        store.collection = None
        assert store.store_event(_make_event()) is False

    def test_store_event_truncates_headline_to_120(self):
        """store_event() truncates news_headline to 120 chars in metadata."""
        store, col = self._mock_store()
        long_headline = "X" * 200
        event = _make_event()
        event.news_headline = long_headline
        store.store_event(event)
        kwargs = col.add.call_args[1]
        stored_headline = kwargs["metadatas"][0]["news_headline"]
        assert len(stored_headline) <= 120

    def test_store_event_exception_returns_false(self):
        """If collection.add() raises, store_event() returns False without propagating."""
        store, col = self._mock_store()
        col.add.side_effect = Exception("write error")
        result = store.store_event(_make_event())
        assert result is False


class TestMacroEventsStoreFiltering:
    def _store_with_events(self, metadatas: list[dict]):
        """Build a store whose collection.get() returns given metadatas."""
        mock_col = MagicMock()
        mock_col.get.return_value = {
            "ids": [str(i) for i in range(len(metadatas))],
            "metadatas": metadatas,
        }
        store = MacroEventsStore.__new__(MacroEventsStore)
        store.available = True
        store.collection = mock_col
        return store

    def _meta(
        self,
        event_date,
        expiry,
        impact="TRANSIENT",
        scope="GLOBAL",
        primary_region="GLOBAL",
    ):
        return {
            "event_date": event_date,
            "expiry": expiry,
            "impact": impact,
            "scope": scope,
            "primary_region": primary_region,
            "detected_date": "2026-03-07",
            "event_type": "TARIFF_TRADE",
            "severity": "MEDIUM",
            "correlation_pct": 0.30,
            "peak_count": 6,
            "total_held": 20,
            "news_headline": "Test",
            "news_detail": "Detail",
            "primary_sector": "",
            "forced_reanalysis": False,
        }

    def test_get_active_events_calls_chromadb_with_expiry_filter(self):
        """get_active_events() passes expiry WHERE clause to ChromaDB."""
        future = (date.today() + timedelta(days=20)).isoformat()
        store = self._store_with_events([self._meta("2026-03-05", future)])
        store.get_active_events()
        assert store.collection.get.called
        call_kwargs = store.collection.get.call_args[1]
        where = call_kwargs.get("where", {})
        assert "expiry" in str(where)

    def test_get_active_events_region_filter_excludes_wrong_region(self):
        """REGIONAL event for .T is excluded when querying for .HK."""
        future = (date.today() + timedelta(days=20)).isoformat()
        meta = self._meta("2026-03-05", future, scope="REGIONAL", primary_region=".T")
        store = self._store_with_events([meta])
        events = store.get_active_events(region_filter=".HK")
        assert all(e.primary_region != ".HK" for e in events)

    def test_get_active_events_region_filter_global_event_always_included(self):
        """GLOBAL scope events are returned regardless of region_filter."""
        future = (date.today() + timedelta(days=20)).isoformat()
        meta = self._meta("2026-03-05", future, scope="GLOBAL", primary_region="GLOBAL")
        store = self._store_with_events([meta])
        events = store.get_active_events(region_filter=".HK")
        # GLOBAL event should pass the region filter
        assert any(e.scope == "GLOBAL" for e in events)

    def test_get_active_events_returns_macro_event_objects(self):
        """get_active_events() returns list of MacroEvent instances."""
        future = (date.today() + timedelta(days=20)).isoformat()
        store = self._store_with_events([self._meta("2026-03-05", future)])
        events = store.get_active_events()
        assert all(isinstance(e, MacroEvent) for e in events)

    def test_get_active_events_sorted_by_date_descending(self):
        """Multiple events are sorted newest-first."""
        future = (date.today() + timedelta(days=20)).isoformat()
        metas = [
            self._meta("2026-01-01", future),
            self._meta("2026-03-05", future),
            self._meta("2026-02-15", future),
        ]
        store = self._store_with_events(metas)
        events = store.get_active_events()
        dates = [e.event_date for e in events]
        assert dates == sorted(dates, reverse=True)

    def test_get_structural_events_since_passes_structural_filter(self):
        """get_structural_events_since() includes impact=STRUCTURAL in WHERE clause."""
        future = (date.today() + timedelta(days=90)).isoformat()
        structural = self._meta("2026-03-05", future, impact="STRUCTURAL")
        store = self._store_with_events([structural])
        store.get_structural_events_since("2026-01-01")
        call_kwargs = store.collection.get.call_args[1]
        where = call_kwargs.get("where", {})
        assert "STRUCTURAL" in str(where)

    def test_get_active_events_unavailable_returns_empty_list(self):
        """get_active_events() on unavailable store returns [] without error."""
        store = MacroEventsStore.__new__(MacroEventsStore)
        store.available = False
        store.collection = None
        assert store.get_active_events() == []

    def test_get_structural_events_unavailable_returns_empty_list(self):
        """get_structural_events_since() on unavailable store returns [] without error."""
        store = MacroEventsStore.__new__(MacroEventsStore)
        store.available = False
        store.collection = None
        assert store.get_structural_events_since("2026-01-01") == []

    def test_get_active_events_exception_returns_empty_list(self):
        """Exception inside get_active_events() is caught; returns []."""
        store = MacroEventsStore.__new__(MacroEventsStore)
        store.available = True
        store.collection = MagicMock()
        store.collection.get.side_effect = Exception("query error")
        assert store.get_active_events() == []

    def test_forced_reanalysis_preserved_in_deserialization(self):
        """forced_reanalysis=True stored in metadata survives round-trip deserialization."""
        future = (date.today() + timedelta(days=90)).isoformat()
        meta = self._meta("2026-03-05", future, impact="STRUCTURAL")
        meta["forced_reanalysis"] = True
        store = self._store_with_events([meta])
        events = store.get_active_events()
        assert len(events) == 1
        assert events[0].forced_reanalysis is True

    def test_forced_reanalysis_false_when_absent_from_metadata(self):
        """Missing forced_reanalysis key in metadata → defaults to False (not error)."""
        future = (date.today() + timedelta(days=20)).isoformat()
        meta = self._meta("2026-03-05", future)
        del meta["forced_reanalysis"]
        store = self._store_with_events([meta])
        events = store.get_active_events()
        assert len(events) == 1
        assert events[0].forced_reanalysis is False

    def test_get_structural_events_since_sorted_newest_first(self):
        """get_structural_events_since() returns events newest-first, matching get_active_events()."""
        future = (date.today() + timedelta(days=90)).isoformat()
        metas = [
            self._meta("2026-01-10", future, impact="STRUCTURAL"),
            self._meta("2026-03-05", future, impact="STRUCTURAL"),
            self._meta("2026-02-01", future, impact="STRUCTURAL"),
        ]
        store = self._store_with_events(metas)
        events = store.get_structural_events_since("2026-01-01")
        dates = [e.event_date for e in events]
        assert dates == sorted(dates, reverse=True), "Expected newest-first ordering"
