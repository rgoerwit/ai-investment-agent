"""Operator-curated ticker-override registry (config/ticker_overrides.json).

Confirmed listing migrations (e.g. 1264.TW → 1264.TWO after a TWSE→TPEx move)
must apply on BOTH the analysis side (normalize_ticker) and the IBKR
position/watchlist side, so reconciliation keys agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import ticker_corrections
from src.ticker_corrections import (
    apply_operator_override,
    load_operator_overrides,
)


@pytest.fixture
def overrides_file(tmp_path, monkeypatch):
    """Point the registry at a temp file and reset the module cache."""
    path = tmp_path / "ticker_overrides.json"
    monkeypatch.setattr(ticker_corrections, "TICKER_OVERRIDES_PATH", path)
    monkeypatch.setattr(ticker_corrections, "_operator_overrides_cache", None)

    def write(payload) -> Path:
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload),
            encoding="utf-8",
        )
        ticker_corrections._operator_overrides_cache = None
        return path

    yield write
    ticker_corrections._operator_overrides_cache = None


class TestRegistryLoading:
    def test_full_entry_shape(self, overrides_file):
        overrides_file({"1264.TW": {"new_ticker": "1264.TWO", "reason": "TWSE→TPEx"}})
        assert load_operator_overrides() == {"1264.TW": "1264.TWO"}

    def test_shorthand_entry_shape(self, overrides_file):
        overrides_file({"1264.tw": "1264.two"})
        assert load_operator_overrides() == {"1264.TW": "1264.TWO"}

    def test_missing_file_is_noop(self, overrides_file):
        assert load_operator_overrides() == {}

    def test_malformed_json_warns_and_noops(self, overrides_file):
        overrides_file("{not valid json")
        assert load_operator_overrides() == {}

    def test_invalid_entry_skipped(self, overrides_file):
        overrides_file({"GOOD.TW": "GOOD.TWO", "BAD.TW": {"reason": "no target"}})
        assert load_operator_overrides() == {"GOOD.TW": "GOOD.TWO"}


class TestOverrideApplication:
    def test_apply_operator_override(self, overrides_file):
        overrides_file({"1264.TW": "1264.TWO"})
        assert apply_operator_override("1264.tw") == ("1264.TWO", True)
        assert apply_operator_override("2330.TW") == ("2330.TW", False)

    def test_normalize_ticker_applies_override(self, overrides_file):
        overrides_file({"1264.TW": "1264.TWO"})
        from src.ticker_utils import normalize_ticker

        assert normalize_ticker("1264.TW") == "1264.TWO"

    def test_position_side_applies_override(self, overrides_file):
        overrides_file({"1264.TW": "1264.TWO"})
        from src.ibkr.portfolio import normalize_positions

        raw = {
            "contractDesc": "1264",
            "listingExchange": "TWSE",
            "currency": "TWD",
            "position": 1000,
            "mktValue": 5000,
            "mktPrice": 50.0,
            "avgCost": 45.0,
            "conid": 12345,
        }
        positions = normalize_positions([raw])
        assert len(positions) == 1
        assert positions[0].ticker.yf == "1264.TWO"

    def test_reconciliation_keys_agree(self, overrides_file):
        """Analysis-side and position-side tickers must match post-override."""
        overrides_file({"1264.TW": "1264.TWO"})
        from src.ibkr.portfolio import normalize_positions
        from src.ticker_utils import normalize_ticker

        raw = {
            "contractDesc": "1264",
            "listingExchange": "TWSE",
            "currency": "TWD",
            "position": 1,
            "mktValue": 1,
            "mktPrice": 1.0,
            "avgCost": 1.0,
            "conid": 1,
        }
        position_key = normalize_positions([raw])[0].ticker.yf
        analysis_key = normalize_ticker("1264.TW")
        assert position_key == analysis_key == "1264.TWO"
