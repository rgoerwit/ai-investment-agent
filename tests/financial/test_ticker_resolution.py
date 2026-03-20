"""
Unit tests for proactive mnemonic→numeric ticker pre-resolution.

Coverage:
- Non-target suffix passes through unchanged
- Already-numeric base passes through unchanged
- Cache hit skips network call
- Successful yahooquery resolution + cache write
- Graceful fallback on network error
- Non-numeric yahooquery result is ignored (not cached)
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.data.fetcher import SmartMarketDataFetcher


@pytest.fixture
def fetcher(tmp_path):
    """DataFetcher instance with cache file redirected to a temp directory."""
    f = SmartMarketDataFetcher()
    # Redirect cache to an isolated temp file so tests don't share state
    f._MNEMONIC_CACHE_FILE = tmp_path / "ticker_mnemonic_map.json"
    f._mnemonic_cache = {}
    return f


class TestPreResolveTicker:
    def test_non_target_suffix_unchanged(self, fetcher):
        """.AS is not in _MNEMONIC_EXCHANGES — should return immediately without network."""
        with patch("yahooquery.search") as mock_search:
            result = asyncio.run(fetcher._pre_resolve_ticker("ASML.AS"))
        assert result == "ASML.AS"
        mock_search.assert_not_called()

    def test_already_numeric_unchanged(self, fetcher):
        """Base '4731' is all-digits — skip resolution, return as-is."""
        with patch("yahooquery.search") as mock_search:
            result = asyncio.run(fetcher._pre_resolve_ticker("4731.KL"))
        assert result == "4731.KL"
        mock_search.assert_not_called()

    def test_cache_hit_skips_network(self, fetcher):
        """When the mnemonic is already cached, yahooquery must not be called."""
        fetcher._mnemonic_cache["SCIENTX.KL"] = "4731.KL"
        with patch("yahooquery.search") as mock_search:
            result = asyncio.run(fetcher._pre_resolve_ticker("SCIENTX.KL"))
        assert result == "4731.KL"
        mock_search.assert_not_called()

    @patch(
        "yahooquery.search",
        return_value={"quotes": [{"symbol": "4731.KL"}]},
    )
    def test_resolves_mnemonic_via_yahooquery(self, mock_search, fetcher):
        """Happy path: mnemonic resolved to numeric ticker."""
        result = asyncio.run(fetcher._pre_resolve_ticker("SCIENTX.KL"))
        assert result == "4731.KL"
        assert fetcher._mnemonic_cache["SCIENTX.KL"] == "4731.KL"

    @patch("yahooquery.search", side_effect=Exception("network error"))
    def test_graceful_fallback_on_error(self, mock_search, fetcher):
        """Must return the original ticker, not raise, when yahooquery fails."""
        result = asyncio.run(fetcher._pre_resolve_ticker("PADINI.KL"))
        assert result == "PADINI.KL"
        assert "PADINI.KL" not in fetcher._mnemonic_cache

    @patch(
        "yahooquery.search",
        return_value={"quotes": [{"symbol": "SCIENTX.KL"}]},
    )
    def test_ignores_non_numeric_result(self, mock_search, fetcher):
        """If yahooquery returns the mnemonic itself, do not cache or substitute."""
        result = asyncio.run(fetcher._pre_resolve_ticker("SCIENTX.KL"))
        assert result == "SCIENTX.KL"
        assert "SCIENTX.KL" not in fetcher._mnemonic_cache

    @patch(
        "yahooquery.search",
        return_value={"quotes": [{"symbol": "4731.KL"}]},
    )
    def test_cache_persisted_to_disk(self, mock_search, fetcher):
        """After resolution, the mapping should be written to the JSON cache file."""
        asyncio.run(fetcher._pre_resolve_ticker("SCIENTX.KL"))
        assert fetcher._MNEMONIC_CACHE_FILE.exists()
        with open(fetcher._MNEMONIC_CACHE_FILE) as fh:
            raw = json.load(fh)
        assert "SCIENTX.KL" in raw
        assert raw["SCIENTX.KL"]["resolved"] == "4731.KL"

    def test_load_mnemonic_cache_ignores_expired(self, fetcher, tmp_path):
        """Entries older than TTL must not be loaded."""
        cache_file = tmp_path / "ticker_mnemonic_map.json"
        old_ts = time.time() - (fetcher._MNEMONIC_CACHE_TTL + 1)
        data = {"OLD.KL": {"resolved": "9999.KL", "ts": old_ts}}
        cache_file.write_text(json.dumps(data))
        fetcher._MNEMONIC_CACHE_FILE = cache_file
        loaded = fetcher._load_mnemonic_cache()
        assert "OLD.KL" not in loaded

    def test_load_mnemonic_cache_accepts_fresh(self, fetcher, tmp_path):
        """Entries within TTL should be loaded correctly."""
        cache_file = tmp_path / "ticker_mnemonic_map.json"
        data = {"SCIENTX.KL": {"resolved": "4731.KL", "ts": time.time()}}
        cache_file.write_text(json.dumps(data))
        fetcher._MNEMONIC_CACHE_FILE = cache_file
        loaded = fetcher._load_mnemonic_cache()
        assert loaded.get("SCIENTX.KL") == "4731.KL"

    def test_load_mnemonic_cache_handles_corrupt_file(self, fetcher, tmp_path):
        """Corrupt JSON should return an empty dict, not raise."""
        cache_file = tmp_path / "ticker_mnemonic_map.json"
        cache_file.write_text("{not valid json")
        fetcher._MNEMONIC_CACHE_FILE = cache_file
        loaded = fetcher._load_mnemonic_cache()
        assert loaded == {}

    @patch(
        "yahooquery.search",
        return_value={"quotes": [{"symbol": "OTHER.KL"}, {"symbol": "4731.KL"}]},
    )
    def test_picks_first_numeric_match(self, mock_search, fetcher):
        """When multiple quotes are returned, resolve to the first numeric match."""
        # "OTHER" is not all-digits, "4731" is — should pick 4731.KL
        result = asyncio.run(fetcher._pre_resolve_ticker("SCIENTX.KL"))
        assert result == "4731.KL"
