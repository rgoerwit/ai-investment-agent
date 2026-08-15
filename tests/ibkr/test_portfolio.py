"""Tests for portfolio reading and normalization."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.fx_normalization import set_fx_rate_cache
from src.ibkr.exceptions import IBKRAPIError
from src.ibkr.models import NormalizedPosition, PortfolioSummary
from src.ibkr.portfolio import (
    _resolve_watchlist_conid,
    build_portfolio_summary,
    normalize_positions,
    read_watchlist,
)
from src.ibkr.ticker import Ticker
from tests.ibkr.reconciler_cases import _FakeFxRateCache


@pytest.fixture(autouse=True)
def _deterministic_fx_cache():
    """normalize_positions() now resolves FX live-first via the shared
    FxRateCache — pin a fixed, network-free cache so these tests don't
    depend on yfinance availability or the real (periodically-refreshed)
    FALLBACK_RATES_TO_USD table."""
    set_fx_rate_cache(_FakeFxRateCache())
    yield
    set_fx_rate_cache(None)


class TestNormalizePositions:
    """Test conversion of raw IBKR positions to NormalizedPosition."""

    def test_standard_position(self):
        raw = [
            {
                "conid": 123456,
                "contractDesc": "7203",
                "listingExchange": "TSEJ",
                "position": 100,
                "avgCost": 2000.0,
                "mktValue": 1400.0,
                "unrealizedPnl": 67.0,
                "currency": "JPY",
                "mktPrice": 2100.0,
            }
        ]
        positions = normalize_positions(raw)
        assert len(positions) == 1
        p = positions[0]
        assert p.yf_ticker == "7203.T"
        assert p.quantity == 100
        assert p.avg_cost_local == 2000.0
        assert p.current_price_local == 2100.0
        assert p.currency == "JPY"
        assert p.market_value_usd == pytest.approx(1400.0)
        assert p.unrealized_pnl_usd == pytest.approx(67.0)
        assert p.market_value_basis == "BROKER_USD"
        assert p.unrealized_pnl_basis == "BROKER_USD"
        assert p.valuation_valid is True

    def test_hk_zero_padding(self):
        raw = [
            {
                "conid": 789,
                "contractDesc": "5",
                "listingExchange": "SEHK",
                "position": 400,
                "avgCost": 58.0,
                "mktValue": 2960.0,
                "currency": "HKD",
                "mktPrice": 59.0,
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].yf_ticker == "0005.HK"

    def test_korean_position_preserves_fixed_width_ibkr_symbol(self):
        raw = [
            {
                "conid": 1060,
                "contractDesc": "010130",
                "listingExchange": "KRX",
                "position": 3,
                "mktValue": 3_000_000,
                "currency": "KRW",
                "mktPrice": 1_429_000,
            }
        ]
        positions = normalize_positions(raw)

        assert positions[0].ticker.ibkr == "010130"
        assert positions[0].yf_ticker == "010130.KS"

    def test_empty_symbol_skipped(self):
        raw = [{"conid": 0, "contractDesc": "", "listingExchange": ""}]
        positions = normalize_positions(raw)
        assert len(positions) == 0

    def test_multiple_positions(self):
        raw = [
            {
                "conid": 1,
                "contractDesc": "7203",
                "listingExchange": "TSEJ",
                "position": 100,
                "mktPrice": 2100,
            },
            {
                "conid": 2,
                "contractDesc": "ASML",
                "listingExchange": "AEB",
                "position": 50,
                "mktPrice": 600,
            },
        ]
        positions = normalize_positions(raw)
        assert len(positions) == 2
        tickers = {p.yf_ticker for p in positions}
        assert "7203.T" in tickers
        assert "ASML.AS" in tickers

    def test_alternative_field_names(self):
        """Test fallback field names (qty, avgPrice, lastPrice)."""
        raw = [
            {
                "conid": 1,
                "contractDesc": "AAPL",
                "listingExchange": "SMART",
                "qty": 25,
                "avgPrice": 150.0,
                "marketValue": 3900.0,
                "lastPrice": 156.0,
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].quantity == 25
        assert positions[0].avg_cost_local == 150.0
        assert positions[0].current_price_local == 156.0

    def test_jpy_market_value_converted_to_usd(self):
        """JPY mktValue is converted to USD using FALLBACK_RATES_TO_USD (0.0067)."""
        raw = [
            {
                "conid": 1,
                "contractDesc": "7203",
                "listingExchange": "TSEJ",
                "position": 100,
                "mktValue": 210_000.0,  # ¥210,000
                "mktPrice": 2_100.0,
                "currency": "JPY",
            }
        ]
        positions = normalize_positions(raw)
        # ¥210,000 × 0.0067 = $1,407
        assert positions[0].market_value_usd == pytest.approx(1407.0, rel=0.01)

    def test_usd_market_value_unchanged(self):
        """USD positions are not double-converted (rate = 1.0)."""
        raw = [
            {
                "conid": 2,
                "contractDesc": "AAPL",
                "listingExchange": "SMART",
                "position": 10,
                "mktValue": 1800.0,
                "currency": "USD",
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].market_value_usd == pytest.approx(1800.0)

    def test_unknown_currency_fails_closed(self):
        """Unknown non-USD currencies must not silently use an identity rate."""
        raw = [
            {
                "conid": 3,
                "contractDesc": "XYZ",
                "listingExchange": "SMART",
                "position": 1,
                "mktValue": 500.0,
                "currency": "ZZZ",  # fictitious currency
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].market_value_usd == 0.0
        assert positions[0].unrealized_pnl_usd == 0.0
        assert positions[0].valuation_valid is False
        assert "ZZZ" in (positions[0].valuation_issue or "")

    @pytest.mark.parametrize(
        ("ticker", "exchange", "currency", "fx_rate"),
        [
            ("7203", "TSEJ", "JPY", 0.0067),
            ("001060", "KRX", "KRW", 0.00075),
            ("3005", "TWSE", "TWD", 0.032),
        ],
    )
    def test_asian_local_values_are_converted_once(
        self,
        ticker,
        exchange,
        currency,
        fx_rate,
    ):
        raw = [
            {
                "conid": 10,
                "contractDesc": ticker,
                "listingExchange": exchange,
                "position": 100,
                "avgCost": 100.0,
                "mktPrice": 90.0,
                "mktValue": 9_000.0,
                "unrealizedPnl": -1_000.0,
                "currency": currency,
            }
        ]

        position = normalize_positions(raw)[0]

        assert position.market_value_usd == pytest.approx(9_000.0 * fx_rate)
        assert position.unrealized_pnl_usd == pytest.approx(-1_000.0 * fx_rate)
        assert position.market_value_basis == "LOCAL_CONVERTED"
        assert position.unrealized_pnl_basis == "LOCAL_CONVERTED"
        assert position.valuation_valid is True

    def test_jpy_values_already_in_usd_are_not_double_converted(self):
        raw = [
            {
                "conid": 11,
                "contractDesc": "7203",
                "listingExchange": "TSEJ",
                "position": 100,
                "avgCost": 100.0,
                "mktPrice": 90.0,
                "mktValue": 60.3,
                "unrealizedPnl": -6.7,
                "currency": "JPY",
            }
        ]

        position = normalize_positions(raw)[0]

        assert position.market_value_usd == pytest.approx(60.3)
        assert position.unrealized_pnl_usd == pytest.approx(-6.7)
        assert position.market_value_basis == "BROKER_USD"
        assert position.unrealized_pnl_basis == "BROKER_USD"

    def test_inconsistent_broker_units_are_quarantined(self):
        raw = [
            {
                "conid": 12,
                "contractDesc": "7203",
                "listingExchange": "TSEJ",
                "position": 100,
                "avgCost": 100.0,
                "mktPrice": 90.0,
                "mktValue": 500.0,
                "unrealizedPnl": -1_000.0,
                "currency": "JPY",
            }
        ]

        position = normalize_positions(raw)[0]

        assert position.valuation_valid is False
        assert position.market_value_usd == 0.0
        assert "market value" in (position.valuation_issue or "")

    @pytest.mark.parametrize(
        ("field", "bad_value", "issue_field"),
        [
            ("position", "not-a-number", "quantity"),
            ("mktValue", {}, "market_value"),
            ("mktPrice", True, "current_price"),
            ("avgCost", "broken", "avg_cost"),
            ("unrealizedPnl", [], "unrealized_pnl"),
            pytest.param(
                "mktValue",
                10**10_000,
                "market_value",
                id="overflowing-market-value",
            ),
        ],
    )
    def test_malformed_numeric_field_quarantines_row_without_aborting_snapshot(
        self,
        field,
        bad_value,
        issue_field,
    ):
        malformed = {
            "conid": 12,
            "contractDesc": "7203",
            "listingExchange": "TSEJ",
            "position": 100,
            "avgCost": 100.0,
            "mktPrice": 90.0,
            "mktValue": 9_000.0,
            "unrealizedPnl": -1_000.0,
            "currency": "JPY",
        }
        malformed[field] = bad_value
        valid = {
            "conid": 13,
            "contractDesc": "AAPL",
            "listingExchange": "SMART",
            "position": 10,
            "avgCost": 175.0,
            "mktPrice": 180.0,
            "mktValue": 1_800.0,
            "unrealizedPnl": 50.0,
            "currency": "USD",
        }

        positions = normalize_positions([malformed, valid])

        assert len(positions) == 2
        assert positions[0].valuation_valid is False
        assert positions[0].market_value_usd == 0.0
        assert issue_field in (positions[0].valuation_issue or "")
        assert positions[1].valuation_valid is True

    def test_non_usd_value_without_current_price_is_quarantined(self):
        raw = [
            {
                "conid": 14,
                "contractDesc": "7203",
                "listingExchange": "TSEJ",
                "position": 100,
                "avgCost": 100.0,
                "mktValue": 9_000.0,
                "unrealizedPnl": -1_000.0,
                "currency": "JPY",
            }
        ]

        position = normalize_positions(raw)[0]

        assert position.valuation_valid is False
        assert position.market_value_usd == 0.0
        assert "market value" in (position.valuation_issue or "")

    def test_lse_price_keeps_ibkr_denomination(self):
        """IBKR quotes .L in GBP and normalize_positions leaves it there.

        The previous ".L + GBP -> x100" rule assumed the *analysis* side always
        held pence. That is true only when the fetcher declined a minor-unit
        conversion; when it succeeded (GAMA.L: 975.5 GBp -> 9.755 GBP) the rule
        produced an ~85x mismatch and a false valuation-reference review.
        Denomination is now reconciled at comparison time by currency code —
        see reconciliation_rules._comparable_prices.
        """
        raw = [
            {
                "conid": 101,
                "contractDesc": "GAMA",
                "listingExchange": "LSE",
                "position": 200,
                "mktValue": 1788.0,
                "currency": "GBP",
                "mktPrice": 8.94,  # IBKR: GBP 8.94
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].yf_ticker == "GAMA.L"
        assert positions[0].current_price_local == pytest.approx(8.94)
        assert positions[0].currency == "GBP"

    def test_lse_currency_defaults_to_ibkr_code(self):
        """An omitted IBKR currency still resolves to the venue's major code."""
        raw = [
            {
                "conid": 102,
                "contractDesc": "KLR",
                "listingExchange": "LSE",
                "position": 50,
                "mktValue": 1101.0,
                # currency field intentionally absent
                "mktPrice": 22.02,
            }
        ]
        positions = normalize_positions(raw)
        assert positions[0].yf_ticker == "KLR.L"
        assert positions[0].currency == "GBP"
        assert positions[0].current_price_local == pytest.approx(22.02)

    def test_lse_fx_rate_matches_its_currency_code(self):
        """The FX rate must correspond to the code on the same record.

        This is the invariant the old rule broke: it rewrote the price to pence
        and scaled the rate to match, so price and rate were consistent but the
        code no longer described what the analysis side held.
        """
        raw = [
            {
                "conid": 103,
                "contractDesc": "GAMA",
                "listingExchange": "LSE",
                "position": 100,
                "mktValue": 894.0,
                "currency": "GBP",
                "mktPrice": 8.94,
            }
        ]
        p = normalize_positions(raw)[0]
        assert p.currency == "GBP"
        assert p.current_price_local == pytest.approx(8.94)
        # Pounds rate, not the pence rate the x100 rule used to force.
        assert p.fx_rate_to_usd == pytest.approx(1.27)

    def test_position_conid_primary_exchange_overrides_stale_raw_twse(self):
        """Held Taiwan conid beats stale raw TWSE metadata after TPEx migration."""
        client = _contract_info_client(
            {
                "symbol": "1264",
                "primaryExch": "TPEX",
                "exchange": "SMART",
                "currency": "TWD",
            }
        )
        raw = [
            {
                "conid": 1264,
                "contractDesc": "1264",
                "listingExchange": "TWSE",
                "position": 1000,
                "mktValue": 5000,
                "currency": "TWD",
                "mktPrice": 50.0,
            }
        ]

        with patch("src.ibkr.portfolio.cache_conid_mapping"):
            positions = normalize_positions(raw, client=client)

        assert positions[0].yf_ticker == "1264.TWO"
        client.get_contract_info.assert_called_once_with(1264, compete=False)

    def test_position_conid_resolves_blank_taiwan_exchange_to_tpex(self):
        client = _contract_info_client(
            {"symbol": "1264", "primaryExch": "TPEX", "currency": "TWD"}
        )
        raw = [
            {
                "conid": 1264,
                "contractDesc": "1264",
                "listingExchange": "",
                "position": 1000,
                "mktValue": 5000,
                "currency": "TWD",
            }
        ]

        with patch("src.ibkr.portfolio.cache_conid_mapping"):
            positions = normalize_positions(raw, client=client)

        assert positions[0].yf_ticker == "1264.TWO"

    def test_position_raw_twse_without_client_preserves_current_fallback(self):
        raw = [
            {
                "conid": 1264,
                "contractDesc": "1264",
                "listingExchange": "TWSE",
                "position": 1000,
                "mktValue": 5000,
                "currency": "TWD",
            }
        ]

        positions = normalize_positions(raw)

        assert positions[0].yf_ticker == "1264.TW"

    def test_us_smart_usd_position_skips_contract_info(self):
        client = _contract_info_client(
            {"symbol": "AAPL", "primaryExch": "NASDAQ", "currency": "USD"}
        )
        raw = [
            {
                "conid": 1,
                "contractDesc": "AAPL",
                "listingExchange": "SMART",
                "position": 5,
                "mktValue": 1000,
                "currency": "USD",
            }
        ]

        positions = normalize_positions(raw, client=client)

        assert positions[0].yf_ticker == "AAPL"
        assert positions[0].ticker_identity_verified is True
        client.get_contract_info.assert_not_called()

    def test_smart_eur_search_result_remains_unverified(self):
        client = _contract_info_client(
            {"symbol": "AGS", "primaryExch": "SMART", "currency": "EUR"}
        )
        raw = [
            {
                "conid": 123,
                "contractDesc": "AGS",
                "listingExchange": "SMART",
                "position": 100,
                "mktValue": 1000,
                "currency": "EUR",
            }
        ]

        with (
            patch("src.ibkr.portfolio.cache_conid_mapping") as cache_mapping,
            patch(
                "src.ibkr.ticker_mapper._yf_search_ticker",
                return_value="AGS.BR",
            ),
        ):
            positions = normalize_positions(raw, client=client)

        assert positions[0].yf_ticker == "AGS.BR"
        assert positions[0].ticker_identity_verified is False
        assert positions[0].ticker_resolution_source == "yfinance_search"
        cache_mapping.assert_not_called()

    def test_korean_multi_exchange_currency_forces_live_resolution(self):
        client = _contract_info_client(
            {"symbol": "35420", "primaryExch": "KOSDAQ", "currency": "KRW"}
        )
        raw = [
            {
                "conid": 35420,
                "contractDesc": "35420",
                "listingExchange": "KRX",
                "position": 3,
                "mktValue": 3_000_000,
                "currency": "KRW",
            }
        ]

        with patch("src.ibkr.portfolio.cache_conid_mapping"):
            positions = normalize_positions(raw, client=client)

        assert positions[0].yf_ticker == "035420.KQ"

    def test_position_contract_info_empty_keeps_position(self):
        client = _contract_info_client({})
        raw = [
            {
                "conid": 1264,
                "contractDesc": "1264",
                "listingExchange": "TWSE",
                "position": 1000,
                "mktValue": 5000,
                "currency": "TWD",
            }
        ]

        positions = normalize_positions(raw, client=client)

        assert len(positions) == 1
        assert positions[0].yf_ticker == "1264.TW"

    def test_position_security_definition_resolves_when_contract_info_unavailable(self):
        client = _contract_info_client({})
        client.get_security_definition.return_value = {
            "ticker": "1264",
            "listingExchange": "TPEX",
            "allExchanges": "TPEX",
            "currency": "TWD",
        }
        raw = [
            {
                "conid": 1264,
                "contractDesc": "1264",
                "position": 1000,
                "mktValue": 5000,
                "currency": "TWD",
            }
        ]

        with patch("src.ibkr.portfolio.cache_conid_mapping"):
            positions = normalize_positions(raw, client=client)

        assert positions[0].yf_ticker == "1264.TWO"
        assert positions[0].ticker_identity_verified is True
        assert positions[0].ticker_resolution_source == "exchange_map"
        client.get_contract_info.assert_called_once_with(1264, compete=False)
        client.get_security_definition.assert_called_once_with(1264)

    def test_position_security_definition_exception_keeps_position(self, caplog):
        client = _contract_info_client({})
        client.get_security_definition.side_effect = Exception(
            "secret path /tmp/private-token"
        )
        raw = [
            {
                "conid": 1264,
                "contractDesc": "1264",
                "position": 1000,
                "mktValue": 5000,
                "currency": "TWD",
            }
        ]

        with caplog.at_level(logging.WARNING, logger="src.ibkr.portfolio"):
            positions = normalize_positions(raw, client=client)

        assert positions[0].yf_ticker == "1264.TW"
        assert any(
            "conid_security_definition_failed" in r.message for r in caplog.records
        )
        assert not any("/tmp/private-token" in r.message for r in caplog.records)

    def test_position_contract_info_exception_keeps_position(self, caplog):
        client = _contract_info_client(Exception("secret path /tmp/private-token"))
        raw = [
            {
                "conid": 1264,
                "contractDesc": "1264",
                "listingExchange": "TWSE",
                "position": 1000,
                "mktValue": 5000,
                "currency": "TWD",
            }
        ]

        with caplog.at_level(logging.WARNING, logger="src.ibkr.portfolio"):
            positions = normalize_positions(raw, client=client)

        assert positions[0].yf_ticker == "1264.TW"
        assert any("conid_contract_info_failed" in r.message for r in caplog.records)
        assert not any("/tmp/private-token" in r.message for r in caplog.records)

    def test_position_contract_info_missing_symbol_keeps_raw_mapping(self):
        client = _contract_info_client({"primaryExch": "TPEX", "currency": "TWD"})
        raw = [
            {
                "conid": 1264,
                "contractDesc": "1264",
                "listingExchange": "TWSE",
                "position": 1000,
                "mktValue": 5000,
                "currency": "TWD",
            }
        ]

        positions = normalize_positions(raw, client=client)

        assert positions[0].yf_ticker == "1264.TW"

    def test_position_bad_conid_skips_live_resolution(self):
        client = _contract_info_client(
            {"symbol": "1264", "primaryExch": "TPEX", "currency": "TWD"}
        )
        raw = [
            {
                "conid": "not-a-number",
                "contractDesc": "1264",
                "listingExchange": "TWSE",
                "position": 1000,
                "mktValue": 5000,
                "currency": "TWD",
            }
        ]

        positions = normalize_positions(raw, client=client)

        assert positions[0].conid == 0
        assert positions[0].yf_ticker == "1264.TW"
        client.get_contract_info.assert_not_called()

    def test_force_live_bypasses_stale_conid_cache(self):
        client = _contract_info_client(
            {"symbol": "1264", "primaryExch": "TPEX", "currency": "TWD"}
        )
        raw = [
            {
                "conid": 1264,
                "contractDesc": "1264",
                "listingExchange": "TWSE",
                "position": 1000,
                "mktValue": 5000,
                "currency": "TWD",
            }
        ]

        with (
            patch("src.ibkr.portfolio.yf_ticker_from_conid", return_value="1264.TW"),
            patch("src.ibkr.portfolio.cache_conid_mapping"),
        ):
            positions = normalize_positions(raw, client=client)

        assert positions[0].yf_ticker == "1264.TWO"


class TestBuildPortfolioSummary:
    """Test portfolio summary construction."""

    def test_from_base_ledger(self):
        ledger = {
            "BASE": {
                "cashbalance": 18200.0,
                "netliquidationvalue": 125430.0,
            }
        }
        positions = [
            NormalizedPosition(
                conid=1,
                ticker=Ticker.from_yf("7203.T", currency="JPY"),
                quantity=100,
                market_value_usd=14000,
                currency="JPY",
            ),
        ]
        summary = build_portfolio_summary(ledger, positions, "U1234567")
        assert summary.account_id == "U1234567"
        assert summary.portfolio_value_usd == 125430.0
        assert summary.cash_balance_usd == 18200.0
        assert summary.position_count == 1
        # Available cash = 18200 - (125430 * 0.03) = 18200 - 3762.9 = 14437.1
        assert summary.available_cash_usd == pytest.approx(14437.1, rel=0.01)

    def test_fallback_to_positions_sum(self):
        ledger = {}  # No ledger data
        positions = [
            NormalizedPosition(
                conid=1, ticker=Ticker.from_yf("A"), quantity=10, market_value_usd=5000
            ),
            NormalizedPosition(
                conid=2, ticker=Ticker.from_yf("B"), quantity=20, market_value_usd=8000
            ),
        ]
        summary = build_portfolio_summary(ledger, positions, "U999")
        assert summary.portfolio_value_usd == 13000.0

    def test_zero_cash_buffer(self):
        ledger = {"BASE": {"cashbalance": 10000, "netliquidationvalue": 100000}}
        summary = build_portfolio_summary(ledger, [], "U1", cash_buffer_pct=0.0)
        assert summary.available_cash_usd == 10000.0

    def test_high_cash_buffer(self):
        ledger = {"BASE": {"cashbalance": 5000, "netliquidationvalue": 100000}}
        summary = build_portfolio_summary(ledger, [], "U1", cash_buffer_pct=0.10)
        # available = 5000 - 10000 = negative → clamped to 0
        assert summary.available_cash_usd == 0.0


# ---------------------------------------------------------------------------
# Helpers for read_watchlist tests
# ---------------------------------------------------------------------------


def _mock_client(rows: list[dict]) -> MagicMock:
    """Return a MagicMock IBKR client whose get_watchlist() returns `rows`."""
    client = MagicMock()
    client.get_watchlist.return_value = rows
    return client


def _contract_info_client(payload: dict | Exception) -> MagicMock:
    """Return a client mock that serves one contract-info response."""
    client = MagicMock()
    if isinstance(payload, Exception):
        client.get_contract_info.side_effect = payload
    else:
        client.get_contract_info.return_value = payload
    client.get_security_definition.return_value = {}
    return client


_RESOLVE = "src.ibkr.portfolio._resolve_watchlist_conid"


class TestReadWatchlistFailClosed:
    """read_watchlist fails closed for an explicit watchlist, soft-fails for default."""

    def test_explicit_name_fetch_error_propagates(self):
        """Explicit --watchlist-name + API error → propagate (no phantom-empty list)."""
        client = MagicMock()
        client.get_watchlist.side_effect = IBKRAPIError("IBKR watchlist fetch failed")
        with pytest.raises(IBKRAPIError):
            read_watchlist(client, "watchlist-2026")

    def test_default_fetch_error_soft_fails_to_empty(self, caplog):
        """Default (unnamed) discovery + API error → best-effort empty set, logged."""
        client = MagicMock()
        client.get_watchlist.side_effect = IBKRAPIError("IBKR watchlist fetch failed")
        with caplog.at_level(logging.WARNING, logger="src.ibkr.portfolio"):
            result = read_watchlist(client, "")
        assert result == set()
        assert any(
            "watchlist_default_fetch_failed" in r.message for r in caplog.records
        )

    def test_not_found_returns_none(self):
        """A genuinely-missing watchlist (get_watchlist → None) stays None, not error."""
        client = MagicMock()
        client.get_watchlist.return_value = None
        assert read_watchlist(client, "watchlist-2026") is None

    def test_empty_watchlist_returns_empty_set(self):
        """An existing-but-empty watchlist (get_watchlist → []) is empty, not failure."""
        client = MagicMock()
        client.get_watchlist.return_value = []
        assert read_watchlist(client, "watchlist-2026") == set()


class TestReadWatchlistRowParsing:
    """read_watchlist must correctly extract conids from all known IBKR row formats
    and emit actionable warnings for unrecognised formats."""

    def test_legacy_format_plain_int_c_field(self):
        """Legacy format: {"C": 12345678} — C is an integer conid."""
        rows = [{"C": 39131511}]
        with patch(_RESOLVE, return_value="5434.TW"):
            result = read_watchlist(_mock_client(rows))
        assert result == {"5434.TW"}

    def test_new_format_c_at_exchange_with_conid_field(self):
        """New IBKR format: {"C": "39131511@TWSE", "conid": 39131511}.

        The "conid" integer field is preferred; the composite "C" string is not
        used as the primary source (but must not block resolution).
        """
        rows = [
            {
                "ST": "STK",
                "C": "39131511@TWSE",
                "conid": 39131511,
                "ticker": "5434",
                "name": "TOPCO SCIENTIFIC CO LTD",
            }
        ]
        with patch(_RESOLVE, return_value="5434.TW") as mock_resolve:
            result = read_watchlist(_mock_client(rows))
        assert result == {"5434.TW"}
        mock_resolve.assert_called_once_with(39131511, mock_resolve.call_args[0][1])

    def test_new_format_c_at_exchange_without_conid_field(self):
        """Fallback: {"C": "39131511@TWSE"} with no "conid" field.

        Must strip the @EXCHANGE suffix and parse the numeric part.
        """
        rows = [{"C": "39131511@TWSE"}]
        with patch(_RESOLVE, return_value="5434.TW"):
            result = read_watchlist(_mock_client(rows))
        assert result == {"5434.TW"}

    def test_spacer_row_silently_skipped(self):
        """Spacer rows {"H": "1"} must be skipped without warning."""
        rows = [{"H": "1"}, {"C": 39131511}]
        with patch(_RESOLVE, return_value="5434.TW"):
            result = read_watchlist(_mock_client(rows))
        assert result == {"5434.TW"}

    def test_mixed_legacy_and_new_format(self):
        """A watchlist with both legacy and new-format rows resolves all securities."""
        rows = [
            {"C": 11111111},  # legacy int
            {"C": "22222222@TWSE", "conid": 22222222},  # new format
            {"C": "33333333@SGX"},  # new format, no conid field
            {"H": "1"},  # spacer
        ]

        def _resolve(conid, _client):
            return {11111111: "5334.T", 22222222: "5434.TW", 33333333: "BEC.SI"}.get(
                conid, ""
            )

        with patch(_RESOLVE, side_effect=_resolve):
            result = read_watchlist(_mock_client(rows))

        assert result == {"5334.T", "5434.TW", "BEC.SI"}

    def test_unknown_format_row_emits_warning(self, caplog):
        """A row with no recognisable conid field emits a WARNING — signals API change."""
        rows = [{"symbol": "XYZ", "exchange": "NYSE"}]  # hypothetical future format
        with caplog.at_level(logging.WARNING, logger="src.ibkr.portfolio"):
            with patch(_RESOLVE, return_value=""):
                result = read_watchlist(_mock_client(rows))
        assert result == set()
        assert any("watchlist_row_unknown_format" in r.message for r in caplog.records)

    def test_unparseable_conid_emits_warning(self, caplog):
        """A row whose extracted conid cannot be cast to int emits a WARNING."""
        rows = [{"C": "not-a-number"}]
        with caplog.at_level(logging.WARNING, logger="src.ibkr.portfolio"):
            with patch(_RESOLVE, return_value=""):
                result = read_watchlist(_mock_client(rows))
        assert result == set()
        assert any("watchlist_bad_conid" in r.message for r in caplog.records)

    def test_watchlist_none_returns_none(self):
        """Client returning None (watchlist not found) propagates as None."""
        client = MagicMock()
        client.get_watchlist.return_value = None
        assert read_watchlist(client) is None

    def test_watchlist_empty_returns_empty_set(self):
        """Client returning [] (empty watchlist) returns an empty set."""
        assert read_watchlist(_mock_client([])) == set()

    def test_watchlist_conid_prefers_primary_exchange(self):
        client = _contract_info_client(
            {
                "symbol": "1264",
                "primaryExch": "TPEX",
                "listingExchange": "TWSE",
                "exchange": "SMART",
                "currency": "TWD",
            }
        )

        with (
            patch("src.ibkr.portfolio.yf_ticker_from_conid", return_value=None),
            patch("src.ibkr.portfolio.cache_conid_mapping"),
        ):
            result = _resolve_watchlist_conid(1264, client)

        assert result == "1264.TWO"

    def test_watchlist_without_client_can_still_use_suffixed_cache(self):
        with patch("src.ibkr.portfolio.yf_ticker_from_conid", return_value="1264.TWO"):
            result = _resolve_watchlist_conid(1264, None)

        assert result == "1264.TWO"

    def test_conid_resolver_uses_compete_false(self):
        client = _contract_info_client(
            {"symbol": "1264", "primaryExch": "TPEX", "currency": "TWD"}
        )

        with (
            patch("src.ibkr.portfolio.yf_ticker_from_conid", return_value=None),
            patch("src.ibkr.portfolio.cache_conid_mapping"),
        ):
            _resolve_watchlist_conid(1264, client)

        client.get_contract_info.assert_called_once_with(1264, compete=False)
