"""Tests for TRADE_BLOCK parser and IBKR order builder."""

import pytest

from src.ibkr.models import TradeBlockData
from src.ibkr.order_builder import (
    build_order_dict,
    calculate_quantity,
    parse_trade_block,
    round_to_lot_size,
)


class TestParseTradeBlock:
    """Test TRADE_BLOCK parsing from trader output."""

    STANDARD_BUY = """
Based on the analysis, here is the trade recommendation:

TRADE_BLOCK:
ACTION: BUY
SIZE: 5.0%
CONVICTION: Medium
ENTRY: 2,145 (Scaled Limit)
STOP: 1,930 (-10.0%)
TARGET_1: 2,575 (+20.0%)
TARGET_2: 3,000 (+40.0%)
R:R: 3.2:1
SPECIAL: Tokyo hours only; JPY exposure; No ADR.
"""

    HOLD_WAIT = """
TRADE_BLOCK:
ACTION: HOLD (Wait for Entry)
SIZE: 2.5%
CONVICTION: Medium
ENTRY: 436.00 (Limit)
STOP: 395.00 (-9.4%)
TARGET_1: 498.00 (+14.2%)
TARGET_2: 575.00 (+31.8%)
R:R: 2.4:1
SPECIAL: High PFIC risk for US taxpayers; LSE direct access required.
"""

    REJECT = """
TRADE_BLOCK:
ACTION: REJECT
SIZE: 0.0%
CONVICTION: High
ENTRY: N/A (Liquidity Fail)
STOP: N/A
TARGET_1: N/A
TARGET_2: N/A
R:R: 0:1
SPECIAL: $162 avg daily turnover makes this security uninvestable.
"""

    CODE_FENCED = """
Here is the trade plan:

```
TRADE_BLOCK:
ACTION: BUY
SIZE: 3.0%
CONVICTION: High
ENTRY: 58.40 (Limit)
STOP: 52.00 (-11.0%)
TARGET_1: 67.00 (+14.7%)
TARGET_2: 75.00 (+28.4%)
R:R: 2.5:1
SPECIAL: HKD exposure; board lot 400 shares.
```
"""

    def test_standard_buy(self):
        result = parse_trade_block(self.STANDARD_BUY)
        assert result is not None
        assert result.action == "BUY"
        assert result.size_pct == 5.0
        assert result.conviction == "Medium"
        assert result.entry_price == 2145.0
        assert result.stop_price == 1930.0
        assert result.target_1_price == 2575.0
        assert result.target_2_price == 3000.0
        assert result.risk_reward == "3.2:1"
        assert "Tokyo" in result.special

    def test_hold_wait(self):
        result = parse_trade_block(self.HOLD_WAIT)
        assert result is not None
        assert result.action == "HOLD"
        assert result.size_pct == 2.5
        assert result.entry_price == 436.0
        assert result.stop_price == 395.0

    def test_reject(self):
        result = parse_trade_block(self.REJECT)
        assert result is not None
        assert result.action == "REJECT"
        assert result.size_pct == 0.0
        assert result.entry_price is None
        assert result.stop_price is None
        assert result.target_1_price is None

    def test_code_fenced(self):
        result = parse_trade_block(self.CODE_FENCED)
        assert result is not None
        assert result.action == "BUY"
        assert result.entry_price == 58.4

    def test_empty_text(self):
        assert parse_trade_block("") is None

    def test_no_trade_block(self):
        assert parse_trade_block("This is just some text without any block.") is None

    def test_partial_block_action_only(self):
        text = "ACTION: SELL\nSIZE: 0%"
        result = parse_trade_block(text)
        assert result is not None
        assert result.action == "SELL"


class TestRoundToLotSize:
    """Test board lot rounding for various exchanges."""

    def test_japan_100_lot(self):
        assert round_to_lot_size(150, "7203.T") == 100

    def test_japan_exact(self):
        assert round_to_lot_size(200, "7203.T") == 200

    def test_hong_kong_100_lot(self):
        assert round_to_lot_size(350, "0005.HK") == 300

    def test_taiwan_1000_lot(self):
        assert round_to_lot_size(1500, "2330.TW") == 1000

    def test_us_no_lot(self):
        assert round_to_lot_size(7, "AAPL") == 7

    def test_korea_1_lot(self):
        assert round_to_lot_size(3, "005930.KS") == 3

    def test_zero_quantity(self):
        assert round_to_lot_size(0, "7203.T") == 0

    def test_below_lot_size(self):
        assert round_to_lot_size(50, "7203.T") == 0


class TestCalculateQuantity:
    """Test order quantity calculation."""

    def test_basic_buy(self):
        qty = calculate_quantity(
            available_cash_usd=10000,
            entry_price_local=100,
            fx_rate_to_usd=1.0,
            size_pct=5.0,
            portfolio_value_usd=100000,
            yf_ticker="AAPL",
        )
        # 5% of 100k = 5000 USD / $100 = 50 shares
        assert qty == 50

    def test_cash_constrained(self):
        qty = calculate_quantity(
            available_cash_usd=2000,
            entry_price_local=100,
            fx_rate_to_usd=1.0,
            size_pct=5.0,
            portfolio_value_usd=100000,
            yf_ticker="AAPL",
        )
        # 5% of 100k = 5000, but only 2000 cash → 2000/100 = 20
        assert qty == 20

    def test_fx_conversion_jpy(self):
        qty = calculate_quantity(
            available_cash_usd=10000,
            entry_price_local=2000,
            fx_rate_to_usd=0.0067,  # JPY
            size_pct=5.0,
            portfolio_value_usd=100000,
            yf_ticker="7203.T",
        )
        # 5% of 100k = 5000 USD / (2000 * 0.0067) = 5000 / 13.4 ≈ 373 → round to 300
        assert qty == 300

    def test_zero_entry_price(self):
        assert calculate_quantity(10000, 0, 1.0, 5.0, 100000, "AAPL") == 0

    def test_zero_portfolio(self):
        assert calculate_quantity(10000, 100, 1.0, 5.0, 0, "AAPL") == 0

    def test_no_cash(self):
        assert calculate_quantity(0, 100, 1.0, 5.0, 100000, "AAPL") == 0

    def test_none_fx_defaults_to_1(self):
        qty = calculate_quantity(
            available_cash_usd=5000,
            entry_price_local=50,
            fx_rate_to_usd=None,
            size_pct=5.0,
            portfolio_value_usd=100000,
            yf_ticker="AAPL",
        )
        assert qty == 100  # 5000 / 50


class TestBuildOrderDict:
    """Test IBKR order dict construction."""

    def test_limit_buy(self):
        order = build_order_dict(
            conid=123456,
            action="BUY",
            quantity=100,
            price=58.40,
            order_type="LMT",
            account_id="U1234567",
        )
        assert order["conid"] == 123456
        assert order["side"] == "BUY"
        assert order["quantity"] == 100
        assert order["price"] == 58.40
        assert order["orderType"] == "LMT"
        assert order["acctId"] == "U1234567"
        assert order["tif"] == "GTC"

    def test_market_sell(self):
        order = build_order_dict(
            conid=789,
            action="SELL",
            quantity=50,
            order_type="MKT",
        )
        assert order["side"] == "SELL"
        assert order["orderType"] == "MKT"
        assert "price" not in order
        assert "acctId" not in order

    def test_no_price_for_market_order(self):
        order = build_order_dict(
            conid=1, action="BUY", quantity=10, price=100.0, order_type="MKT"
        )
        # MKT orders: price is ignored since IBKR determines price
        assert "price" not in order

    def test_day_tif(self):
        order = build_order_dict(
            conid=1, action="BUY", quantity=10, price=50.0, tif="DAY"
        )
        assert order["tif"] == "DAY"
