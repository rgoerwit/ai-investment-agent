"""
TRADE_BLOCK parser and IBKR order dict builder.

Parses the structured TRADE_BLOCK output from the Trader agent
and converts it into IBKR-compatible order dictionaries.
"""

from __future__ import annotations

import re

import structlog

from src.ibkr.models import TradeBlockData

logger = structlog.get_logger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# TRADE_BLOCK Parsing
# ══════════════════════════════════════════════════════════════════════════════

# Regex for each TRADE_BLOCK field — tolerant of whitespace and markdown
_FIELD_PATTERNS = {
    "action": re.compile(r"ACTION:\s*(.+?)(?:\n|$)", re.IGNORECASE),
    "size": re.compile(r"SIZE:\s*([\d.]+)\s*%", re.IGNORECASE),
    "conviction": re.compile(r"CONVICTION:\s*(\w+)", re.IGNORECASE),
    "entry": re.compile(r"ENTRY:\s*(.+?)(?:\n|$)", re.IGNORECASE),
    "stop": re.compile(r"STOP:\s*(.+?)(?:\n|$)", re.IGNORECASE),
    "target_1": re.compile(r"TARGET_1:\s*(.+?)(?:\n|$)", re.IGNORECASE),
    "target_2": re.compile(r"TARGET_2:\s*(.+?)(?:\n|$)", re.IGNORECASE),
    "risk_reward": re.compile(r"R:R:\s*(.+?)(?:\n|$)", re.IGNORECASE),
    "special": re.compile(r"SPECIAL:\s*(.+?)(?:\n|$)", re.IGNORECASE),
}


def parse_price(raw: str | None) -> float | None:
    """Extract numeric price from TRADE_BLOCK field value."""
    if not raw:
        return None
    raw = raw.strip()
    if raw.upper().startswith("N/A"):
        return None
    match = re.match(r"([\d,]+(?:\.\d+)?)", raw)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def parse_trade_block(text: str) -> TradeBlockData | None:
    """
    Parse a TRADE_BLOCK from trader output text.

    Searches for the TRADE_BLOCK section and extracts all fields.
    Tolerant of markdown formatting, code fences, and whitespace.

    Args:
        text: Full trader output containing TRADE_BLOCK

    Returns:
        TradeBlockData if found, None otherwise
    """
    if not text:
        return None

    # Find the TRADE_BLOCK section (may be wrapped in markdown code fences)
    block_match = re.search(
        r"TRADE_BLOCK[:\s]*\n(.*?)(?:\n```|\n---|\n\n\n|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )

    search_text = block_match.group(1) if block_match else text

    # Extract fields
    fields: dict[str, str] = {}
    for field_name, pattern in _FIELD_PATTERNS.items():
        match = pattern.search(search_text)
        if match:
            fields[field_name] = match.group(1).strip()

    # Need at least ACTION to be a valid TRADE_BLOCK
    if "action" not in fields:
        return None

    # Clean action (strip markdown bold, parenthetical notes for classification)
    action_raw = fields["action"].strip("*").strip()
    # Normalize to base action for classification
    action_base = action_raw.split("(")[0].strip().upper()
    if action_base not in ("BUY", "SELL", "HOLD", "REJECT"):
        action_base = action_raw.upper()

    size_pct = 0.0
    if "size" in fields:
        try:
            size_pct = float(fields["size"])
        except ValueError:
            pass

    return TradeBlockData(
        action=action_base,
        size_pct=size_pct,
        conviction=fields.get("conviction", ""),
        entry_price=parse_price(fields.get("entry")),
        stop_price=parse_price(fields.get("stop")),
        target_1_price=parse_price(fields.get("target_1")),
        target_2_price=parse_price(fields.get("target_2")),
        risk_reward=fields.get("risk_reward", ""),
        special=fields.get("special", ""),
    )


# ══════════════════════════════════════════════════════════════════════════════
# IBKR Order Dict Builder
# ══════════════════════════════════════════════════════════════════════════════

# Lot sizes for exchanges that enforce board lots
BOARD_LOT_SIZES: dict[str, int] = {
    ".HK": 100,  # Hong Kong: varies per stock, 100 is common minimum
    ".T": 100,  # Japan: 100 shares per unit (since 2018 standardization)
    ".KS": 1,  # Korea: 1 share minimum
    ".TW": 1000,  # Taiwan TWSE: 1000 shares per board lot
    ".TWO": 1000,  # Taiwan OTC (same board lot as TWSE)
    ".SS": 100,  # Shanghai: 100 shares
    ".SZ": 100,  # Shenzhen: 100 shares
}


def round_to_lot_size(quantity: int, yf_ticker: str) -> int:
    """
    Round quantity down to nearest board lot for the exchange.

    Args:
        quantity: Desired number of shares
        yf_ticker: yfinance ticker (used to determine exchange)

    Returns:
        Quantity rounded down to valid lot size (minimum 0)
    """
    dot_idx = yf_ticker.rfind(".")
    suffix = yf_ticker[dot_idx:] if dot_idx >= 0 else ""
    lot_size = BOARD_LOT_SIZES.get(suffix, 1)
    return max(0, (quantity // lot_size) * lot_size)


def calculate_quantity(
    available_cash_usd: float,
    entry_price_local: float,
    fx_rate_to_usd: float | None,
    size_pct: float,
    portfolio_value_usd: float,
    yf_ticker: str,
) -> int:
    """
    Calculate order quantity from TRADE_BLOCK parameters.

    Uses the smaller of:
    1. Position size % of portfolio / entry price
    2. Available cash / entry price

    Args:
        available_cash_usd: Cash available for new buys (after buffer)
        entry_price_local: Entry price in local currency
        fx_rate_to_usd: FX rate (local currency → USD), None defaults to 1.0
        size_pct: Target position size as percentage of portfolio
        portfolio_value_usd: Total portfolio value in USD
        yf_ticker: Ticker for lot size rounding

    Returns:
        Number of shares (rounded to lot size)
    """
    if entry_price_local <= 0 or portfolio_value_usd <= 0:
        return 0

    fx_rate = fx_rate_to_usd or 1.0
    entry_price_usd = entry_price_local * fx_rate

    if entry_price_usd <= 0:
        return 0

    # Target allocation
    target_usd = portfolio_value_usd * (size_pct / 100.0)
    # Constrain to available cash
    deployable_usd = min(target_usd, available_cash_usd)

    if deployable_usd <= 0:
        return 0

    raw_qty = int(deployable_usd / entry_price_usd)
    return round_to_lot_size(raw_qty, yf_ticker)


def build_order_dict(
    conid: int,
    action: str,
    quantity: int,
    price: float | None = None,
    order_type: str = "LMT",
    tif: str = "GTC",
    account_id: str = "",
) -> dict:
    """
    Build an IBKR-compatible order dictionary for IBind's place_order().

    Args:
        conid: IBKR contract ID
        action: "BUY" or "SELL"
        quantity: Number of shares
        price: Limit price (required for LMT orders)
        order_type: "LMT" or "MKT"
        tif: Time in force — "GTC" (Good Til Cancel) or "DAY"
        account_id: IBKR account ID

    Returns:
        Dict ready for IBind's place_order()
    """
    order = {
        "conid": conid,
        "orderType": order_type,
        "side": action.upper(),
        "quantity": quantity,
        "tif": tif,
    }

    if account_id:
        order["acctId"] = account_id

    if order_type == "LMT" and price is not None:
        order["price"] = round(price, 2)

    return order
