"""Live-order matching and operator-facing order annotations."""

from __future__ import annotations

from dataclasses import dataclass

from src.ibkr.models import ReconciliationItem

_TERMINAL_ORDER_STATUSES = frozenset({"cancelled", "inactive", "filled"})


@dataclass(frozen=True)
class LiveOrderMatch:
    order: dict
    side: str
    quantity: int | None
    price: float | str | None
    order_type: str
    status: str


def _build_live_order_match(order: dict) -> LiveOrderMatch:
    raw_quantity = order.get("remainingSize") or order.get("totalSize")
    try:
        quantity = int(raw_quantity) if raw_quantity is not None else None
    except (TypeError, ValueError):
        quantity = None
    side = "SELL" if str(order.get("side", "")).upper() in {"S", "SELL"} else "BUY"
    return LiveOrderMatch(
        order=order,
        side=side,
        quantity=quantity,
        price=order.get("price") or order.get("auxPrice"),
        order_type=str(order.get("orderType") or "LMT"),
        status=str(order.get("status") or ""),
    )


def find_live_order(
    item: ReconciliationItem,
    live_orders: list[dict] | None,
) -> LiveOrderMatch | None:
    """Return an open match, or a filled match only as historical fallback."""
    if not live_orders:
        return None

    pos = item.ibkr_position
    conid = pos.conid if pos else None
    symbol_candidates = {item.ticker.ibkr.upper(), item.ticker.yf.split(".")[0].upper()}
    if pos and pos.symbol:
        symbol_candidates.add(pos.symbol.upper())

    filled_fallback: LiveOrderMatch | None = None
    for order in live_orders:
        matched = False
        order_conid = order.get("conid")
        order_symbol = (order.get("ticker") or order.get("symbol") or "").upper()
        if conid and order_conid is not None:
            try:
                if int(order_conid) != int(conid):
                    continue
                matched = True
            except (TypeError, ValueError):
                matched = False
        if not matched and order_symbol in symbol_candidates:
            matched = True
        if not matched:
            continue

        status = str(order.get("status") or "").strip().lower()
        if status in _TERMINAL_ORDER_STATUSES:
            if status == "filled" and filled_fallback is None:
                filled_fallback = _build_live_order_match(order)
            continue
        return _build_live_order_match(order)
    return filled_fallback


def build_live_order_note(
    item: ReconciliationItem,
    live_orders: list[dict] | None,
) -> str | None:
    match = find_live_order(item, live_orders)
    if match is None:
        return None

    if isinstance(match.price, int | float):
        price_str = f" @ {float(match.price):.2f}"
    elif match.price:
        price_str = f" @ {match.price}"
    else:
        price_str = ""

    rec_side = "SELL" if item.action in {"SELL", "TRIM"} else "BUY"
    display_qty = match.quantity if match.quantity is not None else "?"
    if match.status.strip().lower() == "filled":
        return (
            f"[ORDER FILLED: {match.side} {display_qty}{price_str} {match.order_type}]"
        )

    if item.action == "REVIEW":
        return (
            f"[OPEN ORDER REVIEW: live {match.side} order {display_qty}{price_str}"
            f" {match.order_type} ({match.status}) exists while the position is under"
            " review - inspect or cancel it before acting]"
        )

    if match.side == rec_side:
        rec_qty = item.suggested_quantity
        if (
            match.quantity is not None
            and rec_qty is not None
            and match.quantity < rec_qty
        ):
            need = rec_qty - match.quantity
            return (
                f"[PARTIAL ORDER: {match.quantity} of {rec_qty} shares already submitted"
                f" — enter {need} more]"
            )
        return (
            f"[ORDER ALREADY SUBMITTED: {match.side} {display_qty}{price_str}"
            f" {match.order_type} ({match.status}) — do not re-enter]"
        )
    return (
        f"[CONFLICT: live {match.side} order {display_qty}{price_str}"
        f" {match.order_type} ({match.status}) while recommending {rec_side}]"
    )


def base_ticker(item: ReconciliationItem) -> str:
    return base_ticker_value(item.ticker.yf)


def base_ticker_value(ticker: str) -> str:
    return ticker.split(".")[0].upper()
