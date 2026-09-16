"""Structural tripwire for raw broker-symbol admission boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BROKER_IDENTITY_FIELDS = {"contractDesc", "listingExchange", "primaryExch"}
_APPROVED_READERS = {
    ("src/ibkr/portfolio.py", "normalize_positions"): {"_resolve_position_identity"},
    ("src/ibkr/portfolio.py", "_resolve_conid_ticker"): {
        "classify_ibkr_symbol",
        "resolve_ibkr_ticker",
    },
    ("src/ibkr/ticker_mapper.py", "resolve_yf_ticker_from_position"): {
        "ibkr_symbol_to_yf"
    },
    ("src/ibkr/security_data_service.py", "_probe_security_sync"): {
        "ibkr_symbol_to_yf"
    },
}
_FORBIDDEN_READER_CALLS = {"from_ibkr", "from_yf"}


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _reader_inventory(sources: dict[str, str]) -> dict[tuple[str, str], set[str]]:
    inventory: dict[tuple[str, str], set[str]] = {}
    for path, source in sources.items():
        tree = ast.parse(source, filename=path)
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ):
            calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
            reads_identity = any(
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "get"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value in _BROKER_IDENTITY_FIELDS
                for call in calls
            ) or any(
                isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in _BROKER_IDENTITY_FIELDS
                for node in ast.walk(function)
            )
            if reads_identity:
                inventory[(path, function.name)] = {
                    name for call in calls if (name := _call_name(call)) is not None
                }
    return inventory


def _assert_approved_ingress(sources: dict[str, str]) -> None:
    inventory = _reader_inventory(sources)
    assert set(inventory) == set(_APPROVED_READERS)
    for reader, required_calls in _APPROVED_READERS.items():
        assert required_calls <= inventory[reader]
        assert not (_FORBIDDEN_READER_CALLS & inventory[reader])


def _ibkr_sources() -> dict[str, str]:
    return {
        str(path.relative_to(_ROOT)): path.read_text(encoding="utf-8")
        for path in sorted((_ROOT / "src" / "ibkr").rglob("*.py"))
    }


def test_raw_broker_identity_readers_route_through_approved_boundaries() -> None:
    _assert_approved_ingress(_ibkr_sources())


@pytest.mark.parametrize(
    "symbol, exchange",
    [
        ('raw.get("contractDesc", "")', 'raw.get("listingExchange", "")'),
        ('raw["contractDesc"]', 'raw["listingExchange"]'),
    ],
)
def test_planted_direct_ticker_construction_is_rejected(symbol, exchange) -> None:
    planted = _ibkr_sources()
    planted["src/ibkr/_bypass_probe.py"] = f"""
def bypass(raw):
    return Ticker.from_ibkr({symbol}, {exchange})
"""

    with pytest.raises(AssertionError):
        _assert_approved_ingress(planted)


def test_approved_reader_cannot_hide_direct_construction_beside_router() -> None:
    planted = _ibkr_sources()
    portfolio_path = "src/ibkr/portfolio.py"
    planted[portfolio_path] = planted[portfolio_path].replace(
        "outcome = _resolve_position_identity(",
        "Ticker.from_ibkr(symbol, exchange, currency)\n        "
        "outcome = _resolve_position_identity(",
        1,
    )

    with pytest.raises(AssertionError):
        _assert_approved_ingress(planted)
