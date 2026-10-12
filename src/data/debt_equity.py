"""Single owner of the ``debtToEquity`` unit contract.

Canonical contract: ``debtToEquity`` is a RATIO of total debt to book equity
(0.45 means debt is 45% of equity). Every provider reports it differently, so
each source's observation is converted to a ratio at the source boundary, before
the multi-source merge compares values. Magnitude guessing downstream (is 8.05 a
ratio or 8.05%?) is not a substitute: the unit is a property of the source.

Verified sources (unit and basis) are listed in ``DE_SOURCE_SPECS``. A source whose
unit *and* basis are not verified contributes nothing; a statement-derived ratio or
a verified source fills the field instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import structlog

logger = structlog.get_logger(__name__)

FIELD = "debtToEquity"
UNIT_TAG = "_debtToEquity_unit"
INPUT_UNIT_TAG = "_debtToEquity_input_unit"
SOURCE_TAG = "_debtToEquity_source"
STATEMENT_SOURCE = "calculated_from_statements"
CROSS_CHECK_SOURCE = "calculated_from_total_debt_book_equity"

# A percent-vs-ratio mislabel disagrees by a factor of 100. Allow noise (share-class
# and period differences) but stay far from 1x so a genuine small disagreement is
# never "corrected".
_UNIT_CONFLICT_LOW = 80.0
_UNIT_CONFLICT_HIGH = 125.0


@dataclass(frozen=True)
class DebtEquitySourceSpec:
    """Verified unit and basis of one source's ``debtToEquity``."""

    unit: Literal["ratio", "percent"] | None
    basis: str
    note: str


DE_SOURCE_SPECS: dict[str, DebtEquitySourceSpec] = {
    # Yahoo's key-statistics "Total Debt/Equity (mrq)" in percent: TOT.TO reported
    # 8.047 against total debt 51.2M and book equity 634M (= 8.07%).
    "yfinance": DebtEquitySourceSpec("percent", "total_debt/equity", "Yahoo info"),
    "yahooquery": DebtEquitySourceSpec("percent", "total_debt/equity", "Yahoo info"),
    # Unverified: FMP's published ratio definition is total liabilities / equity, a
    # different basis from total debt / equity, and its stable-API field basis could
    # not be confirmed against primary documentation.
    "fmp": DebtEquitySourceSpec(None, "unverified", "basis may be liabilities/equity"),
    # Unverified: Alpha Vantage OVERVIEW documents no unit or basis for DebtToEquity.
    "alpha_vantage": DebtEquitySourceSpec(None, "unverified", "undocumented"),
    # Unverified: EODHD documents no definition for Highlights.DebtToEquity.
    "eodhd": DebtEquitySourceSpec(None, "unverified", "undocumented"),
    # The IBKR snapshot source supplies no debt/equity field at all.
}


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def to_ratio(
    value: Any, source: str, *, provenance: str | None = None
) -> tuple[float | None, str | None]:
    """Convert one source's observation to a ratio.

    Returns ``(ratio, input_unit)``, or ``(None, None)`` when the value cannot be
    trusted: non-finite, negative (negative book equity makes the ratio
    meaningless), or from a source whose unit and basis are unverified. A value
    carrying statement provenance is already a ratio whatever the source.
    """
    number = _finite(value)
    if number is None or number < 0:
        return None, None
    if provenance == STATEMENT_SOURCE:
        return number, "ratio"
    spec = DE_SOURCE_SPECS.get(source)
    if spec is None or spec.unit is None:
        return None, None
    if spec.unit == "percent":
        return number / 100.0, "percent"
    return number, "ratio"


def normalize_source_results(
    source_results: dict[str, dict | None],
) -> dict[str, dict | None]:
    """Return source results with every ``debtToEquity`` expressed as a ratio.

    Idempotent: an observation already tagged ``ratio`` is validated, not rescaled.
    Dicts without the field are returned untouched (same object).
    """
    out: dict[str, dict | None] = {}
    for source, data in source_results.items():
        if not data or FIELD not in data:
            out[source] = data
            continue
        raw = data[FIELD]
        if raw is None:
            out[source] = data
            continue
        updated = dict(data)
        if data.get(UNIT_TAG) == "ratio":
            ratio = _finite(raw)
            ratio = ratio if ratio is not None and ratio >= 0 else None
            input_unit = data.get(INPUT_UNIT_TAG, "ratio")
        else:
            provenance = data.get(SOURCE_TAG)
            ratio, input_unit = to_ratio(
                raw,
                source,
                provenance=provenance if isinstance(provenance, str) else None,
            )
        if ratio is None:
            spec = DE_SOURCE_SPECS.get(source)
            logger.info(
                "debt_equity_observation_dropped",
                source=source,
                reason="unverified_source"
                if spec is None or spec.unit is None
                else "invalid_value",
            )
            updated.pop(FIELD, None)
            updated.pop(SOURCE_TAG, None)
            updated.pop(UNIT_TAG, None)
            updated.pop(INPUT_UNIT_TAG, None)
        else:
            updated[FIELD] = ratio
            updated[UNIT_TAG] = "ratio"
            updated[INPUT_UNIT_TAG] = input_unit
        out[source] = updated
    return out


def cross_check(info: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Cross-check the merged ratio against totalDebt / book equity.

    Book equity is ``bookValue * sharesOutstanding`` (both in the financial
    currency until the currency-mismatch repair runs, so call this before it). A
    disagreement of about x100 is a unit error: log it and adopt the derived ratio.
    Anything else is left alone, and so is a non-positive equity.
    """
    ratio = _finite(info.get(FIELD))
    if ratio is None or ratio <= 0:
        return info
    if info.get(SOURCE_TAG) in (STATEMENT_SOURCE, CROSS_CHECK_SOURCE):
        return info
    debt = _finite(info.get("totalDebt"))
    book = _finite(info.get("bookValue"))
    shares = _finite(info.get("sharesOutstanding"))
    if debt is None or debt <= 0 or book is None or book <= 0:
        return info
    if shares is None or shares <= 0:
        return info
    derived = debt / (book * shares)
    if derived <= 0 or not math.isfinite(derived):
        return info
    factor = ratio / derived
    if _UNIT_CONFLICT_LOW <= factor <= _UNIT_CONFLICT_HIGH or (
        1.0 / _UNIT_CONFLICT_HIGH <= factor <= 1.0 / _UNIT_CONFLICT_LOW
    ):
        logger.warning(
            "debt_equity_unit_conflict",
            symbol=symbol,
            merged_ratio=round(ratio, 6),
            derived_ratio=round(derived, 6),
            factor=round(factor, 3),
        )
        info[FIELD] = derived
        info[SOURCE_TAG] = CROSS_CHECK_SOURCE
        info[UNIT_TAG] = "ratio"
    return info
