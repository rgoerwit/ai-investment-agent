"""Shared monetary-token recognition; no FX conversion or field authority."""

from __future__ import annotations

import math
import re

from src.exchange_metadata import SUFFIX_TO_CURRENCY_CODE

# One lexical contract for monetary prose. Currency codes are not magnitudes:
# K in "100K KRW" scales the number; K in "100 KRW" does not.
_CURRENCY_CODES = frozenset(SUFFIX_TO_CURRENCY_CODE.values()) | {"USD"}
_CURRENCY_PATTERN = "|".join(sorted(_CURRENCY_CODES))
MONETARY_AMOUNT_PATTERN = (
    r"(?<![\w.+'’-])([+-]?)[\t ]*[$¥€£₩]?[\t ]*"
    r"(\d(?:[\d,]*\d)?(?:\.\d+)?)[\t ]*"
    r"((?:trillion|billion|million|thousand|tn|bn|mm|mn|[TBMK])\b)?"
    r"(?!\w|\.\d|\.\.|['’]\d|,\d)"
    rf"(?(3)|(?![\t ]+(?!(?:{_CURRENCY_PATTERN})\b)[^\W\d_]+))"
)
MONETARY_AMOUNT_RE = re.compile(MONETARY_AMOUNT_PATTERN, re.IGNORECASE)
_MONETARY_MAGNITUDES = {
    "T": 1e12,
    "TN": 1e12,
    "TRILLION": 1e12,
    "B": 1e9,
    "BN": 1e9,
    "BILLION": 1e9,
    "M": 1e6,
    "MM": 1e6,
    "MN": 1e6,
    "MILLION": 1e6,
    "K": 1e3,
    "THOUSAND": 1e3,
}


def parse_currency_value(sign: str, value_str: str, multiplier: str | None) -> float:
    """Scale a monetary token; currency conversion belongs to the FX layer."""
    value = float(value_str.replace(",", ""))
    if sign == "-":
        value = -value
    return value * (_MONETARY_MAGNITUDES[multiplier.upper()] if multiplier else 1.0)


def parse_monetary_amount(text: str | None) -> float | None:
    """Read a supported finite money token; callers own field/basis selection."""
    if not text or (match := MONETARY_AMOUNT_RE.search(text)) is None:
        return None
    number = match.group(2)
    if "," in number and not re.fullmatch(
        r"(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?", number
    ):
        return (
            None  # Western and Indian grouping; ambiguous decimal commas fail closed.
        )
    try:
        value = parse_currency_value(*match.groups())
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None
