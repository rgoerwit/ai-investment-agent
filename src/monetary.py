"""Shared monetary-token recognition; no FX conversion or field authority."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator

from src.exchange_metadata import (
    CURRENCY_CODE_ALIASES,
    CURRENCY_DISPLAY_FORMATS,
    CURRENCY_SYMBOL_TO_CODE,
    SUFFIX_TO_CURRENCY_CODE,
)
from src.fx_normalization import MINOR_UNIT_CURRENCY_ALIASES, canonical_currency_code

# One lexical contract for monetary prose. Currency codes are not magnitudes:
# K in "100K KRW" scales the number; K in "100 KRW" does not.
_CURRENCY_CODES = (
    frozenset(SUFFIX_TO_CURRENCY_CODE.values())
    | CURRENCY_DISPLAY_FORMATS.keys()
    | MINOR_UNIT_CURRENCY_ALIASES.keys()
)
_CURRENCY_PATTERN = "|".join(sorted(_CURRENCY_CODES | CURRENCY_CODE_ALIASES.keys()))
_SYMBOL_PATTERN = (
    "|".join(
        re.escape(symbol)
        for symbol in sorted(CURRENCY_SYMBOL_TO_CODE, key=len, reverse=True)
    )
    + r"|[$¥]"
)
_SYMBOL_CURRENCIES = {
    symbol.casefold(): code for symbol, code in CURRENCY_SYMBOL_TO_CODE.items()
}
_CURRENCY_TOKEN_PATTERN = rf"(?:{_CURRENCY_PATTERN})(?![A-Z])|(?:{_SYMBOL_PATTERN})"
MONETARY_AMOUNT_PATTERN = (
    r"(?<![\w.+'’(-])(?P<accounting>\()?(?P<sign>[+-])?[\t ]*"
    rf"(?:(?P<currency>{_CURRENCY_PATTERN})(?![A-Z])(?:[\t ]+|(?=\d))"
    rf"|(?P<symbol>{_SYMBOL_PATTERN}))?[\t ]*"
    r"(?(sign)|(?P<currency_sign>[+-])?)[\t ]*"
    r"(?P<number>\d(?:[\d,]*\d)?(?:\.\d+)?)[\t ]*"
    r"(?P<magnitude>(?:trillion|billion|million|thousand|tn|bn|mm|mn|[TBMK])\b)?"
    r"(?!\w|\.\d|\.\.|['’]\d|,\d)"
    rf"(?(magnitude)|(?![\t ]+(?!(?:{_CURRENCY_PATTERN})\b)[^\W\d_]+))"
    rf"(?:[\t ]+(?P<currency_suffix>{_CURRENCY_PATTERN})\b)?"
    r"(?(accounting)[\t ]*\)(?!\))|(?!\)))"
)
MONETARY_AMOUNT_RE = re.compile(MONETARY_AMOUNT_PATTERN, re.IGNORECASE)
_MONETARY_START_RE = re.compile(
    r"(?<![\w.+'’-])(?=\S)(?:[()+-][\t ]*)*"
    rf"(?:{_CURRENCY_TOKEN_PATTERN})?[\t ]*(?:[()+-][\t ]*)*\d",
    re.IGNORECASE,
)
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


def has_valid_monetary_grouping(number: str) -> bool:
    """Check comma grouping; callers retain their own numeric/precision contract."""
    return (
        "," not in number
        or re.fullmatch(
            r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?",
            number,
        )
        is not None
    )


def monetary_currency_codes(match: re.Match[str]) -> tuple[str, ...]:
    """Explicit token currencies, retaining contradictions for fail-closed callers."""
    codes = [match.group("currency"), match.group("currency_suffix")]
    symbol_code = _SYMBOL_CURRENCIES.get((match.group("symbol") or "").casefold())
    return tuple(
        canonical_currency_code(code) or code for code in [*codes, symbol_code] if code
    )


def iter_monetary_tokens(text: str) -> Iterator[re.Match[str]]:
    """Scan candidates once, never re-entering a malformed token's numeric interior."""
    position = 0
    while (start := _MONETARY_START_RE.search(text, position)) is not None:
        match = MONETARY_AMOUNT_RE.match(text, start.start())
        position = match.end() if match else start.end()
        if match is not None:
            if parse_monetary_amount(match.group(0)) is None:
                # Lexical matches may still have invalid grouping or conflicting
                # currencies. Do not substitute a later comparative for them.
                return
            yield match
        elif re.match(r"(?:19|20)\d{2}(?![\w.,])", text[start.start() :]) is None:
            # A leading bare year may introduce a labelled metric. A malformed
            # intended amount must not be replaced by its later comparative.
            return


def extract_monetary_currency(text: str | None) -> str | None:
    """Canonical explicit currency from a money token or a currency metadata field."""
    if not text:
        return None
    codes = {
        code
        for match in iter_monetary_tokens(text)
        for code in monetary_currency_codes(match)
    }
    if not codes:
        if _MONETARY_START_RE.search(text) is not None:
            # Invalid amounts cannot be reinterpreted as currency-only metadata.
            return None
        metadata = re.search(
            rf"(?<!\w)(?P<code>{_CURRENCY_PATTERN})\b|(?P<symbol>{_SYMBOL_PATTERN})",
            text,
            re.I,
        )
        if metadata:
            code = metadata.group("code")
            if code:
                codes.add(canonical_currency_code(code) or code)
            elif (
                symbol_code := _SYMBOL_CURRENCIES.get(
                    metadata.group("symbol").casefold()
                )
            ) is not None:
                codes.add(symbol_code)
    return next(iter(codes)) if len(codes) == 1 else None


def parse_monetary_amount(text: str | None) -> float | None:
    """Read a supported finite money token; callers own field/basis selection."""
    if not text or (start := _MONETARY_START_RE.search(text)) is None:
        return None
    # Do not skip malformed first amounts and silently substitute a later value.
    if (match := MONETARY_AMOUNT_RE.match(text, start.start())) is None:
        return None
    if len(set(monetary_currency_codes(match))) > 1:
        return None
    number = match.group("number")
    if not has_valid_monetary_grouping(number):
        return None  # Ambiguous decimal commas fail closed.
    try:
        value = parse_currency_value(
            "-"
            if match.group("accounting")
            else (match.group("sign") or match.group("currency_sign") or ""),
            number,
            match.group("magnitude"),
        )
    except (ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None
