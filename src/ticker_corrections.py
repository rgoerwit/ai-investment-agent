"""
Ticker Symbol Corrections Database

This module maintains a database of known ticker symbol corrections,
particularly for cases where different data providers use different
ticker abbreviations for the same security.

Common issues:
- Reuters uses abbreviated codes (e.g., "NOV" for Novartis)
- Actual trading symbols may differ (e.g., "NOVN" on SIX Swiss Exchange)
- IBKR may use different conventions than yfinance
"""

import structlog

from src.exchange_metadata import EXCHANGES_BY_SUFFIX, canonical_suffix_for_token

logger = structlog.get_logger(__name__)


# Known corrections: Reuters/Bloomberg codes to actual trading symbols
REUTERS_CORRECTIONS = {
    # Swiss Securities (SIX Swiss Exchange)
    "NOV.N-CH": ("NOVN", "SW", "Novartis AG"),
    "NOV.S-CH": ("NOVN", "SW", "Novartis AG"),
    "ROG.N-CH": ("ROG", "SW", "Roche Holding AG"),
    "ROG.S-CH": ("ROG", "SW", "Roche Holding AG"),
    "NESN.S-CH": ("NESN", "SW", "Nestlé S.A."),
    "NESN.N-CH": ("NESN", "SW", "Nestlé S.A."),
    "UBSG.S-CH": ("UBSG", "SW", "UBS Group AG"),
    "CSGN.S-CH": ("CSGN", "SW", "Credit Suisse Group AG"),
    "ZURN.S-CH": ("ZURN", "SW", "Zurich Insurance Group AG"),
    "ABB.N-CH": ("ABBN", "SW", "ABB Ltd"),
    "LONN.S-CH": ("LONN", "SW", "Lonza Group AG"),
    # German Securities (XETRA/Frankfurt)
    "SAPG.DE": ("SAP", "DE", "SAP SE"),
    "SIE.DE": ("SIE", "DE", "Siemens AG"),
    "DAI.DE": ("DAI", "DE", "Daimler AG"),
    "VOW3.DE": ("VOW3", "DE", "Volkswagen AG"),
    "BAYN.DE": ("BAYN", "DE", "Bayer AG"),
    # UK Securities (London Stock Exchange)
    "BP.L": ("BP", "L", "BP plc"),
    "HSBA.L": ("HSBA", "L", "HSBC Holdings plc"),
    "ULVR.L": ("ULVR", "L", "Unilever PLC"),
    "AZN.L": ("AZN", "L", "AstraZeneca PLC"),
    "GSK.L": ("GSK", "L", "GlaxoSmithKline plc"),
    # Japanese Securities (Tokyo Stock Exchange)
    "7203.T": ("7203", "T", "Toyota Motor Corporation"),
    "6758.T": ("6758", "T", "Sony Group Corporation"),
    "9984.T": ("9984", "T", "SoftBank Group Corp."),
    # Korean Securities (KOSDAQ tickers sometimes mistyped as KOSPI/.KS)
    "206640.KS": ("206640", "KQ", "Boditech Med Inc."),
}


# Alternative ticker formats that should be recognized
ALTERNATIVE_FORMATS = {
    # Format: alternative -> canonical
    "NOVN:SWX": "NOVN.SW",
    "NOVN:VX": "NOVN.SW",
    "NOVN.VX": "NOVN.SW",
    "ROG:SWX": "ROG.SW",
    "NESN:SWX": "NESN.SW",
    # UK — bare codes without .L suffix
    "MEGP": "MEGP.L",
}


KNOWN_VALID_TICKER_NAMES = {
    "NOVN.SW": "Novartis AG",
    "ROG.SW": "Roche Holding AG",
    "NESN.SW": "Nestlé S.A.",
    "UBSG.SW": "UBS Group AG",
    "CSGN.SW": "Credit Suisse Group AG",
    "ZURN.SW": "Zurich Insurance Group AG",
    "ABBN.SW": "ABB Ltd",
    "LONN.SW": "Lonza Group AG",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "NVDA": "NVIDIA Corporation",
    "SAP.DE": "SAP SE",
    "SIE.DE": "Siemens AG",
    "DAI.DE": "Daimler AG",
    "BP.L": "BP plc",
    "HSBA.L": "HSBC Holdings plc",
    "MEGP.L": "ME Group International plc",
    "7203.T": "Toyota Motor Corporation",
    "6758.T": "Sony Group Corporation",
    "206640.KQ": "Boditech Med Inc.",
}


def _derive_ticker_metadata(
    ticker: str,
    *,
    company_name: str | None = None,
) -> dict[str, str]:
    """Build ticker metadata from canonical exchange facts plus optional name."""
    suffix = ""
    symbol = ticker
    if "." in ticker:
        symbol, suffix_token = ticker.rsplit(".", 1)
        suffix = canonical_suffix_for_token(suffix_token) or ""

    if suffix:
        exchange = EXCHANGES_BY_SUFFIX[suffix]
        return {
            "name": company_name or ticker,
            "exchange": exchange.exchange_name,
            "country": exchange.country,
        }

    return {
        "name": company_name or ticker,
        "exchange": "NASDAQ",
        "country": "United States",
    }


class TickerCorrector:
    """Handles ticker symbol corrections and validations."""

    @classmethod
    def apply_correction(cls, ticker: str) -> tuple[str, bool, str | None]:
        """
        Apply known corrections to a ticker symbol.

        Args:
            ticker: Input ticker symbol

        Returns:
            Tuple of (corrected_ticker, was_corrected, company_name)
        """
        ticker = ticker.strip().upper()

        # Check Reuters corrections
        if ticker in REUTERS_CORRECTIONS:
            symbol, suffix, name = REUTERS_CORRECTIONS[ticker]
            corrected = f"{symbol}.{suffix}"
            if corrected == ticker:
                return corrected, False, name

            logger.info(
                "ticker_corrected",
                original=ticker,
                corrected=corrected,
                company=name,
                source="known_corrections",
            )

            return corrected, True, name

        # Check alternative formats
        if ticker in ALTERNATIVE_FORMATS:
            corrected = ALTERNATIVE_FORMATS[ticker]

            logger.info(
                "ticker_format_normalized",
                original=ticker,
                corrected=corrected,
                source="alternative_formats",
            )

            return corrected, True, None

        # No correction needed
        return ticker, False, None

    @classmethod
    def is_known_valid(cls, ticker: str) -> tuple[bool, dict[str, str] | None]:
        """
        Check if ticker is in the known valid list.

        Args:
            ticker: Ticker symbol to validate

        Returns:
            Tuple of (is_valid, ticker_info_dict)
        """
        ticker = ticker.strip().upper()

        company_name = KNOWN_VALID_TICKER_NAMES.get(ticker)
        if company_name is not None:
            return True, _derive_ticker_metadata(ticker, company_name=company_name)

        return False, None

    @classmethod
    def suggest_correction(cls, failed_ticker: str) -> str | None:
        """
        Suggest a correction for a failed ticker lookup.

        Args:
            failed_ticker: The ticker that failed validation

        Returns:
            Suggested ticker or None
        """
        failed_ticker = failed_ticker.strip().upper()

        # Check if there's a known correction
        for known, (symbol, suffix, name) in REUTERS_CORRECTIONS.items():
            if known.startswith(failed_ticker[:3]):
                suggested = f"{symbol}.{suffix}"
                logger.info(
                    "correction_suggested",
                    failed=failed_ticker,
                    suggested=suggested,
                    company=name,
                )
                return suggested

        # Check for partial matches in valid tickers
        for valid_ticker, company_name in KNOWN_VALID_TICKER_NAMES.items():
            if valid_ticker.startswith(failed_ticker[:3]):
                logger.info(
                    "correction_suggested",
                    failed=failed_ticker,
                    suggested=valid_ticker,
                    company=company_name,
                )
                return valid_ticker

        return None

    @classmethod
    def add_correction(
        cls,
        original: str,
        corrected_symbol: str,
        exchange_suffix: str,
        company_name: str,
    ):
        """
        Add a new correction to the database (runtime only).

        Args:
            original: Original ticker format
            corrected_symbol: Corrected symbol
            exchange_suffix: Exchange suffix (e.g., "SW")
            company_name: Company name
        """
        original = original.strip().upper()
        REUTERS_CORRECTIONS[original] = (
            corrected_symbol,
            exchange_suffix,
            company_name,
        )

        corrected_full = f"{corrected_symbol}.{exchange_suffix}"
        if corrected_full not in KNOWN_VALID_TICKER_NAMES:
            KNOWN_VALID_TICKER_NAMES[corrected_full] = company_name

        logger.info(
            "correction_added",
            original=original,
            corrected=corrected_full,
            company=company_name,
        )


# Convenience functions
def correct_ticker(ticker: str) -> str:
    """Apply corrections and return corrected ticker."""
    corrected, _, _ = TickerCorrector.apply_correction(ticker)
    return corrected


def is_valid_ticker(ticker: str) -> bool:
    """Check if ticker is known to be valid."""
    valid, _ = TickerCorrector.is_known_valid(ticker)
    return valid


def get_ticker_metadata(ticker: str) -> dict[str, str] | None:
    """Get metadata for a known ticker."""
    _, metadata = TickerCorrector.is_known_valid(ticker)
    return metadata


if __name__ == "__main__":
    # Test the correction system
    test_cases = [
        "NOV.N-CH",  # Should correct to NOVN.SW
        "NOVN.SW",  # Should be recognized as valid
        "AAPL",  # Should be recognized as valid
        "INVALID",  # Should return suggestion or None
    ]

    print("Testing Ticker Correction System\n")
    print("=" * 60)

    for ticker in test_cases:
        print(f"\nInput: {ticker}")

        # Apply correction
        corrected, was_corrected, name = TickerCorrector.apply_correction(ticker)
        print(f"Corrected: {corrected}")
        print(f"Was corrected: {was_corrected}")
        if name:
            print(f"Company: {name}")

        # Check validity
        valid, metadata = TickerCorrector.is_known_valid(corrected)
        print(f"Is valid: {valid}")
        if metadata:
            print(f"Metadata: {metadata}")

        # If not valid, suggest correction
        if not valid:
            suggestion = TickerCorrector.suggest_correction(ticker)
            if suggestion:
                print(f"Suggestion: {suggestion}")

        print("-" * 60)
