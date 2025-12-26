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
    "BP.L": ("BP.", "L", "BP plc"),
    "HSBA.L": ("HSBA", "L", "HSBC Holdings plc"),
    "ULVR.L": ("ULVR", "L", "Unilever PLC"),
    "AZN.L": ("AZN", "L", "AstraZeneca PLC"),
    "GSK.L": ("GSK", "L", "GlaxoSmithKline plc"),
    
    # Japanese Securities (Tokyo Stock Exchange)
    "7203.T": ("7203", "T", "Toyota Motor Corporation"),
    "6758.T": ("6758", "T", "Sony Group Corporation"),
    "9984.T": ("9984", "T", "SoftBank Group Corp."),
}


# Alternative ticker formats that should be recognized
ALTERNATIVE_FORMATS = {
    # Format: alternative -> canonical
    "NOVN:SWX": "NOVN.SW",
    "NOVN:VX": "NOVN.SW",
    "NOVN.VX": "NOVN.SW",
    "ROG:SWX": "ROG.SW",
    "NESN:SWX": "NESN.SW",
}


# Known valid tickers for quick validation
KNOWN_VALID_TICKERS = {
    # Swiss Securities
    "NOVN.SW": {"name": "Novartis AG", "exchange": "SIX Swiss Exchange", "country": "Switzerland"},
    "ROG.SW": {"name": "Roche Holding AG", "exchange": "SIX Swiss Exchange", "country": "Switzerland"},
    "NESN.SW": {"name": "Nestlé S.A.", "exchange": "SIX Swiss Exchange", "country": "Switzerland"},
    "UBSG.SW": {"name": "UBS Group AG", "exchange": "SIX Swiss Exchange", "country": "Switzerland"},
    "CSGN.SW": {"name": "Credit Suisse Group AG", "exchange": "SIX Swiss Exchange", "country": "Switzerland"},
    "ZURN.SW": {"name": "Zurich Insurance Group AG", "exchange": "SIX Swiss Exchange", "country": "Switzerland"},
    "ABBN.SW": {"name": "ABB Ltd", "exchange": "SIX Swiss Exchange", "country": "Switzerland"},
    "LONN.SW": {"name": "Lonza Group AG", "exchange": "SIX Swiss Exchange", "country": "Switzerland"},
    
    # US Securities
    "AAPL": {"name": "Apple Inc.", "exchange": "NASDAQ", "country": "United States"},
    "MSFT": {"name": "Microsoft Corporation", "exchange": "NASDAQ", "country": "United States"},
    "GOOGL": {"name": "Alphabet Inc.", "exchange": "NASDAQ", "country": "United States"},
    "AMZN": {"name": "Amazon.com Inc.", "exchange": "NASDAQ", "country": "United States"},
    "TSLA": {"name": "Tesla Inc.", "exchange": "NASDAQ", "country": "United States"},
    "META": {"name": "Meta Platforms Inc.", "exchange": "NASDAQ", "country": "United States"},
    "NVDA": {"name": "NVIDIA Corporation", "exchange": "NASDAQ", "country": "United States"},
    
    # German Securities
    "SAP.DE": {"name": "SAP SE", "exchange": "XETRA", "country": "Germany"},
    "SIE.DE": {"name": "Siemens AG", "exchange": "XETRA", "country": "Germany"},
    "DAI.DE": {"name": "Daimler AG", "exchange": "XETRA", "country": "Germany"},
    
    # UK Securities
    "BP.L": {"name": "BP plc", "exchange": "London Stock Exchange", "country": "United Kingdom"},
    "HSBA.L": {"name": "HSBC Holdings plc", "exchange": "London Stock Exchange", "country": "United Kingdom"},
    
    # Japanese Securities
    "7203.T": {"name": "Toyota Motor Corporation", "exchange": "Tokyo Stock Exchange", "country": "Japan"},
    "6758.T": {"name": "Sony Group Corporation", "exchange": "Tokyo Stock Exchange", "country": "Japan"},
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
            
            logger.info("ticker_corrected",
                       original=ticker,
                       corrected=corrected,
                       company=name,
                       source="reuters_database")
            
            return corrected, True, name
        
        # Check alternative formats
        if ticker in ALTERNATIVE_FORMATS:
            corrected = ALTERNATIVE_FORMATS[ticker]
            
            logger.info("ticker_format_normalized",
                       original=ticker,
                       corrected=corrected,
                       source="alternative_formats")
            
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
        
        if ticker in KNOWN_VALID_TICKERS:
            return True, KNOWN_VALID_TICKERS[ticker]
        
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
                logger.info("correction_suggested",
                           failed=failed_ticker,
                           suggested=suggested,
                           company=name)
                return suggested
        
        # Check for partial matches in valid tickers
        for valid_ticker, info in KNOWN_VALID_TICKERS.items():
            if valid_ticker.startswith(failed_ticker[:3]):
                logger.info("correction_suggested",
                           failed=failed_ticker,
                           suggested=valid_ticker,
                           company=info["name"])
                return valid_ticker
        
        return None
    
    @classmethod
    def add_correction(cls, original: str, corrected_symbol: str, 
                       exchange_suffix: str, company_name: str):
        """
        Add a new correction to the database (runtime only).
        
        Args:
            original: Original ticker format
            corrected_symbol: Corrected symbol
            exchange_suffix: Exchange suffix (e.g., "SW")
            company_name: Company name
        """
        original = original.strip().upper()
        REUTERS_CORRECTIONS[original] = (corrected_symbol, exchange_suffix, company_name)
        
        corrected_full = f"{corrected_symbol}.{exchange_suffix}"
        if corrected_full not in KNOWN_VALID_TICKERS:
            KNOWN_VALID_TICKERS[corrected_full] = {
                "name": company_name,
                "exchange": "Unknown",
                "country": "Unknown"
            }
        
        logger.info("correction_added",
                   original=original,
                   corrected=corrected_full,
                   company=company_name)


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
        "NOVN.SW",   # Should be recognized as valid
        "AAPL",      # Should be recognized as valid
        "INVALID",   # Should return suggestion or None
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