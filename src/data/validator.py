"""
Fine-Grained Data Validator

Validates specific aspects of financial data:
1. Basics (price, symbol, currency)
2. Valuation metrics (PE, PB, consistency)
3. Profitability metrics (margins, ROE/ROA)
4. Financial health (D/E, Current Ratio, Cash Flow)
5. Growth metrics (Revenue, Earnings)

Provides actionable feedback on what's missing and what's suspicious.
"""

import structlog
from typing import Any
from dataclasses import dataclass, field

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Detailed validation result."""
    category: str
    passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validated_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)


@dataclass
class OverallValidation:
    """Overall validation summary."""
    basics_ok: bool = False
    valuation_ok: bool = False
    profitability_ok: bool = False
    financial_health_ok: bool = False
    growth_ok: bool = False
    
    results: list[ValidationResult] = field(default_factory=list)
    
    @property
    def total_issues(self) -> int:
        return sum(len(r.issues) for r in self.results)
    
    @property
    def total_warnings(self) -> int:
        return sum(len(r.warnings) for r in self.results)
    
    @property
    def categories_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def categories_total(self) -> int:
        return len(self.results)


class FineGrainedValidator:
    """
    Fine-grained validator that checks specific categories of metrics.
    
    Design principles:
    1. Each category validated independently
    2. Actionable feedback (what's missing, what's wrong)
    3. Warnings vs errors (suspicious vs invalid)
    4. Never break on bad data, always provide diagnostics
    """
    
    def __init__(self):
        logger.info("fine_grained_validator_initialized")
    
    def _safe_float(self, value: Any) -> float | None:
        """Safely convert value to float, returning None if invalid."""
        try:
            if value is None:
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    def _validate_basics(self, data: dict, symbol: str) -> ValidationResult:
        """
        Validate basic required fields.
        
        CRITICAL: price, symbol, currency must be present and valid.
        """
        result = ValidationResult(category="basics", passed=False)
        
        # Check symbol
        data_symbol = data.get('symbol')
        if not data_symbol:
            result.issues.append("Missing 'symbol' field")
        elif str(data_symbol).upper() != symbol.upper():
            result.warnings.append(f"Symbol mismatch: requested '{symbol}', got '{data_symbol}'")
        else:
            result.validated_fields.append('symbol')
        
        # Check price (accept any price field)
        raw_price = data.get('currentPrice') or data.get('regularMarketPrice') or data.get('previousClose')
        price = self._safe_float(raw_price)
        
        if raw_price is None:
            result.issues.append("Missing price (currentPrice/regularMarketPrice/previousClose)")
        elif price is None:
            result.issues.append(f"Invalid price format: {raw_price}")
        elif price <= 0:
            result.issues.append(f"Invalid price: {price} (must be > 0)")
        elif price > 1000000:
            result.warnings.append(f"Unusual price: {price} (>$1M, check currency)")
        else:
            result.validated_fields.append('price')
        
        # Check currency
        currency = data.get('currency')
        if not currency:
            result.warnings.append("Missing 'currency' field (assuming USD)")
        else:
            result.validated_fields.append('currency')
        
        # Check previous close for change calculation
        raw_prev_close = data.get('previousClose')
        prev_close = self._safe_float(raw_prev_close)
        
        if prev_close:
            result.validated_fields.append('previousClose')
            
            if price and prev_close:
                pct_change = abs(price - prev_close) / prev_close
                if pct_change > 0.5:
                    result.warnings.append(
                        f"Large price change: {pct_change:.1%} "
                        "(possible split/error)"
                    )
        
        result.passed = len(result.issues) == 0
        return result
    
    def _validate_valuation(self, data: dict, symbol: str) -> ValidationResult:
        """
        Validate valuation metrics: PE, PB, PEG, Market Cap.
        
        Checks for presence, range validity, and internal consistency.
        """
        result = ValidationResult(category="valuation", passed=True)
        
        # P/E Ratio
        pe = self._safe_float(data.get('trailingPE'))
        if pe is None:
            result.missing_fields.append('trailingPE')
        elif pe < 0:
            result.warnings.append(f"Negative P/E: {pe:.2f} (company has losses)")
            result.validated_fields.append('trailingPE')
        elif pe > 1000:
            result.warnings.append(f"Extreme P/E: {pe:.2f} (check data quality)")
            result.validated_fields.append('trailingPE')
        else:
            result.validated_fields.append('trailingPE')
        
        # P/B Ratio
        pb = self._safe_float(data.get('priceToBook'))
        if pb is None:
            result.missing_fields.append('priceToBook')
        elif pb < 0:
            result.warnings.append(f"Negative P/B: {pb:.2f} (negative book value)")
            result.validated_fields.append('priceToBook')
        elif pb > 50:
            result.warnings.append(f"Extreme P/B: {pb:.2f} (check currency/data)")
            result.validated_fields.append('priceToBook')
        else:
            result.validated_fields.append('priceToBook')
        
        # PEG Ratio
        peg = self._safe_float(data.get('pegRatio'))
        if peg is None:
            result.missing_fields.append('pegRatio')
        elif peg < 0:
            result.warnings.append(f"Negative PEG: {peg:.2f}")
            result.validated_fields.append('pegRatio')
        else:
            result.validated_fields.append('pegRatio')
        
        # Market Cap
        market_cap = self._safe_float(data.get('marketCap'))
        if market_cap is None:
            result.missing_fields.append('marketCap')
        elif market_cap <= 0:
            result.issues.append(f"Invalid market cap: {market_cap}")
        else:
            result.validated_fields.append('marketCap')
        
        # Consistency: PE * EPS should \u2248 Price
        price = self._safe_float(data.get('currentPrice') or data.get('regularMarketPrice'))
        eps = self._safe_float(data.get('trailingEps'))
        
        if pe and eps and price and pe > 0:
            implied_price = pe * eps
            if abs(price - implied_price) / price > 0.1:  # 10% tolerance
                result.warnings.append(
                    f"PE/EPS inconsistency: P/E\u00d7EPS={implied_price:.2f} "
                    f"but price={price:.2f}"
                )
        
        result.passed = len(result.issues) == 0
        return result
    
    def _validate_profitability(self, data: dict, symbol: str) -> ValidationResult:
        """
        Validate profitability metrics: Margins, ROE, ROA.
        
        Checks for presence and valid ranges (typically 0-1).
        """
        result = ValidationResult(category="profitability", passed=True)
        
        margins = {
            'profitMargins': 'Profit Margin',
            'operatingMargins': 'Operating Margin',
            'grossMargins': 'Gross Margin',
        }
        
        for key, name in margins.items():
            value = self._safe_float(data.get(key))
            if value is None:
                result.missing_fields.append(key)
            elif value < -1 or value > 1:
                result.warnings.append(
                    f"{name} outside normal range: {value:.2%} "
                    "(expected -100% to 100%)"
                )
                result.validated_fields.append(key)
            else:
                result.validated_fields.append(key)
        
        # ROE
        roe = self._safe_float(data.get('returnOnEquity'))
        if roe is None:
            result.missing_fields.append('returnOnEquity')
        elif roe < -2 or roe > 2:
            result.warnings.append(
                f"ROE outside typical range: {roe:.2%} "
                "(expected -200% to 200%)"
            )
            result.validated_fields.append('returnOnEquity')
        else:
            result.validated_fields.append('returnOnEquity')
        
        # ROA
        roa = self._safe_float(data.get('returnOnAssets'))
        if roa is None:
            result.missing_fields.append('returnOnAssets')
        elif roa < -1 or roa > 1:
            result.warnings.append(
                f"ROA outside typical range: {roa:.2%} "
                "(expected -100% to 100%)"
            )
            result.validated_fields.append('returnOnAssets')
        else:
            result.validated_fields.append('returnOnAssets')
        
        # Consistency: ROE should be > ROA (leverage effect)
        if roe is not None and roa is not None and roa > 0:
            if roe < roa:
                result.warnings.append(
                    f"Unusual: ROE ({roe:.2%}) < ROA ({roa:.2%}) "
                    "(typically ROE > ROA)"
                )
        
        result.passed = len(result.issues) == 0
        return result
    
    def _validate_financial_health(self, data: dict, symbol: str) -> ValidationResult:
        """
        Validate financial health: D/E, Current Ratio, Cash Flow.
        
        Checks for leverage, liquidity, and cash generation.
        """
        result = ValidationResult(category="financial_health", passed=True)
        
        # Debt to Equity
        de = self._safe_float(data.get('debtToEquity'))
        if de is None:
            result.missing_fields.append('debtToEquity')
        elif de < 0:
            result.warnings.append(
                f"Negative D/E: {de:.2f} "
                "(negative equity or debt, check bank exception)"
            )
            result.validated_fields.append('debtToEquity')
        elif de > 10:
            result.warnings.append(
                f"Extreme leverage: D/E = {de:.2f} "
                "(very high debt relative to equity)"
            )
            result.validated_fields.append('debtToEquity')
        else:
            result.validated_fields.append('debtToEquity')
        
        # Current Ratio
        cr = self._safe_float(data.get('currentRatio'))
        if cr is None:
            result.missing_fields.append('currentRatio')
        elif cr < 0:
            result.issues.append(f"Invalid current ratio: {cr:.2f} (must be > 0)")
        elif cr < 0.5:
            result.warnings.append(
                f"Low current ratio: {cr:.2f} "
                "(potential liquidity concerns)"
            )
            result.validated_fields.append('currentRatio')
        else:
            result.validated_fields.append('currentRatio')
        
        # Quick Ratio
        qr = self._safe_float(data.get('quickRatio'))
        if qr is not None:
            if qr < 0:
                result.issues.append(f"Invalid quick ratio: {qr:.2f}")
            else:
                result.validated_fields.append('quickRatio')
        else:
            result.missing_fields.append('quickRatio')
        
        # Operating Cash Flow
        ocf = self._safe_float(data.get('operatingCashflow'))
        if ocf is None:
            result.missing_fields.append('operatingCashflow')
        else:
            result.validated_fields.append('operatingCashflow')
            
            # For non-financial companies, OCF should typically be positive
            # (Banks can have large negative OCF from operations)
            if ocf < 0 and not self._is_financial_company(data):
                result.warnings.append(
                    f"Negative operating cash flow: {ocf:,.0f} "
                    "(burning cash)"
                )
        
        # Free Cash Flow
        fcf = self._safe_float(data.get('freeCashflow'))
        if fcf is None:
            result.missing_fields.append('freeCashflow')
        else:
            result.validated_fields.append('freeCashflow')
            
            if fcf < 0 and not self._is_financial_company(data):
                result.warnings.append(
                    f"Negative free cash flow: {fcf:,.0f} "
                    "(FCF = OCF - CapEx)"
                )
        
        result.passed = len(result.issues) == 0
        return result
    
    def _validate_growth(self, data: dict, symbol: str) -> ValidationResult:
        """
        Validate growth metrics: Revenue Growth, Earnings Growth.
        
        Checks for presence and reasonable ranges.
        """
        result = ValidationResult(category="growth", passed=True)
        
        # Revenue Growth
        rev_growth = self._safe_float(data.get('revenueGrowth'))
        if rev_growth is None:
            result.missing_fields.append('revenueGrowth')
        elif rev_growth < -0.9 or rev_growth > 10:
            result.warnings.append(
                f"Extreme revenue growth: {rev_growth:.1%} "
                "(check data quality)"
            )
            result.validated_fields.append('revenueGrowth')
        else:
            result.validated_fields.append('revenueGrowth')
        
        # Earnings Growth
        earnings_growth = self._safe_float(data.get('earningsGrowth'))
        if earnings_growth is None:
            result.missing_fields.append('earningsGrowth')
        elif earnings_growth < -2 or earnings_growth > 20:
            result.warnings.append(
                f"Extreme earnings growth: {earnings_growth:.1%} "
                "(high volatility or data issue)"
            )
            result.validated_fields.append('earningsGrowth')
        else:
            result.validated_fields.append('earningsGrowth')
        
        result.passed = len(result.issues) == 0
        return result
    
    def _is_financial_company(self, data: dict) -> bool:
        """Check if company is in financial sector (banks have different metrics)."""
        industry = str(data.get('industry', '')).lower()
        sector = str(data.get('sector', '')).lower()
        
        financial_keywords = ['bank', 'financial', 'insurance', 'capital markets']
        
        return any(kw in industry or kw in sector for kw in financial_keywords)
    
    def validate_comprehensive(self, data: dict, symbol: str) -> OverallValidation:
        """
        Comprehensive validation across all categories.
        
        Returns OverallValidation with detailed results for each category.
        """
        overall = OverallValidation()
        
        # Validate each category
        basics = self._validate_basics(data, symbol)
        valuation = self._validate_valuation(data, symbol)
        profitability = self._validate_profitability(data, symbol)
        financial_health = self._validate_financial_health(data, symbol)
        growth = self._validate_growth(data, symbol)
        
        # Set flags
        overall.basics_ok = basics.passed
        overall.valuation_ok = valuation.passed
        overall.profitability_ok = profitability.passed
        overall.financial_health_ok = financial_health.passed
        overall.growth_ok = growth.passed
        
        # Store results
        overall.results = [
            basics,
            valuation,
            profitability,
            financial_health,
            growth,
        ]
        
        # Log summary
        logger.info("validation_complete",
                   symbol=symbol,
                   categories_passed=f"{overall.categories_passed}/{overall.categories_total}",
                   issues=overall.total_issues,
                   warnings=overall.total_warnings)
        
        return overall
    
    def get_validation_summary(self, validation: OverallValidation) -> str:
        """Get human-readable validation summary."""
        lines = [
            f"Validation Summary: {validation.categories_passed}/{validation.categories_total} categories passed",
            f"Total Issues: {validation.total_issues}",
            f"Total Warnings: {validation.total_warnings}",
            "",
        ]
        
        for result in validation.results:
            status = "✓" if result.passed else "✗"
            lines.append(f"{status} {result.category.upper()}")
            
            if result.validated_fields:
                lines.append(f"  Validated: {', '.join(result.validated_fields)}")
            
            if result.missing_fields:
                lines.append(f"  Missing: {', '.join(result.missing_fields)}")
            
            if result.issues:
                for issue in result.issues:
                    lines.append(f"  ⚠ ISSUE: {issue}")
            
            if result.warnings:
                for warning in result.warnings:
                    lines.append(f"  ⚡ WARNING: {warning}")
            
            lines.append("")
        
        return "\n".join(lines)


# Singleton instance
validator = FineGrainedValidator()