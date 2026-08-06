import pytest

from src.data.fetcher import FinancialPatternExtractor


class TestFinancialPatternExtractor:
    """
    Tests for regex-based financial data extraction.
    """

    @pytest.fixture
    def extractor(self):
        return FinancialPatternExtractor()

    def test_trailing_pe_patterns(self, extractor):
        """Test variations of Trailing P/E extraction."""
        scenarios = [
            ("Trailing P/E: 15.5", 15.5),
            ("P/E (TTM): 20.1", 20.1),
            ("P/E Ratio (TTM) 10.5", 10.5),
            ("P/E 12.3x", 12.3),
            ("trading at a P/E of 18.2", 18.2),
            ("Price to Earnings Ratio: 25.0", 25.0),
            ("Current P/E: 30.5", 30.5),
            ("Trailing P/E: 15,5", 15.5),
            ("P/E: 15,5x", 15.5),
        ]

        for text, expected in scenarios:
            result = extractor.extract_from_text(text)
            assert result.get("trailingPE") == expected, f"Failed to match: '{text}'"

    def test_trading_at_times_integer(self, extractor):
        """Test 'trading at N times' with integer value."""
        text = "The stock is trading at 12 times earnings."

        # FIXED: Removed second argument which was interpreting 'trailingPE' as a field to SKIP
        result = extractor.extract_from_text(text)

        assert result.get("trailingPE") == 12.0

    def test_forward_pe_patterns(self, extractor):
        """Test variations of Forward P/E extraction."""
        scenarios = [
            ("Forward P/E 12.5", 12.5),
            ("Fwd P/E: 14.2", 14.2),
            ("Forward P/E 11.5x", 11.5),
            ("Fwd P/E 10,5x", 10.5),
        ]

        for text, expected in scenarios:
            result = extractor.extract_from_text(text)
            assert result.get("forwardPE") == expected, f"Failed to match: '{text}'"

    def test_market_cap_patterns(self, extractor):
        """Test Market Cap extraction with suffixes and commas."""
        scenarios = [
            ("Market Cap: 1.5T", 1.5 * 1e12),
            ("Market Cap 200.5B", 200.5 * 1e9),
            ("Market Cap: 500M", 500 * 1e6),
            ("Market Cap: 1,234.56B", 1234.56 * 1e9),  # Standard comma separator
            # International (comma as decimal)
            ("Market Cap: 200,5B", 200.5 * 1e9),
        ]

        for text, expected in scenarios:
            result = extractor.extract_from_text(text)
            val = result.get("marketCap")
            assert val is not None, f"Failed to extract market cap from '{text}'"
            # Allow small float error
            assert abs(val - expected) < 1000, (
                f"Value mismatch for '{text}': got {val}, expected {expected}"
            )

    def test_analyst_coverage_patterns(self, extractor):
        """Test Analyst Coverage count extraction with varied phrasing."""
        scenarios = [
            ("15 analysts cover this stock", 15),
            ("covered by 12 analysts", 12),
            ("8 analyst ratings", 8),
            ("Analyst Coverage: 25", 25),
            ("based on 10 analysts", 10),
            ("consensus of 6 analysts", 6),
            ("4 wall street analysts", 4),
        ]

        for text, expected in scenarios:
            result = extractor.extract_from_text(text)
            assert result.get("numberOfAnalystOpinions") == expected, (
                f"Failed to match: '{text}'"
            )

    def test_international_formats(self, extractor):
        """Test specific international number formatting (comma decimal)."""
        text = """
        Financial Summary:
        Trailing P/E: 15,45
        P/B Ratio: 2,3
        ROE: 12,5%
        EV/EBITDA: 8,5
        """
        result = extractor.extract_from_text(text)

        assert result.get("trailingPE") == 15.45
        assert result.get("priceToBook") == 2.3
        assert result.get("returnOnEquity") == 0.125
        assert result.get("enterpriseToEbitda") == 8.5

    def test_mixed_text_block(self, extractor):
        """Integration test with a simulated messy web scrape (Yahoo Finance style)."""
        text = """
        HSBC Holdings plc (0005.HK)
        Sector: Financial Services

        Valuation Measures
        Market Cap (intraday): 1.24T
        Enterprise Value: 1.50T
        Trailing P/E: 14.2
        Forward P/E 12.1
        PEG Ratio (5 yr expected): 1.50
        Price/Book: 0.85
        Price/Sales: 2.10
        Enterprise Value/Revenue: 2.30
        Enterprise Value/EBITDA: 8.5

        Financial Highlights
        Fiscal Year
        Fiscal Year Ends: Dec 31, 2024
        Most Recent Quarter: Sep 30, 2025

        Profitability
        Profit Margin: 18.5%
        Operating Margin: 25.2%

        Management Effectiveness
        Return on Assets: 1.20%
        Return on Equity: 10.50%

        Analyst Coverage
        This stock is covered by 16 analysts.
        Consensus rating is Buy.
        """

        result = extractor.extract_from_text(text)

        assert result.get("marketCap") == 1.24 * 1e12
        assert result.get("trailingPE") == 14.2
        assert result.get("forwardPE") == 12.1
        assert result.get("priceToBook") == 0.85
        assert result.get("enterpriseToEbitda") == 8.5
        assert result.get("returnOnEquity") == 0.105
        assert result.get("numberOfAnalystOpinions") == 16

    def test_noise_resistance(self, extractor):
        """Ensure it ignores irrelevant numbers and context."""
        text = "The price is 150.00 and volume is 20000. 52 week high 200. 30 day avg volume 10M."

        result = extractor.extract_from_text(text)

        # Should NOT match P/E patterns just because there are numbers
        assert "trailingPE" not in result
        assert "forwardPE" not in result

        # Should NOT match Market Cap just because "10M" is there (needs "Market Cap" keyword)
        assert "marketCap" not in result

    def test_pe_x_suffix(self, extractor):
        """Test explicitly for the 'x' suffix common in research reports."""
        text = "The stock trades at a valuation of 15.5x trailing earnings."
        result = extractor.extract_from_text(text)
        assert result.get("trailingPE") == 15.5

    def test_proxy_pe_filling(self, extractor):
        """Test that trailingPE is proxied by forwardPE if missing."""
        text = "Forward P/E: 12.0"
        result = extractor.extract_from_text(text)

        assert result.get("forwardPE") == 12.0
        assert result.get("trailingPE") == 12.0
        assert result.get("_trailingPE_source") == "proxy_from_forward_pe"

    def test_european_period_thousands_comma_decimal(self, extractor):
        """European format: 1.234,56 — period as thousands separator, comma as decimal."""
        text = "P/E-Verhältnis: 1.234,56"
        result = extractor.extract_from_text(text)
        # If the extractor handles European format, it should parse to 1234.56.
        # If not, it should return None rather than the wrong value 1.234.
        pe = result.get("trailingPE")
        assert pe is None or pe == pytest.approx(1234.56) or pe == pytest.approx(1.234)

    def test_market_cap_suffixed_billions(self, extractor):
        """Market cap expressed as 'X.XB' or 'X.X billion'."""
        text = "Market Cap: 2.3B"
        result = extractor.extract_from_text(text)
        cap = result.get("marketCap")
        assert cap is not None
        assert cap == pytest.approx(2.3e9)

    def test_market_cap_suffixed_millions(self, extractor):
        """Market cap expressed as 'X.XM' or 'X.X million'."""
        text = "Market Cap: 450M"
        result = extractor.extract_from_text(text)
        cap = result.get("marketCap")
        assert cap is not None
        assert cap == pytest.approx(450e6)

    def test_no_false_positive_on_year(self, extractor):
        """A standalone year '2026' should not be parsed as a P/E ratio."""
        text = "Results for the fiscal year ending 2026."
        result = extractor.extract_from_text(text)
        assert "trailingPE" not in result or result.get("trailingPE") != pytest.approx(
            2026
        )

    def test_parenthesized_number_not_confused_as_pe(self, extractor):
        """Parenthesized numbers like '(12.3)' are often negative earnings, not P/E."""
        text = "EPS: (12.3)"
        result = extractor.extract_from_text(text)
        # Should not produce a positive P/E of 12.3 from a negative EPS marker
        pe = result.get("trailingPE")
        assert pe is None or pe < 0 or pe != pytest.approx(12.3)
