"""Tests for chart data extractors."""

import pytest

from src.charts.extractors.data_block import (
    ChartRawData,
    extract_chart_data_from_data_block,
)
from src.charts.extractors.valuation import (
    ValuationParams,
    ValuationTargets,
    _calculate_growth_adjusted,
    _calculate_pe_normalization,
    _calculate_peg_based,
    _extract_params,
    calculate_valuation_targets,
)


class TestDataBlockExtractor:
    """Tests for DATA_BLOCK chart data extraction."""

    def test_extract_all_chart_fields(self):
        """Test extraction of all chart-relevant fields from DATA_BLOCK."""
        report = """
        Some analysis text here...

        ### --- START DATA_BLOCK ---
        SECTOR: General/Diversified
        RAW_HEALTH_SCORE: 8/12
        ADJUSTED_HEALTH_SCORE: 80% (based on 10 available points)
        FIFTY_TWO_WEEK_HIGH: 175.50
        FIFTY_TWO_WEEK_LOW: 120.25
        CURRENT_PRICE: 145.00
        MOVING_AVG_50: 142.30
        MOVING_AVG_200: 138.50
        EXTERNAL_ANALYST_TARGET_HIGH: 180.00
        EXTERNAL_ANALYST_TARGET_LOW: 155.00
        EXTERNAL_ANALYST_TARGET_MEAN: 167.50
        PE_RATIO_TTM: 15.5
        ### --- END DATA_BLOCK ---

        More analysis...
        """

        result = extract_chart_data_from_data_block(report)

        assert result.current_price == 145.00
        assert result.fifty_two_week_high == 175.50
        assert result.fifty_two_week_low == 120.25
        assert result.moving_avg_50 == 142.30
        assert result.moving_avg_200 == 138.50
        assert result.external_target_high == 180.00
        assert result.external_target_low == 155.00
        assert result.external_target_mean == 167.50

    def test_extract_partial_data(self):
        """Test extraction when some fields are N/A."""
        report = """
        ### --- START DATA_BLOCK ---
        FIFTY_TWO_WEEK_HIGH: 175.50
        FIFTY_TWO_WEEK_LOW: 120.25
        CURRENT_PRICE: 145.00
        MOVING_AVG_50: N/A
        MOVING_AVG_200: N/A
        EXTERNAL_ANALYST_TARGET_HIGH: N/A
        EXTERNAL_ANALYST_TARGET_LOW: N/A
        EXTERNAL_ANALYST_TARGET_MEAN: N/A
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)

        assert result.current_price == 145.00
        assert result.fifty_two_week_high == 175.50
        assert result.fifty_two_week_low == 120.25
        assert result.moving_avg_50 is None
        assert result.moving_avg_200 is None
        assert result.external_target_high is None

    def test_extract_empty_report(self):
        """Test extraction from empty report returns empty ChartRawData."""
        result = extract_chart_data_from_data_block("")
        assert result.current_price is None
        assert result.fifty_two_week_high is None

    def test_extract_no_data_block(self):
        """Test extraction when DATA_BLOCK is missing."""
        report = "This is a report without any DATA_BLOCK section."
        result = extract_chart_data_from_data_block(report)
        assert result.current_price is None

    def test_extract_uses_last_data_block(self):
        """Test that extraction uses the last DATA_BLOCK (self-correction pattern)."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 150.00
        FIFTY_TWO_WEEK_LOW: 80.00
        ### --- END DATA_BLOCK ---

        Wait, I made an error. Let me correct:

        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 145.00
        FIFTY_TWO_WEEK_HIGH: 175.50
        FIFTY_TWO_WEEK_LOW: 120.25
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)

        # Should use the second (corrected) block
        assert result.current_price == 145.00
        assert result.fifty_two_week_high == 175.50

    def test_extract_with_commas_in_numbers(self):
        """Test extraction handles numbers with comma separators."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 1,234.56
        FIFTY_TWO_WEEK_HIGH: 2,500.00
        FIFTY_TWO_WEEK_LOW: 1,000.00
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.current_price == 1234.56
        assert result.fifty_two_week_high == 2500.00

    def test_extract_with_markdown_and_currency(self):
        """Test extraction handles markdown formatting and currency symbols."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: **$145.00**
        FIFTY_TWO_WEEK_HIGH: $175.50
        FIFTY_TWO_WEEK_LOW: _120.25_
        EXTERNAL_ANALYST_TARGET_HIGH: `$180.00`
        EXTERNAL_ANALYST_TARGET_LOW: **155.00**
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.current_price == 145.00
        assert result.fifty_two_week_high == 175.50
        assert result.fifty_two_week_low == 120.25
        assert result.external_target_high == 180.00
        assert result.external_target_low == 155.00


class TestExtendedDataExtraction:
    """Tests for extended data extraction (D/E, ROA, VIE, CMIC, etc.)."""

    def test_extract_de_ratio_from_data_block(self):
        """Test D/E ratio extraction from DATA_BLOCK (v7.4+ format)."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        DE_RATIO: 0.147
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.de_ratio == 0.147

    def test_extract_de_ratio_with_na(self):
        """Test D/E ratio is None when N/A in DATA_BLOCK."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        DE_RATIO: N/A
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.de_ratio is None

    def test_extract_roa_from_data_block(self):
        """Test ROA extraction from DATA_BLOCK (v7.4+ format)."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        ROA_PERCENT: 16.22
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.roa == 16.22

    def test_extract_roa_not_found(self):
        """Test ROA is None when not present in DATA_BLOCK."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.roa is None

    def test_extract_vie_structure_yes(self):
        """Test VIE structure detection from DATA_BLOCK."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        VIE_STRUCTURE: YES
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.vie_structure is True

    def test_extract_vie_structure_no(self):
        """Test VIE structure is False when NO in DATA_BLOCK."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        VIE_STRUCTURE: NO
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.vie_structure is False

    def test_extract_vie_structure_na(self):
        """Test VIE is None when N/A in DATA_BLOCK."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        VIE_STRUCTURE: N/A
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.vie_structure is None

    def test_extract_cmic_flagged(self):
        """Test CMIC flag detection from DATA_BLOCK."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        CMIC_STATUS: FLAGGED
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.cmic_flagged is True

    def test_extract_cmic_clear(self):
        """Test CMIC is False when CLEAR in DATA_BLOCK."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        CMIC_STATUS: CLEAR
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.cmic_flagged is False

    def test_extract_jurisdiction(self):
        """Test JURISDICTION extraction from DATA_BLOCK."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        JURISDICTION: Japan.TSE
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.jurisdiction == "Japan.TSE"

    def test_extract_jurisdiction_hong_kong(self):
        """Test JURISDICTION extraction for Hong Kong (no space)."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 55.00
        FIFTY_TWO_WEEK_HIGH: 65.00
        FIFTY_TWO_WEEK_LOW: 45.00
        JURISDICTION: HongKong.HKEX
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.jurisdiction == "HongKong.HKEX"

    def test_extract_jurisdiction_with_space(self):
        """Test JURISDICTION extraction handles spaces (e.g., 'Hong Kong')."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 55.00
        FIFTY_TWO_WEEK_HIGH: 65.00
        FIFTY_TWO_WEEK_LOW: 45.00
        JURISDICTION: Hong Kong.HKEX
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.jurisdiction == "Hong Kong.HKEX"

    def test_extract_pfic_risk_high(self):
        """Test PFIC risk extraction."""
        report = """
        ### --- START DATA_BLOCK ---
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        PFIC_RISK: HIGH
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.pfic_risk == "HIGH"

    def test_extract_adjusted_scores(self):
        """Test extraction of adjusted health and growth scores."""
        report = """
        ### --- START DATA_BLOCK ---
        SECTOR: General/Diversified
        RAW_HEALTH_SCORE: 10/12
        ADJUSTED_HEALTH_SCORE: 83%
        RAW_GROWTH_SCORE: 4/6
        ADJUSTED_GROWTH_SCORE: 67%
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.adjusted_health_score == 83.0
        assert result.adjusted_growth_score == 67.0

    def test_extract_analyst_coverage(self):
        """Test analyst coverage extraction."""
        report = """
        ### --- START DATA_BLOCK ---
        ANALYST_COVERAGE_ENGLISH: 5
        CURRENT_PRICE: 100.00
        FIFTY_TWO_WEEK_HIGH: 120.00
        FIFTY_TWO_WEEK_LOW: 80.00
        ### --- END DATA_BLOCK ---
        """

        result = extract_chart_data_from_data_block(report)
        assert result.analyst_coverage == 5


class TestValuationParamsExtractor:
    """Tests for VALUATION_PARAMS extraction (Valuation Calculator output)."""

    def test_extract_all_params(self):
        """Test extraction of all valuation parameters."""
        report = """
        Analysis text...

        ### --- START VALUATION_PARAMS ---
        METHOD: P/E_NORMALIZATION
        SECTOR: Technology
        SECTOR_MEDIAN_PE: 25
        CURRENT_PE: 18.5
        PEG_RATIO: N/A
        GROWTH_SCORE_PCT: N/A
        CURRENT_PRICE: 150.00
        CONFIDENCE: HIGH
        ### --- END VALUATION_PARAMS ---

        More text...
        """

        result = _extract_params(report)

        assert result.method == "P/E_NORMALIZATION"
        assert result.sector == "Technology"
        assert result.sector_median_pe == 25
        assert result.current_pe == 18.5
        assert result.peg_ratio is None  # N/A
        assert result.growth_score_pct is None  # N/A
        assert result.current_price == 150.00
        assert result.confidence == "HIGH"

    def test_extract_with_currency_symbols(self):
        """Test extraction handles currency symbols in price."""
        report = """
        ### --- START VALUATION_PARAMS ---
        METHOD: P/E_NORMALIZATION
        SECTOR: Finance
        SECTOR_MEDIAN_PE: 12
        CURRENT_PE: 10.0
        CURRENT_PRICE: $145.00
        CONFIDENCE: MEDIUM
        ### --- END VALUATION_PARAMS ---
        """

        result = _extract_params(report)
        assert result.current_price == 145.00

    def test_extract_empty_report(self):
        """Test extraction from empty report."""
        result = _extract_params("")
        assert result.method is None
        assert result.current_price is None

    def test_extract_no_params_block(self):
        """Test extraction when VALUATION_PARAMS block is missing."""
        report = "This report has no valuation params block."
        result = _extract_params(report)
        assert result.method is None

    def test_extract_uses_last_block(self):
        """Test that extraction uses last VALUATION_PARAMS block (self-correction)."""
        report = """
        ### --- START VALUATION_PARAMS ---
        METHOD: P/E_NORMALIZATION
        CURRENT_PRICE: 100.00
        ### --- END VALUATION_PARAMS ---

        Wait, let me correct that:

        ### --- START VALUATION_PARAMS ---
        METHOD: PEG_BASED
        CURRENT_PRICE: 150.00
        PEG_RATIO: 0.8
        ### --- END VALUATION_PARAMS ---
        """

        result = _extract_params(report)
        assert result.method == "PEG_BASED"
        assert result.current_price == 150.00
        assert result.peg_ratio == 0.8


class TestPENormalizationCalculation:
    """Tests for P/E normalization target calculation."""

    def test_calculate_undervalued_stock(self):
        """Test P/E normalization for undervalued stock (current PE < sector median)."""
        params = ValuationParams(
            method="P/E_NORMALIZATION",
            sector="Technology",
            sector_median_pe=25,
            current_pe=18.5,
            current_price=150.00,
            confidence="HIGH",
        )

        result = _calculate_pe_normalization(params)

        # Fair value = 150 * (25 / 18.5) = 202.70
        # Low = 202.70 * 0.85 = 172.30
        # High = 202.70 * 1.15 = 233.11
        assert result.low == pytest.approx(172.30, rel=0.01)
        assert result.high == pytest.approx(233.11, rel=0.01)
        assert result.confidence == "HIGH"
        assert "P/E normalization" in result.methodology

    def test_calculate_overvalued_stock(self):
        """Test P/E normalization for overvalued stock (current PE > sector median)."""
        params = ValuationParams(
            method="P/E_NORMALIZATION",
            sector="Finance",
            sector_median_pe=12,
            current_pe=18,
            current_price=100.00,
            confidence="MEDIUM",
        )

        result = _calculate_pe_normalization(params)

        # Fair value = 100 * (12 / 18) = 66.67
        # Low = 66.67 * 0.85 = 56.67
        # High = 66.67 * 1.15 = 76.67
        assert result.low == pytest.approx(56.67, rel=0.01)
        assert result.high == pytest.approx(76.67, rel=0.01)

    def test_missing_current_pe(self):
        """Test calculation fails gracefully with missing current PE."""
        params = ValuationParams(
            method="P/E_NORMALIZATION",
            sector_median_pe=25,
            current_pe=None,  # Missing
            current_price=150.00,
            confidence="HIGH",
        )

        result = _calculate_pe_normalization(params)
        assert result.low is None
        assert result.high is None

    def test_zero_current_pe(self):
        """Test calculation fails gracefully with zero PE (division by zero guard)."""
        params = ValuationParams(
            method="P/E_NORMALIZATION",
            sector_median_pe=25,
            current_pe=0,  # Zero
            current_price=150.00,
        )

        result = _calculate_pe_normalization(params)
        assert result.low is None

    def test_negative_current_pe(self):
        """Test calculation handles negative PE (loss-making company) gracefully."""
        params = ValuationParams(
            method="P/E_NORMALIZATION",
            sector_median_pe=25,
            current_pe=-5.0,  # Negative PE (company losing money)
            current_price=150.00,
            confidence="HIGH",
        )

        result = _calculate_pe_normalization(params)

        # Should return empty targets (can't normalize negative PE)
        assert result.low is None
        assert result.high is None


class TestPEGBasedCalculation:
    """Tests for PEG-based target calculation."""

    def test_calculate_undervalued_peg(self):
        """Test PEG-based calculation for undervalued stock (PEG < 1.0)."""
        params = ValuationParams(
            method="PEG_BASED",
            peg_ratio=0.8,
            current_price=100.00,
            confidence="MEDIUM",
        )

        result = _calculate_peg_based(params)

        # Fair value = 100 * (1 / 0.8) = 125 (capped at 200 = 100% upside)
        # Low = 125 * 0.85 = 106.25
        # High = 125 * 1.15 = 143.75
        assert result.low == pytest.approx(106.25, rel=0.01)
        assert result.high == pytest.approx(143.75, rel=0.01)
        assert result.confidence == "MEDIUM"

    def test_calculate_extreme_peg(self):
        """Test PEG-based calculation with very low PEG (capped at 100% upside)."""
        params = ValuationParams(
            method="PEG_BASED",
            peg_ratio=0.3,  # Implies 233% upside, should be capped
            current_price=100.00,
        )

        result = _calculate_peg_based(params)

        # Fair value = 100 * (1 / 0.3) = 333.33 → capped at 200
        # Low = 200 * 0.85 = 170
        # High = 200 * 1.15 = 230
        assert result.low == pytest.approx(170.00, rel=0.01)
        assert result.high == pytest.approx(230.00, rel=0.01)

    def test_missing_peg_ratio(self):
        """Test calculation fails gracefully with missing PEG."""
        params = ValuationParams(
            method="PEG_BASED",
            peg_ratio=None,
            current_price=100.00,
        )

        result = _calculate_peg_based(params)
        assert result.low is None

    def test_negative_peg_ratio(self):
        """Test calculation handles negative PEG (declining earnings) gracefully."""
        params = ValuationParams(
            method="PEG_BASED",
            peg_ratio=-1.5,  # Negative PEG (earnings declining)
            current_price=100.00,
            confidence="LOW",
        )

        result = _calculate_peg_based(params)

        # Should return empty targets (negative PEG is meaningless)
        assert result.low is None
        assert result.high is None


class TestGrowthAdjustedCalculation:
    """Tests for growth-adjusted target calculation (fallback method)."""

    def test_calculate_high_growth(self):
        """Test growth-adjusted calculation for high growth stock."""
        params = ValuationParams(
            method="GROWTH_ADJUSTED",
            growth_score_pct=65,  # 65% growth score
            current_price=100.00,
            confidence="LOW",
        )

        result = _calculate_growth_adjusted(params)

        # Upside = 65 * 0.5 / 100 = 32.5%
        # Fair value = 100 * 1.325 = 132.50
        # Low = 132.50 * 0.90 = 119.25
        # High = 132.50 * 1.10 = 145.75
        assert result.low == pytest.approx(119.25, rel=0.01)
        assert result.high == pytest.approx(145.75, rel=0.01)

    def test_calculate_capped_upside(self):
        """Test growth-adjusted calculation caps upside at 50%."""
        params = ValuationParams(
            method="GROWTH_ADJUSTED",
            growth_score_pct=150,  # Would imply 75% upside, capped at 50%
            current_price=100.00,
        )

        result = _calculate_growth_adjusted(params)

        # Upside capped at 50%
        # Fair value = 100 * 1.5 = 150
        # Low = 150 * 0.90 = 135
        # High = 150 * 1.10 = 165
        assert result.low == pytest.approx(135.00, rel=0.01)
        assert result.high == pytest.approx(165.00, rel=0.01)


class TestCalculateValuationTargets:
    """Tests for the main calculate_valuation_targets function."""

    def test_pe_normalization_method(self):
        """Test full calculation with P/E normalization method."""
        report = """
        ### --- START VALUATION_PARAMS ---
        METHOD: P/E_NORMALIZATION
        SECTOR: Technology
        SECTOR_MEDIAN_PE: 25
        CURRENT_PE: 20
        CURRENT_PRICE: 100.00
        CONFIDENCE: HIGH
        ### --- END VALUATION_PARAMS ---
        """

        result = calculate_valuation_targets(report)

        # Fair value = 100 * (25 / 20) = 125
        assert result.low == pytest.approx(106.25, rel=0.01)
        assert result.high == pytest.approx(143.75, rel=0.01)
        assert result.confidence == "HIGH"

    def test_peg_based_method(self):
        """Test full calculation with PEG-based method."""
        report = """
        ### --- START VALUATION_PARAMS ---
        METHOD: PEG_BASED
        PEG_RATIO: 0.9
        CURRENT_PRICE: 100.00
        CONFIDENCE: MEDIUM
        ### --- END VALUATION_PARAMS ---
        """

        result = calculate_valuation_targets(report)

        assert result.low is not None
        assert result.high is not None
        assert result.confidence == "MEDIUM"

    def test_growth_adjusted_method(self):
        """Test full calculation with growth-adjusted method."""
        report = """
        ### --- START VALUATION_PARAMS ---
        METHOD: GROWTH_ADJUSTED
        GROWTH_SCORE_PCT: 60
        CURRENT_PRICE: 100.00
        CONFIDENCE: LOW
        ### --- END VALUATION_PARAMS ---
        """

        result = calculate_valuation_targets(report)

        assert result.low is not None
        assert result.high is not None
        assert result.confidence == "LOW"

    def test_insufficient_data_method(self):
        """Test handling of INSUFFICIENT_DATA method."""
        report = """
        ### --- START VALUATION_PARAMS ---
        METHOD: INSUFFICIENT_DATA
        CURRENT_PRICE: 100.00
        CONFIDENCE: LOW
        ### --- END VALUATION_PARAMS ---
        """

        result = calculate_valuation_targets(report)

        assert result.low is None
        assert result.high is None
        assert "Insufficient data" in result.methodology
        assert result.confidence == "LOW"

    def test_empty_report(self):
        """Test handling of empty report."""
        result = calculate_valuation_targets("")
        assert result.low is None
        assert result.high is None

    def test_unknown_method(self):
        """Test handling of unknown valuation method."""
        report = """
        ### --- START VALUATION_PARAMS ---
        METHOD: UNKNOWN_METHOD
        CURRENT_PRICE: 100.00
        ### --- END VALUATION_PARAMS ---
        """

        result = calculate_valuation_targets(report)
        assert result.low is None
