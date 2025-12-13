"""
Tests for Red-Flag Financial Health Validator

This module tests the pre-screening validator that catches extreme financial risks
before proceeding to the bull/bear debate phase.

Red-flag criteria tested:
1. Extreme Leverage: D/E ratio > 500%
2. Earnings Quality Disconnect: Positive income but negative FCF >2x income
3. Refinancing Risk: Interest coverage <2.0x with D/E >100%

Run with: pytest tests/test_red_flag_validator.py -v
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from src.agents import AgentState, create_financial_health_validator_node
from src.validators.red_flag_detector import RedFlagDetector


class TestMetricExtraction:
    """Test financial metric extraction from Fundamentals Analyst reports."""

    def test_extract_from_complete_data_block(self):
        """Test extraction from a complete DATA_BLOCK."""
        report = """
### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 7/12
ADJUSTED_HEALTH_SCORE: 58% (7/12 available)
RAW_GROWTH_SCORE: 4/6
ADJUSTED_GROWTH_SCORE: 67% (4/6 available)
US_REVENUE_PERCENT: Not disclosed
ANALYST_COVERAGE_ENGLISH: 8
PE_RATIO_TTM: 12.34
PE_RATIO_FORWARD: 10.50
PEG_RATIO: 0.85
ADR_EXISTS: NO
### --- END DATA_BLOCK ---

### FINANCIAL HEALTH DETAIL
**Profitability (2/3 pts)**:
- ROE: 15.5%: 1 pts
- ROA: 8.2%: 1 pts
- Operating Margin: 9.5%: 0 pts

**Leverage (1/2 pts)**:
- D/E: 120: 0 pts
- NetDebt/EBITDA: 1.5: 1 pts

**Liquidity (2/2 pts)**:
- Current Ratio: 1.8: 1 pts
- Positive TTM OCF: 1 pts

**Cash Generation (2/2 pts)**:
- Positive FCF: 1 pts
- FCF Yield: 5.2%: 1 pts

**Interest Coverage**: 3.5x

**Free Cash Flow**: $450M
**Net Income**: $600M
"""

        metrics = RedFlagDetector.extract_metrics(report)

        assert metrics['adjusted_health_score'] == 58.0
        assert metrics['pe_ratio'] == 12.34
        assert metrics['debt_to_equity'] == 120.0
        assert metrics['interest_coverage'] == 3.5
        assert metrics['fcf'] == 450_000_000  # 450M
        assert metrics['net_income'] == 600_000_000  # 600M

    def test_extract_multiple_data_blocks_uses_last(self):
        """Test that extraction uses the LAST DATA_BLOCK (self-correction pattern)."""
        report = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 30% (incorrect)
PE_RATIO_TTM: 99.99
### --- END DATA_BLOCK ---

[Agent recalculates...]

### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 75% (corrected)
PE_RATIO_TTM: 14.50
### --- END DATA_BLOCK ---
"""

        metrics = RedFlagDetector.extract_metrics(report)

        # Should use the LAST (corrected) block
        assert metrics['adjusted_health_score'] == 75.0
        assert metrics['pe_ratio'] == 14.50

    def test_extract_handles_missing_data_block(self):
        """Test extraction when DATA_BLOCK is missing."""
        report = "No DATA_BLOCK in this report"

        metrics = RedFlagDetector.extract_metrics(report)

        # All metrics should be None
        assert all(v is None for v in metrics.values())

    def test_extract_debt_to_equity_conversion(self):
        """Test D/E ratio conversion from ratio to percentage."""
        # Case 1: Already percentage (>10)
        report1 = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 50%
### --- END DATA_BLOCK ---
D/E: 250
"""
        metrics1 = RedFlagDetector.extract_metrics(report1)
        assert metrics1['debt_to_equity'] == 250.0

        # Case 2: Ratio format (<10) - convert to percentage
        report2 = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 50%
### --- END DATA_BLOCK ---
Debt/Equity: 2.5
"""
        metrics2 = RedFlagDetector.extract_metrics(report2)
        assert metrics2['debt_to_equity'] == 250.0  # 2.5 * 100

    def test_extract_fcf_with_multipliers(self):
        """Test FCF extraction with B/M/K multipliers."""
        # Billions
        report_b = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 70%
### --- END DATA_BLOCK ---
Free Cash Flow: $1.5B
"""
        metrics_b = RedFlagDetector.extract_metrics(report_b)
        assert metrics_b['fcf'] == 1_500_000_000

        # Millions
        report_m = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 70%
### --- END DATA_BLOCK ---
FCF: $250M
"""
        metrics_m = RedFlagDetector.extract_metrics(report_m)
        assert metrics_m['fcf'] == 250_000_000

        # Thousands
        report_k = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 70%
### --- END DATA_BLOCK ---
Free Cash Flow: 500K
"""
        metrics_k = RedFlagDetector.extract_metrics(report_k)
        assert metrics_k['fcf'] == 500_000

    def test_extract_negative_fcf(self):
        """Test extraction of negative FCF."""
        report = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 40%
### --- END DATA_BLOCK ---
Free Cash Flow: -$850M
"""
        metrics = RedFlagDetector.extract_metrics(report)
        assert metrics['fcf'] == -850_000_000


class TestRedFlagValidatorNode:
    """Test the financial health validator node logic."""

    @pytest.fixture
    def validator_node(self):
        """Create validator node fixture."""
        return create_financial_health_validator_node()

    @pytest.fixture
    def base_state(self):
        """Create a base AgentState for testing."""
        return {
            'company_of_interest': 'TEST.HK',
            'company_name': 'Test Company Ltd',
            'fundamentals_report': '',
            'messages': [],
        }

    @pytest.mark.asyncio
    async def test_extreme_leverage_triggers_reject(self, validator_node, base_state):
        """Test that D/E > 500% triggers AUTO_REJECT."""
        base_state['fundamentals_report'] = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 40%
PE_RATIO_TTM: 15.0
### --- END DATA_BLOCK ---

**Leverage (0/2 pts)**:
- D/E: 600
- Interest Coverage: 1.2x
"""

        result = await validator_node(base_state, {})

        assert result['pre_screening_result'] == 'REJECT'
        assert len(result['red_flags']) > 0
        assert any(flag['type'] == 'EXTREME_LEVERAGE' for flag in result['red_flags'])
        assert any(flag['action'] == 'AUTO_REJECT' for flag in result['red_flags'])

    @pytest.mark.asyncio
    async def test_earnings_quality_disconnect_triggers_reject(self, validator_node, base_state):
        """Test that positive income + deeply negative FCF triggers AUTO_REJECT."""
        base_state['fundamentals_report'] = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 60%
PE_RATIO_TTM: 12.0
### --- END DATA_BLOCK ---

**Cash Generation**:
Net Income: $500M
Free Cash Flow: -$1,200M
"""

        result = await validator_node(base_state, {})

        assert result['pre_screening_result'] == 'REJECT'
        assert len(result['red_flags']) > 0
        assert any(flag['type'] == 'EARNINGS_QUALITY' for flag in result['red_flags'])
        # FCF is -1200M, which is 2.4x the net income of 500M

    @pytest.mark.asyncio
    async def test_refinancing_risk_triggers_reject(self, validator_node, base_state):
        """Test that low interest coverage + high leverage triggers AUTO_REJECT."""
        base_state['fundamentals_report'] = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 45%
PE_RATIO_TTM: 8.0
### --- END DATA_BLOCK ---

**Leverage**:
- D/E: 180
- Interest Coverage: 1.5x
"""

        result = await validator_node(base_state, {})

        assert result['pre_screening_result'] == 'REJECT'
        assert len(result['red_flags']) > 0
        assert any(flag['type'] == 'REFINANCING_RISK' for flag in result['red_flags'])

    @pytest.mark.asyncio
    async def test_healthy_company_passes(self, validator_node, base_state):
        """Test that a healthy company passes all checks."""
        base_state['fundamentals_report'] = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 75%
PE_RATIO_TTM: 14.5
### --- END DATA_BLOCK ---

**Leverage (2/2 pts)**:
- D/E: 65
- Interest Coverage: 8.5x

**Cash Generation (2/2 pts)**:
- Net Income: $800M
- Free Cash Flow: $650M
- FCF Yield: 6.2%
"""

        result = await validator_node(base_state, {})

        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_marginal_metrics_pass_if_below_thresholds(self, validator_node, base_state):
        """Test that marginal but acceptable metrics pass."""
        # D/E at 450% (below 500% threshold)
        # Interest coverage at 2.5x (above 2.0x threshold)
        # Negative FCF but <2x net income
        base_state['fundamentals_report'] = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 52%
PE_RATIO_TTM: 16.0
### --- END DATA_BLOCK ---

**Leverage**:
- D/E: 450
- Interest Coverage: 2.5x

**Cash Generation**:
- Net Income: $400M
- Free Cash Flow: -$600M
"""

        result = await validator_node(base_state, {})

        # Should PASS - none of the thresholds breached
        # - D/E 450% < 500%
        # - Interest coverage 2.5x > 2.0x
        # - FCF -600M is 1.5x net income (not >2x)
        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_multiple_red_flags_all_reported(self, validator_node, base_state):
        """Test that multiple red flags are all detected and reported."""
        base_state['fundamentals_report'] = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 25%
PE_RATIO_TTM: 20.0
### --- END DATA_BLOCK ---

**Leverage**:
D/E: 650
Interest Coverage: 1.3x

**Cash Generation**:
Net Income: $200M
Free Cash Flow: -$500M
"""

        result = await validator_node(base_state, {})

        assert result['pre_screening_result'] == 'REJECT'
        assert len(result['red_flags']) >= 2

        flag_types = [flag['type'] for flag in result['red_flags']]
        assert 'EXTREME_LEVERAGE' in flag_types
        assert 'REFINANCING_RISK' in flag_types
        assert 'EARNINGS_QUALITY' in flag_types

    @pytest.mark.asyncio
    async def test_no_fundamentals_report_passes(self, validator_node, base_state):
        """Test that missing fundamentals report results in PASS (graceful degradation)."""
        base_state['fundamentals_report'] = ''

        result = await validator_node(base_state, {})

        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_incomplete_data_does_not_false_positive(self, validator_node, base_state):
        """Test that incomplete data doesn't trigger false positives."""
        # Missing some metrics - should not trigger red flags
        base_state['fundamentals_report'] = """
### --- START DATA_BLOCK ---
ADJUSTED_HEALTH_SCORE: 60%
PE_RATIO_TTM: 13.5
### --- END DATA_BLOCK ---

**Leverage**:
- D/E: 85
- NetDebt/EBITDA: N/A

**Cash Generation**:
- Free Cash Flow: N/A
"""

        result = await validator_node(base_state, {})

        # Should PASS - None values don't trigger checks
        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_edge_case_exactly_at_threshold(self, validator_node, base_state):
        """Test edge cases at exact threshold values."""
        # D/E exactly 500%
        base_state['fundamentals_report'] = """
D/E: 500
"""

        result = await validator_node(base_state, {})

        # Should PASS - threshold is >, not >=
        assert result['pre_screening_result'] == 'PASS'

    @pytest.mark.asyncio
    async def test_interest_coverage_low_but_leverage_ok(self, validator_node, base_state):
        """Test that low interest coverage alone doesn't trigger if leverage is OK."""
        base_state['fundamentals_report'] = """
**Leverage**:
- D/E: 50
- Interest Coverage: 1.5x
"""

        result = await validator_node(base_state, {})

        # Should PASS - refinancing risk requires BOTH low coverage AND high leverage
        assert result['pre_screening_result'] == 'PASS'


class TestRedFlagIntegration:
    """Integration tests for red-flag validator in the graph workflow."""

    @pytest.mark.asyncio
    async def test_validator_state_propagation(self):
        """Test that validator correctly updates state fields."""
        from src.agents import create_financial_health_validator_node

        validator = create_financial_health_validator_node()

        state = {
            'company_of_interest': '9999.HK',
            'company_name': 'Failing Corp',
            'fundamentals_report': 'D/E: 700\nInterest Coverage: 0.8x',
            'messages': [],
        }

        result = await validator(state, {})

        # Verify state updates
        assert 'red_flags' in result
        assert 'pre_screening_result' in result
        assert isinstance(result['red_flags'], list)
        assert result['pre_screening_result'] in ['PASS', 'REJECT']

    def test_red_flag_structure(self):
        """Test that red flags have required structure."""
        from src.agents import create_financial_health_validator_node
        import asyncio

        validator = create_financial_health_validator_node()

        state = {
            'company_of_interest': 'TEST',
            'company_name': 'Test',
            'fundamentals_report': 'D/E: 600',
            'messages': [],
        }

        result = asyncio.run(validator(state, {}))

        for flag in result['red_flags']:
            # Verify required fields
            assert 'type' in flag
            assert 'severity' in flag
            assert 'detail' in flag
            assert 'action' in flag
            assert 'rationale' in flag

            # Verify value constraints
            assert flag['severity'] in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
            assert flag['action'] in ['AUTO_REJECT', 'WARNING']


class TestRealWorldEdgeCases:
    """
    Edge case tests for real-world equity scenarios with unusual data patterns.

    These tests simulate actual problematic companies encountered in production:
    - Zombie companies (extreme leverage, barely surviving)
    - Accounting fraud patterns (positive earnings, deeply negative cash flow)
    - Distressed refinancing situations (debt maturity wall approaching)
    """

    @pytest.fixture
    def validator_node(self):
        """Create validator node fixture."""
        return create_financial_health_validator_node()

    @pytest.mark.asyncio
    async def test_zombie_company_with_extreme_metrics(self, validator_node):
        """
        Test a 'zombie company' with extreme leverage and multiple red flags.

        Real-world scenario: Heavily indebted company (800% D/E) with minimal
        cash flow and dangerously low interest coverage. Often found in distressed
        retail, shipping, or legacy manufacturing sectors.

        Expected: Multiple red flags (EXTREME_LEVERAGE + REFINANCING_RISK)
        """
        state = {
            'company_of_interest': '9999.HK',
            'company_name': 'Zombie Retail Corp',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 2/12
ADJUSTED_HEALTH_SCORE: 17% (2/12 available)
PE_RATIO_TTM: 45.2
PE_RATIO_FORWARD: N/A
PEG_RATIO: N/A
### --- END DATA_BLOCK ---

### FINANCIAL HEALTH DETAIL

**Profitability (0/3 pts)**:
- ROE: -12.3%: 0 pts
- ROA: -3.8%: 0 pts
- Operating Margin: 1.2%: 0 pts

**Leverage (0/2 pts)**:
- **D/E: 8.2** (ratio format - converts to 820%)
- **NetDebt/EBITDA**: 12.5x
- **Interest Coverage**: 1.1x

**Liquidity (1/2 pts)**:
- Current Ratio: 0.95: 0 pts
- Positive TTM OCF: 1 pts (barely positive at $50M)

**Cash Generation (1/2 pts)**:
- Free Cash Flow: $15M (minimal, down from $200M prior year)
- Net Income: -$120M (loss-making)
- FCF Yield: 0.8%: 0 pts
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        # Should trigger REJECT with multiple red flags
        assert result['pre_screening_result'] == 'REJECT'
        assert len(result['red_flags']) >= 2

        flag_types = [flag['type'] for flag in result['red_flags']]
        # Extreme leverage: D/E 820% > 500%
        assert 'EXTREME_LEVERAGE' in flag_types
        # Refinancing risk: Interest coverage 1.1x < 2.0x AND D/E 820% > 100%
        assert 'REFINANCING_RISK' in flag_types

    @pytest.mark.asyncio
    async def test_accounting_fraud_pattern_large_numbers(self, validator_node):
        """
        Test accounting fraud pattern with large dollar amounts and comma formatting.

        Real-world scenario: Company reports strong earnings but has massive negative
        FCF, suggesting revenue recognition fraud or aggressive capitalization of
        expenses. Classic pattern seen in Luckin Coffee, Wirecard, Enron.

        This test also validates parsing of:
        - Comma-separated large numbers ($1,250.5M)
        - Negative values with dollar signs (-$3,800.2M)
        - Mixed formatting in same report

        Expected: EARNINGS_QUALITY red flag
        """
        state = {
            'company_of_interest': 'FRAUD.CN',
            'company_name': 'Suspicious Growth Inc',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 6/12
ADJUSTED_HEALTH_SCORE: 50% (6/12 available)
PE_RATIO_TTM: 8.5
ANALYST_COVERAGE_ENGLISH: 3
### --- END DATA_BLOCK ---

### FINANCIAL HEALTH DETAIL

**Profitability (2/3 pts)**:
- ROE: 18.5%: 1 pts
- ROA: 9.2%: 1 pts
- Operating Margin: 8.1%: 0 pts

**Leverage (2/2 pts)**:
- D/E: 75: 1 pts
- NetDebt/EBITDA: 1.8: 1 pts

**Liquidity (1/2 pts)**:
- Current Ratio: 1.4: 1 pts
- Negative TTM OCF: 0 pts (red flag!)

**Cash Generation (1/2 pts)**:
- **Net Income**: $1,250.5M (comma-separated, reported earnings strong)
- **Free Cash Flow**: -$3,800.2M (deeply negative, comma-separated with sign)
- Working Capital Change: -$4.2B (massive cash outflow!)
- FCF Yield: -12.3%: 0 pts

### NOTES
Company reports record revenues and earnings, but operating cash flow is deeply
negative due to massive increases in receivables and inventory. Classic fraud pattern
where revenue is recognized but cash never collected. FCF is 3.0x net income in
negative direction.
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        # Should trigger REJECT with earnings quality red flag
        assert result['pre_screening_result'] == 'REJECT'
        assert len(result['red_flags']) >= 1

        flag_types = [flag['type'] for flag in result['red_flags']]
        # Earnings quality: Positive income $1,250.5M but negative FCF -$3,800.2M (>2x)
        assert 'EARNINGS_QUALITY' in flag_types

        # Verify the flag detail contains actual numbers
        earnings_flag = next(f for f in result['red_flags'] if f['type'] == 'EARNINGS_QUALITY')
        # Numbers are formatted as full integers, e.g., $1,250,500,000
        assert '1250' in earnings_flag['detail'] or '1,250' in earnings_flag['detail']
        assert '3800' in earnings_flag['detail'] or '3,800' in earnings_flag['detail']

    @pytest.mark.asyncio
    async def test_distressed_debt_maturity_wall(self, validator_node):
        """
        Test company approaching debt maturity wall with refinancing pressure.

        Real-world scenario: Company has moderate-high leverage (280% D/E) but faces
        imminent debt maturities with deteriorating interest coverage. Rising rates
        make refinancing challenging. Common in overleveraged real estate, telecom,
        or energy companies.

        This test validates:
        - D/E ratio in percentage format (already >100)
        - Interest coverage just below threshold (1.8x < 2.0x)
        - Proper triggering of REFINANCING_RISK flag

        Expected: REFINANCING_RISK red flag (but NOT extreme leverage, since 280% < 500%)
        """
        state = {
            'company_of_interest': '8888.HK',
            'company_name': 'Overleveraged Property Developer',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
RAW_HEALTH_SCORE: 5/12
ADJUSTED_HEALTH_SCORE: 42% (5/12 available)
PE_RATIO_TTM: 6.2
PE_RATIO_FORWARD: 5.8
### --- END DATA_BLOCK ---

### FINANCIAL HEALTH DETAIL

**Profitability (2/3 pts)**:
- ROE: 11.2%: 1 pts (declining from 15.8% prior year)
- ROA: 4.1%: 1 pts
- Operating Margin: 12.5%: 0 pts

**Leverage (0/2 pts)** - CONCERN:
- **D/E: 280** (high but not extreme)
- **NetDebt/EBITDA**: 5.2x (elevated)
- **Interest Coverage**: 1.8x (deteriorating - was 3.5x two years ago)

**Liquidity (1/2 pts)** - WATCH:
- Current Ratio: 1.1: 1 pts
- Positive TTM OCF: 0 pts (negative $200M due to project delays)

**Cash Generation (2/2 pts)**:
- Net Income: $450M (still profitable)
- Free Cash Flow: $180M (positive but down 60% YoY)
- FCF Yield: 4.2%: 1 pts

### RISK FACTORS
- $2.5B in debt maturities due within 18 months
- Rising interest rates increase refinancing costs
- Slowing property market reduces asset sale options
- Interest coverage declining rapidly (1.8x, was 3.5x in 2021)
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        # Should trigger REJECT with refinancing risk
        assert result['pre_screening_result'] == 'REJECT'
        assert len(result['red_flags']) == 1  # Only REFINANCING_RISK, not extreme leverage

        flag = result['red_flags'][0]
        assert flag['type'] == 'REFINANCING_RISK'
        assert flag['severity'] == 'CRITICAL'
        assert flag['action'] == 'AUTO_REJECT'

        # Verify detail mentions both metrics
        assert '1.8' in flag['detail']  # Interest coverage
        assert '280' in flag['detail']  # D/E ratio

        # Should NOT trigger extreme leverage (280% < 500% threshold)
        flag_types = [flag['type'] for flag in result['red_flags']]
        assert 'EXTREME_LEVERAGE' not in flag_types


class TestSectorAwareRedFlags:
    """Test sector-specific threshold adjustments for capital-intensive sectors."""

    @pytest.fixture
    def validator_node(self):
        """Create validator node fixture."""
        return create_financial_health_validator_node()

    @pytest.mark.asyncio
    async def test_utilities_sector_higher_leverage_threshold(self, validator_node):
        """
        Test that utilities sector allows higher D/E ratio (800% vs 500%).

        Real-world scenario: Electric utilities commonly operate with D/E ratios
        of 200-300% due to capital-intensive infrastructure investments. This is
        normal and sustainable given regulated cash flows.
        """
        state = {
            'company_of_interest': 'UTIL.T',
            'company_name': 'Tokyo Electric Power',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Utilities
SECTOR_ADJUSTMENTS: D/E threshold raised to 800% (vs 500% standard) for capital-intensive utilities
ADJUSTED_HEALTH_SCORE: 55%
PE_RATIO_TTM: 12.5
### --- END DATA_BLOCK ---

**Leverage**:
- D/E: 250 (normal for utilities)
- Interest Coverage: 2.8x
- NetDebt/EBITDA: 4.5x
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        # Should PASS - 250% D/E is below the 800% utilities threshold
        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_shipping_sector_allows_higher_leverage(self, validator_node):
        """
        Test that shipping/commodities sector allows higher D/E (800% threshold).

        Real-world scenario: Pulp/paper companies like SUZ commonly have D/E ratios
        of 200-300% due to capital-intensive mills and cyclical nature. This is
        acceptable if they generate strong EBITDA during upturns.
        """
        state = {
            'company_of_interest': 'SUZ',
            'company_name': 'Suzano Pulp & Paper',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Shipping & Cyclical Commodities
SECTOR_ADJUSTMENTS: D/E threshold raised to 800% (vs 500% standard). Interest coverage threshold lowered to 1.5x (vs 2.0x standard) for capital-intensive sector.
ADJUSTED_HEALTH_SCORE: 60%
PE_RATIO_TTM: 10.2
### --- END DATA_BLOCK ---

**Leverage** - Capital-intensive sector norms:
- D/E: 220 (typical for pulp/paper)
- Interest Coverage: 1.8x (generates strong EBITDA, coverage acceptable for sector)
- NetDebt/EBITDA: 3.2x

**Cash Generation**:
- EBITDA: $2.5B (strong operating performance)
- Free Cash Flow: $800M (positive, cyclical downturns OK)
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        # Should PASS - 220% D/E < 800% threshold, 1.8x coverage > 1.5x threshold for shipping
        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_shipping_sector_refinancing_risk_adjusted_thresholds(self, validator_node):
        """
        Test that shipping sector uses adjusted thresholds for refinancing risk.

        Refinancing risk for shipping: coverage < 1.5x (vs 2.0x) + D/E > 200% (vs 100%)
        """
        state = {
            'company_of_interest': 'SHIP.HK',
            'company_name': 'Asian Shipping Corp',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Shipping & Cyclical Commodities
ADJUSTED_HEALTH_SCORE: 52%
### --- END DATA_BLOCK ---

**Leverage**:
- D/E: 180 (below 200% sector threshold)
- Interest Coverage: 1.6x (above 1.5x sector threshold)
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        # Should PASS - both metrics above sector-adjusted thresholds
        # D/E 180% < 200% threshold, coverage 1.6x > 1.5x threshold
        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_shipping_sector_fails_when_exceeds_sector_thresholds(self, validator_node):
        """
        Test that shipping sector still fails when metrics exceed SECTOR thresholds.
        """
        state = {
            'company_of_interest': 'FAIL.HK',
            'company_name': 'Failing Shipping Corp',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Shipping & Cyclical Commodities
ADJUSTED_HEALTH_SCORE: 30%
### --- END DATA_BLOCK ---

**Leverage**:
- D/E: 850 (exceeds 800% sector threshold)
- Interest Coverage: 1.2x (below 1.5x sector threshold)
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        # Should REJECT - D/E 850% > 800% sector threshold
        assert result['pre_screening_result'] == 'REJECT'
        flag_types = [flag['type'] for flag in result['red_flags']]
        assert 'EXTREME_LEVERAGE' in flag_types

    @pytest.mark.asyncio
    async def test_banking_sector_skips_leverage_checks(self, validator_node):
        """
        Test that banking sector skips D/E checks entirely (leverage is their business).

        Real-world scenario: Banks have D/E ratios of 1000%+ by design (deposits are
        liabilities). D/E is meaningless - focus is on Tier 1 Capital, NPL ratios.
        """
        state = {
            'company_of_interest': '0005.HK',
            'company_name': 'HSBC Holdings',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Banking
SECTOR_ADJUSTMENTS: D/E ratio excluded (not applicable for banks) - Leverage score denominator adjusted to 1 pt. ROE threshold lowered to 12% (vs 15% standard). ROA threshold lowered to 1.0% (vs 7% standard).
ADJUSTED_HEALTH_SCORE: 65%
PE_RATIO_TTM: 9.8
### --- END DATA_BLOCK ---

**Leverage** - NOT APPLICABLE FOR BANKS:
- D/E: 1200 (meaningless for banks - deposits are liabilities)
- Tier 1 Capital Ratio: 14.5% (regulatory capital - strong)
- NPL Ratio: 1.8% (asset quality - good)
- Interest Coverage: N/A (not applicable for banks)

**Profitability**:
- ROE: 13.2% (above 12% bank threshold)
- ROA: 0.95% (below 1.0% bank threshold but improving)
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        # Should PASS - banking sector skips all D/E and coverage checks
        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_general_sector_uses_standard_thresholds(self, validator_node):
        """
        Test that General/Diversified sector uses standard thresholds (500% D/E).
        """
        state = {
            'company_of_interest': 'GEN.HK',
            'company_name': 'General Manufacturing',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: General/Diversified
SECTOR_ADJUSTMENTS: None - standard thresholds applied
ADJUSTED_HEALTH_SCORE: 55%
### --- END DATA_BLOCK ---

**Leverage**:
- D/E: 520 (exceeds 500% standard threshold)
- Interest Coverage: 1.9x
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        # Should REJECT - 520% D/E > 500% standard threshold
        assert result['pre_screening_result'] == 'REJECT'
        flag_types = [flag['type'] for flag in result['red_flags']]
        assert 'EXTREME_LEVERAGE' in flag_types

    @pytest.mark.asyncio
    async def test_sector_detection_from_report(self, validator_node):
        """
        Test that sector is correctly detected from SECTOR field in DATA_BLOCK.
        """
        from src.validators.red_flag_detector import RedFlagDetector, Sector

        # Test utilities detection
        report_utilities = """
### --- START DATA_BLOCK ---
SECTOR: Utilities
ADJUSTED_HEALTH_SCORE: 60%
### --- END DATA_BLOCK ---
"""
        sector_utilities = RedFlagDetector.detect_sector(report_utilities)
        assert sector_utilities == Sector.UTILITIES

        # Test shipping detection
        report_shipping = """
### --- START DATA_BLOCK ---
SECTOR: Shipping & Cyclical Commodities
### --- END DATA_BLOCK ---
"""
        sector_shipping = RedFlagDetector.detect_sector(report_shipping)
        assert sector_shipping == Sector.SHIPPING

        # Test banking detection
        report_banking = """
### --- START DATA_BLOCK ---
SECTOR: Banking
### --- END DATA_BLOCK ---
"""
        sector_banking = RedFlagDetector.detect_sector(report_banking)
        assert sector_banking == Sector.BANKING

        # Test fallback to GENERAL
        report_no_sector = """
No SECTOR field here
"""
        sector_general = RedFlagDetector.detect_sector(report_no_sector)
        assert sector_general == Sector.GENERAL


class TestRealWorldSectorExamples:
    """
    Real-world ticker examples for each sector with pass/fail scenarios.

    These tests use realistic financial profiles based on actual international equities
    to ensure the sector-aware thresholds work correctly in production scenarios.
    """

    @pytest.fixture
    def validator_node(self):
        """Create validator node fixture."""
        return create_financial_health_validator_node()

    # --- UTILITIES SECTOR ---

    @pytest.mark.asyncio
    async def test_utilities_pass_tokyo_electric_power(self, validator_node):
        """
        PASS: Tokyo Electric Power (9501.T) - Typical Japanese utility.

        Profile: D/E ~200%, coverage 3.2x, regulated monopoly with stable cash flows.
        Should PASS utilities sector thresholds (D/E < 800%, coverage > 1.5x).
        """
        state = {
            'company_of_interest': '9501.T',
            'company_name': 'Tokyo Electric Power Company (TEPCO)',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Utilities
SECTOR_ADJUSTMENTS: D/E threshold raised to 800% (vs 500% standard) for capital-intensive utilities. Interest coverage threshold lowered to 1.5x (vs 2.0x standard).
ADJUSTED_HEALTH_SCORE: 58%
PE_RATIO_TTM: 11.2
### --- END DATA_BLOCK ---

**Leverage** (Utility-Adjusted):
- D/E: 210 (typical for regulated utility)
- NetDebt/EBITDA: 5.8x
- Interest Coverage: 3.2x (stable regulated cash flows)

**Cash Generation**:
- EBITDA: ¥850B
- Free Cash Flow: ¥180B
- Operating Cash Flow: ¥320B
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_utilities_fail_distressed_spanish_utility(self, validator_node):
        """
        FAIL: Distressed European utility (hypothetical) - Extreme leverage post-acquisition.

        Profile: D/E 920% (exceeds 800% utilities threshold), coverage 1.2x.
        Should REJECT even with utilities sector adjustments.
        """
        state = {
            'company_of_interest': 'UTIL.MC',
            'company_name': 'Distressed Utility Corp (Spain)',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Utilities
SECTOR_ADJUSTMENTS: D/E threshold raised to 800% (vs 500% standard) for capital-intensive utilities
ADJUSTED_HEALTH_SCORE: 32%
PE_RATIO_TTM: 8.5
### --- END DATA_BLOCK ---

**Leverage** - CRITICAL CONCERN:
- D/E: 920 (extreme even for utility - post-acquisition debt bomb)
- NetDebt/EBITDA: 12.5x
- Interest Coverage: 1.2x (stressed by rising rates)

**Cash Generation**:
- EBITDA: €1.2B (declining)
- Free Cash Flow: -€200M (negative due to capex overruns)
- Debt Maturities: €5B due in 18 months
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        assert result['pre_screening_result'] == 'REJECT'
        flag_types = [flag['type'] for flag in result['red_flags']]
        assert 'EXTREME_LEVERAGE' in flag_types  # 920% > 800% utilities threshold

    # --- SHIPPING & CYCLICAL COMMODITIES SECTOR ---

    @pytest.mark.asyncio
    async def test_shipping_pass_suzano_pulp(self, validator_node):
        """
        PASS: Suzano (SUZ) - Brazilian pulp & paper leader.

        Profile: D/E 220%, coverage 1.8x, strong EBITDA generation in cyclical industry.
        Should PASS shipping/commodities thresholds (D/E < 800%, coverage > 1.5x).
        This is the exact use case that motivated sector-aware thresholds.
        """
        state = {
            'company_of_interest': 'SUZ',
            'company_name': 'Suzano S.A. (Pulp & Paper)',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Shipping & Cyclical Commodities
SECTOR_ADJUSTMENTS: D/E threshold raised to 800% (vs 500% standard). Interest coverage threshold lowered to 1.5x (vs 2.0x standard) for capital-intensive sector.
ADJUSTED_HEALTH_SCORE: 62%
PE_RATIO_TTM: 9.8
### --- END DATA_BLOCK ---

**Leverage** - Capital-Intensive Sector Norms:
- D/E: 220 (typical for pulp/paper - capex-heavy mills)
- NetDebt/EBITDA: 3.1x
- Interest Coverage: 1.8x (strong EBITDA covers debt comfortably)

**Cash Generation**:
- EBITDA: R$18.5B (strong operating performance)
- Free Cash Flow: R$4.2B (positive even during pulp price downturn)
- Operating Margin: 42% (best-in-class cost structure)

**Sector Context**:
- Pulp prices cyclical but company generates cash through cycle
- Vertical integration (forestry + mills) supports margins
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_shipping_fail_overleveraged_dry_bulk(self, validator_node):
        """
        FAIL: Overleveraged dry bulk shipper (common pattern in shipping distress).

        Profile: D/E 850% (exceeds 800% threshold), coverage 0.9x, commodity downturn.
        Should REJECT - classic shipping bankruptcy pattern.
        """
        state = {
            'company_of_interest': '2866.HK',
            'company_name': 'Failing Dry Bulk Shipper',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Shipping & Cyclical Commodities
SECTOR_ADJUSTMENTS: D/E threshold raised to 800% for capital-intensive sector
ADJUSTED_HEALTH_SCORE: 28%
PE_RATIO_TTM: N/A (losses)
### --- END DATA_BLOCK ---

**Leverage** - CRITICAL DISTRESS:
- D/E: 850 (extreme even for shipping - overleveraged at cycle peak)
- NetDebt/EBITDA: 18.2x
- Interest Coverage: 0.9x (cannot service debt)

**Cash Generation**:
- EBITDA: $45M (collapsed from $200M in 2021 peak)
- Free Cash Flow: -$120M (negative)
- Baltic Dry Index: Down 70% from peak (freight rate collapse)

**Risk Factors**:
- Ordered ships at cycle peak (2021) with high leverage
- Commodity downturn crushed freight rates
- Debt maturities approaching with no refinancing options
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        assert result['pre_screening_result'] == 'REJECT'
        flag_types = [flag['type'] for flag in result['red_flags']]
        assert 'EXTREME_LEVERAGE' in flag_types  # 850% > 800% threshold
        assert 'REFINANCING_RISK' in flag_types  # Coverage 0.9x < 1.5x + D/E > 200%

    # --- BANKING SECTOR ---

    @pytest.mark.asyncio
    async def test_banking_pass_hsbc_high_leverage(self, validator_node):
        """
        PASS: HSBC (0005.HK) - Global bank with typical bank capital structure.

        Profile: D/E 1200% (normal for banks), strong Tier 1 capital, low NPLs.
        Should PASS - banking sector skips D/E checks entirely.
        """
        state = {
            'company_of_interest': '0005.HK',
            'company_name': 'HSBC Holdings',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Banking
SECTOR_ADJUSTMENTS: D/E ratio excluded (not applicable for banks) - Leverage score denominator adjusted to 1 pt. ROE threshold lowered to 12% (vs 15% standard). ROA threshold lowered to 1.0% (vs 7% standard).
ADJUSTED_HEALTH_SCORE: 68%
PE_RATIO_TTM: 9.2
### --- END DATA_BLOCK ---

**Leverage** - NOT APPLICABLE FOR BANKS:
- D/E: 1200 (deposits are liabilities - meaningless metric)
- Tier 1 Capital Ratio: 14.8% (regulatory capital - well above 10% minimum)
- Common Equity Tier 1: 13.5% (strong)
- Leverage Ratio: 5.2% (above 3% regulatory minimum)

**Asset Quality**:
- NPL Ratio: 1.6% (non-performing loans - good quality)
- Loan Loss Coverage: 180% (strong provisioning)

**Profitability**:
- ROE: 13.8% (above 12% bank threshold)
- ROA: 0.62% (typical for large global bank)
- Net Interest Margin: 1.68%
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_banking_pass_even_with_extreme_de(self, validator_node):
        """
        PASS: Regional Japanese bank with 2000% D/E (normal for banks).

        Validates that banking sector truly skips D/E checks regardless of ratio.
        """
        state = {
            'company_of_interest': '8411.T',
            'company_name': 'Mizuho Financial Group',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Banking
SECTOR_ADJUSTMENTS: D/E ratio excluded (not applicable for banks)
ADJUSTED_HEALTH_SCORE: 55%
PE_RATIO_TTM: 7.8
### --- END DATA_BLOCK ---

**Leverage** - NOT APPLICABLE:
- D/E: 2100 (21:1 leverage typical for Japanese megabank)
- Tier 1 Capital Ratio: 11.2% (adequate)
- Interest Coverage: N/A (not applicable for banks)

**Asset Quality**:
- NPL Ratio: 2.3% (acceptable for regional exposure)
- Coverage Ratio: 145%

**Profitability**:
- ROE: 8.5% (below 12% threshold but improving)
- ROA: 0.42%
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        # Should PASS - banking sector skips all leverage checks
        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    # --- TECHNOLOGY SECTOR ---

    @pytest.mark.asyncio
    async def test_technology_pass_samsung_electronics(self, validator_node):
        """
        PASS: Samsung Electronics (005930.KS) - Tech leader with conservative balance sheet.

        Profile: D/E 45%, strong cash generation, minimal leverage.
        Should PASS standard thresholds easily.
        """
        state = {
            'company_of_interest': '005930.KS',
            'company_name': 'Samsung Electronics',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Technology & Software
SECTOR_ADJUSTMENTS: None - standard thresholds applied
ADJUSTED_HEALTH_SCORE: 78%
PE_RATIO_TTM: 12.5
### --- END DATA_BLOCK ---

**Leverage** - Conservative:
- D/E: 45 (minimal leverage for tech leader)
- NetDebt/EBITDA: 0.2x (net cash position)
- Interest Coverage: 28.5x (trivial debt burden)

**Cash Generation**:
- EBITDA: ₩85T
- Free Cash Flow: ₩32T (strong)
- Operating Cash Flow: ₩52T
- Net Cash: ₩42T (fortress balance sheet)

**Profitability**:
- ROE: 14.2%
- Operating Margin: 12.8% (memory cycle downturn)
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_technology_fail_overleveraged_lbo(self, validator_node):
        """
        FAIL: Overleveraged tech company post-LBO (private equity casualty).

        Profile: D/E 580% (exceeds 500% standard threshold), coverage 1.6x.
        Should REJECT - tech companies shouldn't operate with extreme leverage.
        """
        state = {
            'company_of_interest': 'TECH.HK',
            'company_name': 'Overleveraged Tech Corp',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: Technology & Software
SECTOR_ADJUSTMENTS: None - standard thresholds applied
ADJUSTED_HEALTH_SCORE: 38%
PE_RATIO_TTM: 15.2
### --- END DATA_BLOCK ---

**Leverage** - EXTREME FOR TECH:
- D/E: 580 (overleveraged post-LBO - PE firm loaded with debt)
- NetDebt/EBITDA: 6.8x
- Interest Coverage: 1.6x (tight for tech company)

**Cash Generation**:
- EBITDA: $120M
- Free Cash Flow: $15M (minimal after debt service)
- Revenue Growth: -5% (post-acquisition synergies failed)

**Risk Factors**:
- Private equity LBO in 2021 at peak valuations
- Failed to achieve projected cost synergies
- Customer attrition post-acquisition (25% lost)
- Debt matures in 2026 with refinancing risk
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        assert result['pre_screening_result'] == 'REJECT'
        flag_types = [flag['type'] for flag in result['red_flags']]
        assert 'EXTREME_LEVERAGE' in flag_types  # 580% > 500% standard threshold

    # --- GENERAL SECTOR ---

    @pytest.mark.asyncio
    async def test_general_pass_toyota_motor(self, validator_node):
        """
        PASS: Toyota Motor (7203.T) - Auto manufacturer with strong balance sheet.

        Profile: D/E 180%, coverage 8.2x, best-in-class operational execution.
        Should PASS standard thresholds comfortably.
        """
        state = {
            'company_of_interest': '7203.T',
            'company_name': 'Toyota Motor Corporation',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: General/Diversified
SECTOR_ADJUSTMENTS: None - standard thresholds applied
ADJUSTED_HEALTH_SCORE: 72%
PE_RATIO_TTM: 9.8
### --- END DATA_BLOCK ---

**Leverage** - Manageable:
- D/E: 180 (includes Toyota Financial Services - captive auto finance)
- NetDebt/EBITDA: 2.1x
- Interest Coverage: 8.2x (comfortable debt service)

**Cash Generation**:
- EBITDA: ¥4.8T
- Free Cash Flow: ¥2.2T (strong)
- Operating Cash Flow: ¥3.8T
- Cash & Equivalents: ¥5.1T

**Profitability**:
- ROE: 16.8%
- Operating Margin: 10.2% (industry-leading)
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        assert result['pre_screening_result'] == 'PASS'
        assert len(result['red_flags']) == 0

    @pytest.mark.asyncio
    async def test_general_fail_chinese_property_developer(self, validator_node):
        """
        FAIL: Chinese property developer (Evergrande-style distress).

        Profile: D/E 620%, coverage 0.8x, classic China property crisis pattern.
        Should REJECT - multiple red flags (extreme leverage + refinancing risk).
        """
        state = {
            'company_of_interest': '3333.HK',
            'company_name': 'Distressed Property Developer',
            'fundamentals_report': """
### --- START DATA_BLOCK ---
SECTOR: General/Diversified
SECTOR_ADJUSTMENTS: None - standard thresholds applied
ADJUSTED_HEALTH_SCORE: 18%
PE_RATIO_TTM: 3.2 (distressed valuation)
### --- END DATA_BLOCK ---

**Leverage** - CRITICAL DISTRESS:
- D/E: 620 (extreme - China property crisis)
- NetDebt/EBITDA: 15.8x
- Interest Coverage: 0.8x (cannot service debt)

**Cash Generation**:
- EBITDA: ¥12B (collapsing)
- Free Cash Flow: -¥45B (massive outflow)
- Net Income: ¥8B (positive but questionable quality)

**Liquidity Crisis**:
- Cash: ¥15B
- Short-term debt: ¥180B (maturity wall)
- Asset sales: Stalled (no buyers in distressed market)
- Government intervention: Uncertain

**Risk Factors**:
- Three Red Lines policy breach (all three metrics failed)
- Offshore bond default imminent
- Property sales down 70% YoY
- Credit rating: CCC+ (substantial risk)
""",
            'messages': [],
        }

        result = await validator_node(state, {})

        assert result['pre_screening_result'] == 'REJECT'
        flag_types = [flag['type'] for flag in result['red_flags']]
        assert 'EXTREME_LEVERAGE' in flag_types  # 620% > 500% standard threshold
        assert 'REFINANCING_RISK' in flag_types  # Coverage 0.8x < 2.0x + D/E > 100%
