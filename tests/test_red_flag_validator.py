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
