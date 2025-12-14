"""
Quiet Mode Report Generator for Multi-Agent Trading System
FIXED: Handles LangGraph list outputs to prevent 'list object has no attribute startswith' errors.
FIXED: Added deduplication to prevent stuttering output in final reports.
FIXED: Case-insensitive regex matching for decision extraction.
UPDATED: Added brief_mode flag for condensed output.
UPDATED: Added comprehensive error handling and fallback logic for missing Portfolio Manager output.
"""

import sys
import logging
from typing import Dict, Optional, Any
from datetime import datetime
import re

# Local import for utility function to avoid circular dependency at module level
# We import inside the method where it is needed

class QuietModeReporter:
    """Generates clean markdown reports with minimal output."""

    def __init__(self, ticker: str, company_name: Optional[str] = None, quick_mode: bool = False):
        self.ticker = ticker.upper()
        self.company_name = company_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.quick_mode = quick_mode

    def _normalize_string(self, content: Any) -> str:
        """
        Safely convert content to string, handling lists from LangGraph state accumulation.
        FIXED: Deduplicates list items to prevent repetition loop artifacts.
        """
        if content is None:
            return ""

        if isinstance(content, list):
            # Deduplication logic
            seen = set()
            unique_items = []
            for item in content:
                if not item:
                    continue
                item_str = str(item).strip()
                # Simple hash check for duplicates
                # We check if the first 100 chars match to catch near-duplicates
                # or identical tool outputs repeated in the loop
                key = item_str[:100]
                if key not in seen:
                    seen.add(key)
                    unique_items.append(item_str)

            return "\n\n".join(unique_items)

        return str(content)

    def extract_decision(self, final_decision: str) -> str:
        """Extract BUY/SELL/HOLD decision from final decision text."""

        # Normalize input first
        final_decision = self._normalize_string(final_decision)

        # Look for explicit decision markers in order of preference
        # Use UPPER CASE matching since we upper() the input string

        # 1. "Action:" in FINAL EXECUTION PARAMETERS (highest priority)
        action_match = re.search(
            r'\bACTION\s*:\s*\*?\*?([A-Z]+)\*?\*?',
            final_decision.upper()
        )
        if action_match:
            decision = action_match.group(1)
            if decision in ['BUY', 'SELL', 'HOLD']:
                return decision

        # 2. "FINAL DECISION:"
        final_decision_match = re.search(
            r'\bFINAL\s+DECISION\s*:\s*\*?\*?([A-Z]+)\*?\*?',
            final_decision.upper()
        )
        if final_decision_match:
            decision = final_decision_match.group(1)
            if decision in ['BUY', 'SELL', 'HOLD']:
                return decision

        # 3. "Decision:" fallback
        decision_match = re.search(
            r'\bDECISION\s*:\s*\*?\*?([A-Z]+)\*?\*?',
            final_decision.upper()
        )
        if decision_match:
            decision = decision_match.group(1)
            if decision in ['BUY', 'SELL', 'HOLD']:
                return decision

        # 4. Generic keyword search (risky, but better than nothing)
        generic_match = re.search(r'\b(BUY|SELL|HOLD)\b', final_decision.upper())
        if generic_match:
            decision = generic_match.group(1)
            return decision

        return "HOLD"  # Default to HOLD if completely unclear

    def _extract_decision_rationale(self, final_decision: str) -> str:
        """
        Extract only the decision rationale section from final_trade_decision.
        Looks for patterns like "DECISION RATIONALE:" or "RATIONALE:".
        """
        final_decision = self._normalize_string(final_decision)

        # Try to find decision rationale section
        rationale_patterns = [
            r'(?:DECISION\s+)?RATIONALE\s*:(.+?)(?:\n\n|\Z)',
            r'REASONING\s*:(.+?)(?:\n\n|\Z)',
            r'JUSTIFICATION\s*:(.+?)(?:\n\n|\Z)'
        ]

        for pattern in rationale_patterns:
            match = re.search(pattern, final_decision, re.IGNORECASE | re.DOTALL)
            if match:
                rationale = match.group(1).strip()
                return self._clean_text(rationale)

        # Fallback: if no specific section found, look for paragraph after decision statement
        decision_keywords = ['BUY', 'SELL', 'HOLD']
        lines = final_decision.split('\n')

        for i, line in enumerate(lines):
            if any(keyword in line.upper() for keyword in decision_keywords):
                # Get next non-empty lines as rationale
                rationale_lines = []
                for j in range(i+1, min(i+6, len(lines))):
                    if lines[j].strip():
                        rationale_lines.append(lines[j])
                if rationale_lines:
                    return self._clean_text('\n'.join(rationale_lines))

        # Last resort: return first 3-4 lines of cleaned text
        paragraphs = [p.strip() for p in final_decision.split('\n\n') if p.strip()]
        if paragraphs:
            return self._clean_text('\n\n'.join(paragraphs[:2]))

        return ""

    def _get_final_decision_text(self, result: Dict) -> str:
        """
        Extract final decision text from result dictionary with comprehensive fallback logic.

        CRITICAL FIX: Handles cases where Portfolio Manager fails to write to final_trade_decision.
        Fallback hierarchy:
        1. result['final_trade_decision'] - Primary output field (AgentState)
        2. result['investment_plan'] - Research Manager synthesis (fallback)
        3. result['trader_investment_plan'] - Trader proposal (last resort)
        4. Error message with debugging info

        Returns:
            str: The final decision text, or an error message with debugging context
        """
        # Try primary field
        final_decision_raw = self._normalize_string(result.get('final_trade_decision', ''))
        if final_decision_raw and final_decision_raw.strip():
            return final_decision_raw

        # Log warning and try fallbacks
        import structlog
        logger = structlog.get_logger(__name__)
        logger.warning(
            "final_trade_decision is empty - Portfolio Manager may have failed",
            ticker=self.ticker,
            has_investment_plan=bool(result.get('investment_plan')),
            has_trader_plan=bool(result.get('trader_investment_plan'))
        )

        # Fallback 1: Research Manager's investment plan
        investment_plan = self._normalize_string(result.get('investment_plan', ''))
        if investment_plan and investment_plan.strip():
            logger.info("Using investment_plan as fallback for final decision", ticker=self.ticker)
            return f"⚠️ **Note: Portfolio Manager output missing - using Research Manager synthesis**\n\n{investment_plan}"

        # Fallback 2: Trader's proposal
        trader_plan = self._normalize_string(result.get('trader_investment_plan', ''))
        if trader_plan and trader_plan.strip():
            logger.info("Using trader_investment_plan as fallback for final decision", ticker=self.ticker)
            return f"⚠️ **Note: Portfolio Manager output missing - using Trader proposal**\n\n{trader_plan}"

        # Complete failure - generate error report with debugging context
        logger.error(
            "All decision fields are empty - analysis likely incomplete",
            ticker=self.ticker,
            available_keys=list(result.keys())
        )

        error_msg = f"""## ⚠️ Analysis Error

**Ticker**: {self.ticker}
**Issue**: Portfolio Manager failed to produce final decision

**Debugging Information**:
- `final_trade_decision`: Empty
- `investment_plan`: {'Present' if result.get('investment_plan') else 'Missing'}
- `trader_investment_plan`: {'Present' if result.get('trader_investment_plan') else 'Missing'}
- `market_report`: {'Present' if result.get('market_report') else 'Missing'}
- `fundamentals_report`: {'Present' if result.get('fundamentals_report') else 'Missing'}

**Possible Causes**:
1. Portfolio Manager agent crashed/timeout during LLM call
2. LangGraph routing error prevented Portfolio Manager execution
3. Rate limiting caused silent failure
4. Memory/resource constraints

**Action Required**:
Re-run analysis with verbose logging: `poetry run python -m src.main --ticker {self.ticker}`
"""
        return error_msg

    def generate_report(self, result: Dict, brief_mode: bool = False) -> str:
        """
        Generate markdown report from analysis results.

        Args:
            result: Dictionary containing analysis results
            brief_mode: If True, output only header, summary, and decision rationale
        """

        # Get final decision with comprehensive error handling
        final_decision_raw = self._get_final_decision_text(result)
        decision = self.extract_decision(final_decision_raw)

        # Build title
        if self.company_name:
            title = f"# {self.ticker} ({self.company_name}): {decision}"
        else:
            title = f"# {self.ticker}: {decision}"

        # Build report sections
        report_parts = [
            title,
            f"\n**Analysis Date:** {self.timestamp}\n",
            "---\n"
        ]

        # Red Flag Pre-Screening (if applicable)
        red_flags = result.get('red_flags', [])
        pre_screening_result = result.get('pre_screening_result', 'PASS')

        if red_flags or pre_screening_result == 'REJECT':
            report_parts.append("\n## 🚨 Red Flag Pre-Screening\n\n")

            if pre_screening_result == 'REJECT':
                report_parts.append("**Status**: CRITICAL RED FLAGS DETECTED - AUTO-REJECT\n\n")
            else:
                report_parts.append("**Status**: ⚠️ Warnings Detected - Proceed with Caution\n\n")

            if red_flags:
                for flag in red_flags:
                    flag_type = flag.get('type', 'UNKNOWN')
                    severity = flag.get('severity', 'UNKNOWN')
                    detail = flag.get('detail', 'No details')

                    report_parts.append(f"- **{flag_type}** ({severity}): {detail}\n")

            if pre_screening_result == 'REJECT':
                report_parts.append("\n*Debate phase skipped due to critical red flags. ")
                report_parts.append("Stock routed directly to Portfolio Manager for final decision.*\n")

            report_parts.append("\n---\n\n")

        # Executive Summary (always included)
        if final_decision_raw:
            report_parts.append("## Executive Summary\n")
            cleaned = self._clean_text(final_decision_raw)
            report_parts.append(f"{cleaned}\n\n---\n")
        else:
            # This shouldn't happen with new fallback logic, but handle it anyway
            report_parts.append("## Executive Summary\n")
            report_parts.append("**Error**: No decision output available from any agent.\n\n---\n")

        # If brief mode, add only decision rationale and exit
        if brief_mode:
            rationale = self._extract_decision_rationale(final_decision_raw)
            if rationale:
                report_parts.append("## Decision Rationale\n")
                report_parts.append(f"{rationale}\n\n---\n")

            # Footer
            mode_indicator = "Brief Mode, Quick Models" if self.quick_mode else "Brief Mode"
            report_parts.append(
                f"*Generated by Multi-Agent Trading System ({mode_indicator}) - {self.timestamp}*\n"
            )
            return "".join(report_parts)

        # Full mode: include all sections
        # Helper function to add sections safely
        def add_section(key, title):
            raw_content = result.get(key, '')
            content = self._normalize_string(raw_content)

            if content and not content.startswith('Error'):
                report_parts.append(f"## {title}\n")
                report_parts.append(f"{self._clean_text(content)}\n\n")

        add_section('market_report', 'Technical Analysis')

        # Clean fundamentals: keep only final self-corrected DATA_BLOCK
        # Import inside function to prevent circular dependency with utils.py
        fund_report = result.get('fundamentals_report', '')
        if fund_report:
            try:
                from src.utils import clean_duplicate_data_blocks
                fund_report = self._normalize_string(fund_report)
                fund_report = clean_duplicate_data_blocks(fund_report)
                result['fundamentals_report'] = fund_report
            except ImportError:
                pass # Fallback if utils not available

        add_section('fundamentals_report', 'Fundamental Analysis')
        add_section('sentiment_report', 'Market Sentiment')
        add_section('news_report', 'News & Catalysts')
        add_section('investment_plan', 'Investment Recommendation')

        # CRITICAL: Include consultant review if present (external cross-validation)
        consultant_review = result.get('consultant_review', '')
        if consultant_review and consultant_review.strip():
            # Check if it's a real review (not an error message or "N/A")
            normalized = self._normalize_string(consultant_review)
            if normalized and "N/A (consultant disabled" not in normalized and not normalized.startswith('Consultant Review Error'):
                report_parts.append("## 🔍 External Consultant Review (Cross-Validation)\n")
                report_parts.append("*Independent review by OpenAI ChatGPT to validate Gemini analysis*\n\n")
                report_parts.append(f"{self._clean_text(normalized)}\n\n")

        add_section('trader_investment_plan', 'Trading Strategy')

        # Risk Assessment (if present)
        risk_state = result.get('risk_debate_state', {})
        if risk_state:
            # Handle both dict and list (take last if list)
            if isinstance(risk_state, list):
                risk_state = risk_state[-1] if risk_state else {}

            risk_history = risk_state.get('history', '') if isinstance(risk_state, dict) else ''
            if risk_history:
                report_parts.append("## Risk Assessment\n")
                report_parts.append(f"{self._clean_text(risk_history)}\n\n")

        # Footer
        mode_suffix = " (Quick Models)" if self.quick_mode else ""
        report_parts.append(f"*Generated by Multi-Agent Trading System{mode_suffix} - {self.timestamp}*\n")

        return "".join(report_parts)

    def _clean_text(self, text: str) -> str:
        """Clean up text for markdown output."""
        if not text:
            return ""

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        # Remove agent prefixes if present
        text = re.sub(
            r'^(Bull Analyst:|Bear Analyst:|Risky Analyst:|Safe Analyst:|'
            r'Neutral Analyst:|Trader:|Portfolio Manager:)\s*',
            '',
            text,
            flags=re.MULTILINE
        )

        if not text.endswith("\n"):
            return text + "\n"
        return text


def suppress_logging():
    """
    Suppress all logging output except critical errors.
    Ensures logging goes to stderr so it doesn't pollute stdout reports.
    """
    import logging
    import warnings

    # Configure root logger to only show CRITICAL errors, directed to stderr
    logging.basicConfig(
        level=logging.CRITICAL,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True  # Override any existing configuration
    )

    # Explicitly set root logger level (basicConfig might not work if already configured)
    logging.root.setLevel(logging.CRITICAL)

    # Suppress all existing loggers
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        logging.getLogger(name).propagate = False

    # Suppress common noisy libraries
    for logger_name in ['httpx', 'openai', 'httpcore', 'langchain', 'langgraph']:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Suppress structlog (used by token_tracker and agents)
    try:
        import structlog

        def null_processor(logger, method_name, event_dict):
            """Drop all log events."""
            raise structlog.DropEvent

        structlog.configure(
            processors=[null_processor],
            wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=False,
        )
    except ImportError:
        pass  # structlog not available, skip
