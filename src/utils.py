"""
This module provides utility classes for post-processing the output of the
agentic workflow, such as extracting a clean signal and enabling agents to
reflect on their performance for continuous learning.
"""

import re
from collections.abc import Callable

import structlog

from src.agents import AgentState, extract_string_content
from src.config import Config
from src.llms import quick_thinking_llm
from src.memory import FinancialSituationMemory

logger = structlog.get_logger(__name__)


class SignalProcessor:
    """
    Parses the final natural language output from the Portfolio Manager into a
    clean, machine-readable signal (BUY, SELL, or HOLD).
    """

    def __init__(self, config: Config):
        self.config = config
        self.llm = quick_thinking_llm

    async def process_signal(self, full_signal: str) -> str:
        """
        Extracts the final investment decision from a text block.
        It first tries a robust regex and falls back to an LLM call if needed.
        """
        # 1. Try a robust regex first for efficiency and reliability
        match = re.search(r"\b(BUY|SELL|HOLD)\b", full_signal.upper())
        if match:
            signal = match.group(1)
            logger.info("signal_extracted_via_regex", signal=signal)
            return signal

        # 2. If regex fails, use an LLM call as a fallback
        logger.warning("regex_failed_falling_back_to_llm_for_signal_extraction")
        try:
            messages = [
                (
                    "system",
                    "You are an assistant designed to extract the final investment decision: SELL, BUY, or HOLD from a financial report. Respond with only the single-word decision.",
                ),
                ("human", full_signal),
            ]
            result = await self.llm.ainvoke(messages)
            # CRITICAL FIX: Normalize response.content to string (Gemini may return dict)
            content = extract_string_content(result.content).strip().upper()

            if content in ["BUY", "SELL", "HOLD"]:
                logger.info("signal_extracted_via_llm", signal=content)
                return content

            logger.error("llm_signal_extraction_failed_invalid_output", output=content)
            return "ERROR_UNPARSABLE_SIGNAL"
        except Exception as e:
            logger.error("llm_signal_extraction_exception", error=str(e), exc_info=True)
            return "ERROR_PROCESSING_SIGNAL"


class Reflector:
    """
    Orchestrates the learning process for the agents by prompting them to
    reflect on a trade's outcome and store the resulting lesson.
    """

    def __init__(self, config: Config):
        self.config = config
        self.llm = quick_thinking_llm
        self.reflection_prompt = """You are an expert financial analyst reviewing a past decision.
        Your goal is to generate a concise, one-sentence lesson from this experience to improve future performance.

        Analyze the provided context, the decision/analysis made, and the financial outcome.
        - First, determine if the decision was correct or incorrect.
        - Identify the most critical factors that led to the outcome.
        - Formulate a single, powerful heuristic or lesson.

        Market Context & Analysis:
        {situation}

        Outcome (Profit/Loss): ${returns_losses:,.2f}

        Your output must be a single sentence. Example: 'In a market with strong technical momentum, high valuation concerns can be temporarily overlooked.'
        """

    async def reflect(
        self,
        current_state: AgentState,
        returns_losses: float,
        memory: FinancialSituationMemory,
        component_key_func: Callable[[AgentState], str],
    ):
        """
        Generates a reflection and stores it in the agent's memory.

        Args:
            current_state: The final state of the graph run.
            returns_losses: The profit or loss from the trade.
            memory: The memory object of the agent that will learn.
            component_key_func: A function to extract the specific text for the agent to reflect on.
        """
        if not self.config.enable_memory:
            return

        decision_text = component_key_func(current_state)
        situation = (
            f"Reports:\n"
            f"- Market: {current_state.get('market_report', 'N/A')}\n"
            f"- Sentiment: {current_state.get('sentiment_report', 'N/A')}\n"
            f"- News: {current_state.get('news_report', 'N/A')}\n"
            f"- Fundamentals: {current_state.get('fundamentals_report', 'N/A')}\n\n"
            f"Decision/Analysis Text to reflect on:\n{decision_text}"
        )

        prompt = self.reflection_prompt.format(
            situation=situation, returns_losses=returns_losses
        )

        try:
            result = await self.llm.ainvoke(prompt)
            # CRITICAL FIX: Normalize response.content to string (Gemini may return dict)
            lesson = extract_string_content(result.content).strip()

            # The situation (context) and the lesson (result) are stored.
            await memory.add_situations([(situation, lesson)])
            logger.info(
                "reflection_completed_and_memory_updated", agent_name=memory.name
            )
        except Exception as e:
            logger.error(
                "reflection_failed", agent_name=memory.name, error=str(e), exc_info=True
            )


def clean_duplicate_data_blocks(report: str) -> str:
    """
    Remove all DATA_BLOCKs except the last one from a fundamentals report.

    The fundamentals analyst self-corrects by outputting multiple DATA_BLOCKs
    as it refines calculations. This is INTENTIONAL and shows good reasoning.
    We keep only the final (most accurate) version for user-facing output.

    Why we keep the last block:
    - First block: Initial calculation (may have errors)
    - Last block: Self-corrected, verified calculation (accurate)

    Args:
        report: Full fundamentals analyst report text

    Returns:
        Cleaned report with only the final DATA_BLOCK

    Example:
        >>> report = '''
        ... ### --- START DATA_BLOCK ---
        ... FINANCIAL_HEALTH_SCORE: 7/12  # Wrong!
        ... ### --- END DATA_BLOCK ---
        ...
        ... [agent recalculates and corrects]
        ...
        ... ### --- START DATA_BLOCK ---
        ... FINANCIAL_HEALTH_SCORE: 3/12  # Correct!
        ... ### --- END DATA_BLOCK ---
        ... '''
        >>> clean = clean_duplicate_data_blocks(report)
        # Result: Only the corrected (3/12) block remains
    """

    if not report or not isinstance(report, str):
        return report

    # Pattern to match DATA_BLOCK sections
    pattern = r"### --- START DATA_BLOCK ---.*?### --- END DATA_BLOCK ---"

    # Find all occurrences
    blocks = list(re.finditer(pattern, report, re.DOTALL))

    if len(blocks) <= 1:
        # No duplicates, return as-is
        return report

    # Remove all blocks except the last one (the corrected version)
    cleaned_report = report

    for i, block in enumerate(blocks[:-1], 1):  # All except last
        # Replace with note explaining why it was removed
        replacement = (
            f"### --- DATA_BLOCK #{i} REMOVED ---\n"
            f"*(Agent self-corrected below - keeping final accurate version)*\n\n"
        )
        cleaned_report = cleaned_report.replace(block.group(0), replacement, 1)

    return cleaned_report


def detect_truncation(text: str) -> dict:
    """
    Detect if text appears truncated and identify the truncation source.

    Truncation can occur from:
    1. Code-level limits (tool outputs, search results) - HIGH confidence
    2. LLM output limits (model stops mid-generation) - MEDIUM confidence

    Args:
        text: The text to analyze for truncation

    Returns:
        dict with keys:
        - truncated: bool - whether truncation was detected
        - source: str | None - "code" or "llm" or None
        - marker: str | None - the truncation marker or indicator found
        - confidence: str - "high", "medium", or "low"

    Example:
        >>> detect_truncation("Some content\\n[...TRUNCATED 5000 chars...]\\nMore")
        {'truncated': True, 'source': 'code', 'marker': '[...TRUNCATED', 'confidence': 'high'}

        >>> detect_truncation("The company's revenue grew by 15% driven by")
        {'truncated': True, 'source': 'llm', 'marker': "ends with: '...'", 'confidence': 'medium'}
    """
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "truncated": False,
            "source": None,
            "marker": None,
            "confidence": "high",
        }

    text = text.strip()

    # Code-level truncation markers (HIGH confidence)
    code_markers = [
        "[...TRUNCATED",
        "[...truncated",
        "[... truncated",
        "...truncated for efficiency]",
        "...truncated for display",
        "[NOTE: Data truncated",
    ]
    for marker in code_markers:
        if marker in text:
            return {
                "truncated": True,
                "source": "code",
                "marker": marker,
                "confidence": "high",
            }

    # Check for incomplete structured blocks FIRST (MEDIUM confidence)
    # These blocks should have both start and required fields
    block_completeness = [
        ("PM_BLOCK:", ["VERDICT:", "RISK_ZONE:"]),  # PM output needs verdict
        (
            "DATA_BLOCK:",
            ["HEALTH_SCORE:", "GROWTH_SCORE:"],
        ),  # Fundamentals needs scores
        ("FORENSIC_DATA_BLOCK:", ["VERDICT:", "STATUS:"]),  # Auditor needs verdict
        ("VALUE_TRAP_BLOCK:", ["SCORE:", "VERDICT:"]),  # Value trap needs score
    ]

    has_complete_block = False
    for block_start, required_fields in block_completeness:
        if block_start in text:
            # Block started - check if any required field is present
            has_required = any(field in text for field in required_fields)
            if not has_required:
                return {
                    "truncated": True,
                    "source": "llm",
                    "marker": f"incomplete {block_start} block (missing {required_fields})",
                    "confidence": "medium",
                }
            else:
                has_complete_block = True

    # If we have complete structured blocks, consider output complete
    if has_complete_block:
        return {
            "truncated": False,
            "source": None,
            "marker": None,
            "confidence": "high",
        }

    # LLM truncation heuristics (MEDIUM confidence)
    # Check if ends mid-sentence (not with valid ending punctuation)
    # Only apply if no structured blocks were found
    valid_endings = ".!?)]}>`*|\"'\n"
    stripped = text.rstrip()
    if stripped and stripped[-1] not in valid_endings:
        return {
            "truncated": True,
            "source": "llm",
            "marker": f"ends with: '{stripped[-30:]}'",
            "confidence": "medium",
        }

    return {"truncated": False, "source": None, "marker": None, "confidence": "high"}
