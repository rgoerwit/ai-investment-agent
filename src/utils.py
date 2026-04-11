"""
This module provides utility classes for post-processing the output of the
agentic workflow, such as extracting a clean signal and detecting truncation.
"""

import re
import unicodedata
from collections.abc import Callable

import structlog

from src.agents import extract_string_content
from src.config import Config
from src.data_block_utils import has_parseable_data_block, has_parseable_fenced_block
from src.llms import quick_thinking_llm

logger = structlog.get_logger(__name__)


def _line_starts_with_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, re.MULTILINE) for pattern in patterns)


def _has_complete_legacy_block(
    text: str,
    header_patterns: tuple[str, ...],
    required_fields: tuple[str, ...],
) -> bool:
    if not _line_starts_with_any(text, header_patterns):
        return False
    return any(field in text for field in required_fields)


def _ends_with_known_structured_terminator(text: str, agent: str | None) -> bool:
    raw_data_end = "=== END RAW DATA ==="
    if agent is not None and agent != "junior_fundamentals_analyst":
        return False
    return text.rstrip().endswith(raw_data_end)


_TERMINAL_FIELD_PATTERN = re.compile(r"^\s*[A-Z][A-Z0-9_ /()%.\-]{1,80}:\s*(.+?)\s*$")
_TERMINAL_SCALAR_VALUES = frozenset(
    {
        "N/A",
        "NA",
        "NONE",
        "BUY",
        "SELL",
        "HOLD",
        "PASS",
        "FAIL",
        "APPROVED",
        "REJECTED",
        "CAUTIOUS",
        "CLEAN",
        "YES",
        "NO",
        "TRUE",
        "FALSE",
    }
)
_TERMINAL_DANGLING_WORDS = frozenset(
    {
        "AND",
        "OR",
        "TO",
        "FOR",
        "WITH",
        "BY",
        "FROM",
        "DUE",
        "BECAUSE",
        "VIA",
        "THROUGH",
        "OF",
        "IN",
        "ON",
        "AT",
        "THE",
        "A",
        "AN",
    }
)


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _is_complete_scalar_terminal_field_line(text: str) -> bool:
    last_line = _last_nonempty_line(text)
    if not last_line:
        return False

    match = _TERMINAL_FIELD_PATTERN.match(last_line)
    if not match:
        return False

    value = re.sub(r"\s+", " ", match.group(1)).strip()
    if not value:
        return False

    normalized = value.upper()
    if normalized in _TERMINAL_SCALAR_VALUES:
        return True

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return True

    if re.fullmatch(
        r"[-+]?[$€£¥]?\d[\d,]*(?:\.\d+)?%?(?:[KMBT]|X)?(?:\s*/\s*[-+]?[$€£¥]?\d[\d,]*(?:\.\d+)?%?(?:[KMBT]|X)?)?",
        normalized,
    ):
        return True

    if re.fullmatch(r"[A-Z0-9][A-Z0-9./_\-]{0,31}", normalized):
        return True

    words = normalized.split()
    if words and words[-1] in _TERMINAL_DANGLING_WORDS:
        return False

    return False


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

    # Pattern to match DATA_BLOCK sections (tolerates optional descriptive text after "DATA_BLOCK")
    pattern = r"### --- START DATA_BLOCK[^\n]*---.*?### --- END DATA_BLOCK ---"

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


def detect_truncation(text: str, agent: str | None = None) -> dict:
    """
    Detect if text appears truncated and identify the truncation source.

    Truncation can occur from:
    1. Code-level limits (tool outputs, search results) - HIGH confidence
    2. LLM output limits (model stops mid-generation) - MEDIUM confidence

    Args:
        text: The text to analyze for truncation
        agent: Agent key (e.g. "portfolio_manager"). When set, structured-block
               checks only fire for the agent that produces that block type,
               preventing false positives when other agents merely reference it.

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

    if _ends_with_known_structured_terminator(text, agent):
        return {
            "truncated": False,
            "source": None,
            "marker": None,
            "confidence": "high",
        }

    # Check for incomplete structured blocks FIRST (MEDIUM confidence)
    # These blocks should have a valid start marker plus required fields.
    # We keep the current lenient "any required field" behavior for legacy
    # line-start variants to avoid broad behavior changes.
    block_rules: tuple[
        tuple[str, str, tuple[str, ...], tuple[str, ...], Callable[[str], bool]],
        ...,
    ] = (
        (
            "PM_BLOCK",
            "portfolio_manager",
            (
                r"^\s*PM_BLOCK:\s*$",
                r"^\s*#+\s*--- START PM_BLOCK[^\n]*---\s*$",
            ),
            ("VERDICT:", "RISK_ZONE:", "ZONE:"),
            lambda value: has_parseable_fenced_block(value, "PM_BLOCK"),
        ),
        (
            "DATA_BLOCK",
            "fundamentals_analyst",
            (
                r"^\s*DATA_BLOCK:\s*$",
                r"^\s*###\s+DATA_BLOCK(?:\b.*)?$",
                r"^\s*#+\s*--- START DATA_BLOCK[^\n]*---\s*$",
            ),
            ("HEALTH_SCORE:", "GROWTH_SCORE:"),
            has_parseable_data_block,
        ),
        (
            "FORENSIC_DATA_BLOCK",
            "global_forensic_auditor",
            (
                r"^\s*FORENSIC_DATA_BLOCK:\s*$",
                r"^\s*#+\s*--- START FORENSIC_DATA_BLOCK[^\n]*---\s*$",
            ),
            ("VERDICT:", "STATUS:"),
            lambda value: has_parseable_fenced_block(value, "FORENSIC_DATA_BLOCK"),
        ),
        (
            "VALUE_TRAP_BLOCK",
            "value_trap_detector",
            (
                r"^\s*VALUE_TRAP_BLOCK:\s*$",
                r"^\s*###\s+VALUE_TRAP_BLOCK(?:\b.*)?$",
                r"^\s*#+\s*--- START VALUE_TRAP_BLOCK[^\n]*---\s*$",
            ),
            ("SCORE:", "VERDICT:"),
            lambda value: has_parseable_fenced_block(value, "VALUE_TRAP_BLOCK"),
        ),
    )

    for (
        block_name,
        owner,
        start_patterns,
        required_fields,
        parseable_check,
    ) in block_rules:
        if agent and agent != owner:
            continue
        if parseable_check(text) or _has_complete_legacy_block(
            text, start_patterns, required_fields
        ):
            return {
                "truncated": False,
                "source": None,
                "marker": None,
                "confidence": "high",
            }
        if _line_starts_with_any(text, start_patterns):
            return {
                "truncated": True,
                "source": "llm",
                "marker": f"incomplete {block_name} block (missing {required_fields})",
                "confidence": "medium",
            }

    # LLM truncation heuristics (MEDIUM confidence)
    # Check if ends mid-sentence (not with valid ending punctuation)
    # Only apply if no structured blocks were found
    # Use Unicode categories instead of a character whitelist so that smart
    # quotes (GPT), CJK periods/brackets (JP/KR/CN), and other international
    # punctuation are recognised as valid endings.
    _non_punct_valid = frozenset(
        ">`|\n"
    )  # Markdown/table chars that aren't Unicode punctuation
    stripped = text.rstrip()
    if stripped:
        if _is_complete_scalar_terminal_field_line(stripped):
            return {
                "truncated": False,
                "source": None,
                "marker": None,
                "confidence": "high",
            }
        last = stripped[-1]
        cat = unicodedata.category(last)
        # Pe=close bracket, Pf=final quote, Po=period/exclamation/etc.
        if last not in _non_punct_valid and cat not in ("Pe", "Pf", "Po"):
            return {
                "truncated": True,
                "source": "llm",
                "marker": f"ends with: '{stripped[-30:]}'",
                "confidence": "medium",
            }

    return {"truncated": False, "source": None, "marker": None, "confidence": "high"}
