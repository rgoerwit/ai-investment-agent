from __future__ import annotations

from fractions import Fraction

# These fractions intentionally preserve large headroom for structured and
# final-decision nodes while tightening naturally short-form agents.
AGENT_OUTPUT_BUDGET_FRACTIONS: dict[str, Fraction] = {
    "Market Analyst": Fraction(1, 16),
    "Sentiment Analyst": Fraction(1, 32),
    "News Analyst": Fraction(1, 8),
    "Foreign Language Analyst": Fraction(1, 16),
    "Legal Counsel": Fraction(1, 16),
    "Value Trap Detector": Fraction(1, 16),
    "Valuation Calculator": Fraction(1, 32),
    "Global Forensic Auditor": Fraction(1, 4),
    "Trader": Fraction(1, 8),
    "Risky Analyst": Fraction(1, 8),
    "Safe Analyst": Fraction(1, 8),
    "Neutral Analyst": Fraction(1, 8),
    "Consultant": Fraction(1, 8),
    "Bull Researcher": Fraction(3, 16),
    "Bear Researcher": Fraction(3, 16),
    "Research Manager": Fraction(1, 4),
    "Fundamentals Analyst": Fraction(1, 3),
    "Junior Fundamentals Analyst": Fraction(1, 3),
    "Portfolio Manager": Fraction(1, 2),
}

DEFAULT_AGENT_BUDGET_FRACTION = Fraction(1, 1)


def get_agent_output_budget(agent_name: str, base_tokens: int) -> int:
    """Return the configured output budget for the given agent."""
    fraction = AGENT_OUTPUT_BUDGET_FRACTIONS.get(
        agent_name, DEFAULT_AGENT_BUDGET_FRACTION
    )
    # Ceiling division preserves the intended fraction when base tokens do not
    # divide cleanly (for example 32768 * 1/3).
    return max(
        256,
        (base_tokens * fraction.numerator + fraction.denominator - 1)
        // fraction.denominator,
    )
