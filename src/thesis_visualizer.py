"""
Thesis Compliance Visualizer for Multi-Agent Trading System.
Generates ASCII/Unicode bar charts to help users quickly understand investment decisions.

Color Support:
- ANSI terminal colors are optional (use_color=True)
- Colorblind-safe palette: Cyan (pass), Blue (warn/neutral), Magenta (fail)
- Avoids red-green which affects ~8% of men with protanopia/deuteranopia
- Avoids yellow which is hard to see on light backgrounds
- All colors use bold for maximum visibility across light/dark themes
"""

import re
from dataclasses import dataclass


class AnsiColors:
    """
    Colorblind-safe ANSI color codes with light/dark theme compatibility.

    Color palette designed for:
    - Colorblind accessibility (no pure red/green - safe for protanopia, deuteranopia)
    - Light background visibility (avoiding pure yellow which is hard to see)
    - Dark background visibility (using bright variants with bold)

    Based on IBM/Wong colorblind-safe palette recommendations.
    """

    # Positive outcomes (pass, good scores) - Cyan works on both light/dark
    PASS = "\033[1;96m"  # Bold bright cyan
    GOOD = "\033[1;94m"  # Bold blue

    # Warning/neutral - Using bold blue instead of yellow for light theme visibility
    # Yellow (93) is hard to see on light backgrounds; blue is universally visible
    WARN = "\033[1;94m"  # Bold blue (visible on light/dark)
    NEUTRAL = "\033[0m"  # Normal (no color)

    # Negative outcomes (fail, bad scores) - Magenta works on both light/dark
    FAIL = "\033[1;95m"  # Bold magenta
    BAD = "\033[1;35m"  # Bold dark magenta

    # Decisions - Using distinct colors for each
    BUY = "\033[1;96m"  # Bold cyan (positive)
    HOLD = "\033[1;94m"  # Bold blue (neutral)
    SELL = "\033[1;95m"  # Bold magenta (negative)

    # Formatting
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disabled(cls) -> "AnsiColors":
        """Return a no-op colors instance for non-terminal output."""

        class NoColors:
            PASS = GOOD = WARN = NEUTRAL = FAIL = BAD = ""
            BUY = HOLD = SELL = BOLD = RESET = ""

        return NoColors()


@dataclass
class ThesisMetrics:
    """Extracted thesis compliance metrics from Portfolio Manager output."""

    # Core scores (higher = better, min 50%)
    health_score: float | None = None
    growth_score: float | None = None

    # Valuation (lower = better)
    pe_ratio: float | None = None
    peg_ratio: float | None = None

    # Hard fail checks
    liquidity_pass: bool | None = None
    liquidity_value: str | None = None
    analyst_coverage: int | None = None
    analyst_pass: bool | None = None
    us_revenue: str | None = None
    us_revenue_pass: bool | None = None

    # Risk tally
    total_risk: float | None = None

    # Decision
    decision: str | None = None


class ThesisVisualizer:
    """Generates ASCII bar charts for thesis compliance visualization."""

    # Unicode bar characters
    FILLED = "▓"
    EMPTY = "░"
    BAR_WIDTH = 20

    # Thresholds from thesis
    HEALTH_MIN = 50.0
    GROWTH_MIN = 50.0
    PE_MAX = 18.0
    PEG_MAX = 1.2
    ANALYST_MAX = 15
    US_REVENUE_MAX = 35.0
    LIQUIDITY_MIN = 500_000  # $500k
    RISK_HIGH = 2.0
    RISK_MODERATE = 1.0

    def __init__(self, final_decision_text: str):
        """
        Initialize visualizer with Portfolio Manager output.

        Args:
            final_decision_text: The final_trade_decision text from result dict
        """
        self.text = final_decision_text
        self.metrics = self._extract_metrics()

    def _extract_metrics(self) -> ThesisMetrics:
        """Extract thesis metrics from Portfolio Manager output."""
        metrics = ThesisMetrics()

        # Note: Patterns use \*?\*? to handle optional markdown bold markers

        # Financial Health: 71% (Adjusted) - PASS
        # Also matches: **Financial Health**: 71%
        health_match = re.search(
            r"\*?\*?Financial Health\*?\*?[:\s]*(\d+(?:\.\d+)?)\s*%",
            self.text,
            re.IGNORECASE,
        )
        if health_match:
            metrics.health_score = float(health_match.group(1))

        # Growth Transition: 100% (Adjusted) - PASS
        growth_match = re.search(
            r"\*?\*?Growth\s*(?:Transition)?\*?\*?[:\s]*(\d+(?:\.\d+)?)\s*%",
            self.text,
            re.IGNORECASE,
        )
        if growth_match:
            metrics.growth_score = float(growth_match.group(1))

        # P/E Ratio: 11.80 (PEG: 0.16) - PASS
        pe_match = re.search(
            r"\*?\*?P/?E\s*(?:Ratio)?\*?\*?[:\s]*(\d+(?:\.\d+)?)",
            self.text,
            re.IGNORECASE,
        )
        if pe_match:
            metrics.pe_ratio = float(pe_match.group(1))

        peg_match = re.search(r"PEG[:\s()]*(\d+(?:\.\d+)?)", self.text, re.IGNORECASE)
        if peg_match:
            metrics.peg_ratio = float(peg_match.group(1))

        # Liquidity: $86.9M Daily Avg - PASS
        liquidity_match = re.search(
            r"\*?\*?Liquidity\*?\*?[:\s]*\$?([\d.]+)\s*([MBK])?.*?(PASS|FAIL|MARGINAL)",
            self.text,
            re.IGNORECASE,
        )
        if liquidity_match:
            value = float(liquidity_match.group(1))
            unit = (liquidity_match.group(2) or "").upper()
            # Format value: show as int if whole number, else float
            value_str = str(int(value)) if value == int(value) else str(value)
            if unit == "M":
                metrics.liquidity_value = f"${value_str}M"
            elif unit == "B":
                metrics.liquidity_value = f"${value_str}B"
            elif unit == "K":
                metrics.liquidity_value = f"${value_str}K"
            else:
                metrics.liquidity_value = f"${value_str}"
            metrics.liquidity_pass = liquidity_match.group(3).upper() in (
                "PASS",
                "MARGINAL",
            )

        # Analyst Coverage: 8 - PASS
        analyst_match = re.search(
            r"\*?\*?Analyst Coverage\*?\*?[:\s]*(\d+).*?(PASS|FAIL)",
            self.text,
            re.IGNORECASE,
        )
        if analyst_match:
            metrics.analyst_coverage = int(analyst_match.group(1))
            metrics.analyst_pass = analyst_match.group(2).upper() == "PASS"

        # US Revenue: ~15-20% - PASS or Not disclosed
        us_rev_match = re.search(
            r"\*?\*?US Revenue\*?\*?[:\s]*([\d~.\-]+%?|Not disclosed|N/A).*?(PASS|FAIL|MARGINAL|N/A)?",
            self.text,
            re.IGNORECASE,
        )
        if us_rev_match:
            metrics.us_revenue = us_rev_match.group(1).strip()
            result = (us_rev_match.group(2) or "").upper()
            metrics.us_revenue_pass = result in ("PASS", "N/A", "MARGINAL", "")

        # TOTAL RISK COUNT: 1.33 or **TOTAL RISK COUNT**: **1.33**
        risk_match = re.search(
            r"\*?\*?TOTAL RISK (?:COUNT|SCORE)?\*?\*?[:\s]*\*?\*?(\d+(?:\.\d+)?)\*?\*?",
            self.text,
            re.IGNORECASE,
        )
        if risk_match:
            metrics.total_risk = float(risk_match.group(1))

        # Decision: BUY/SELL/HOLD
        decision_match = re.search(
            r"(?:FINAL DECISION|Action)[:\s]*\*?\*?(BUY|SELL|HOLD)\*?\*?",
            self.text,
            re.IGNORECASE,
        )
        if decision_match:
            metrics.decision = decision_match.group(1).upper()

        return metrics

    def _bar(
        self,
        value: float,
        max_value: float,
        width: int = BAR_WIDTH,
        invert: bool = False,
    ) -> str:
        """
        Generate a bar chart string.

        Args:
            value: Current value
            max_value: Maximum value for scaling
            width: Bar width in characters
            invert: If True, lower values fill more (for "lower is better" metrics)

        Returns:
            Bar string like "▓▓▓▓▓▓▓░░░░░░░░░░░░░"
        """
        if max_value <= 0:
            return self.EMPTY * width

        ratio = min(value / max_value, 1.0)
        if invert:
            # For "lower is better" - show how much room is left under max
            ratio = 1.0 - ratio
            # But we want to show the value, so invert again
            ratio = value / max_value

        filled = int(ratio * width)
        empty = width - filled
        return self.FILLED * filled + self.EMPTY * empty

    def _check(self, passed: bool | None, c: AnsiColors | None = None) -> str:
        """Return checkmark or X based on pass status, optionally colored."""
        if passed is None:
            return "?"
        if c is None:
            return "✓" if passed else "✗"
        # With colors
        if passed:
            return f"{c.PASS}✓{c.RESET}"
        else:
            return f"{c.FAIL}✗{c.RESET}"

    def _zone(self, risk: float, c: AnsiColors | None = None) -> str:
        """Determine risk zone from total risk count, optionally colored."""
        if risk >= self.RISK_HIGH:
            zone = "HIGH"
            color = c.FAIL if c else ""
        elif risk >= self.RISK_MODERATE:
            zone = "MODERATE"
            color = c.WARN if c else ""
        else:
            zone = "LOW"
            color = c.PASS if c else ""

        if c:
            return f"{color}{zone}{c.RESET}"
        return zone

    def generate(self, use_color: bool = False) -> str:
        """
        Generate the complete thesis compliance visual.

        Args:
            use_color: If True, add ANSI color codes for terminal display.
                       Uses colorblind-safe palette (cyan/yellow/magenta).

        Returns:
            ASCII visualization (optionally with ANSI colors for terminal)
        """
        m = self.metrics
        c = AnsiColors() if use_color else None

        # Check if we have enough data to generate a meaningful visual
        has_scores = m.health_score is not None or m.growth_score is not None
        has_valuation = m.pe_ratio is not None
        if not has_scores and not has_valuation:
            return ""  # Not enough data for visualization

        lines = [
            "```" if not use_color else "",
            "THESIS COMPLIANCE VISUAL",
            "━" * 56,
            "",
        ]
        # Remove empty first line if using colors (no markdown fence)
        if use_color:
            lines = lines[1:]

        # Core Scores Section (Higher = Better)
        if m.health_score is not None or m.growth_score is not None:
            lines.append("CORE SCORES (Higher = Better)")
            lines.append("─" * 56)

            if m.health_score is not None:
                bar = self._bar(m.health_score, 100.0)
                check = self._check(m.health_score >= self.HEALTH_MIN, c)
                lines.append(
                    f"Financial Health  {bar} {m.health_score:5.1f}% {check} (min 50%)"
                )

            if m.growth_score is not None:
                bar = self._bar(m.growth_score, 100.0)
                check = self._check(m.growth_score >= self.GROWTH_MIN, c)
                lines.append(
                    f"Growth Transition {bar} {m.growth_score:5.1f}% {check} (min 50%)"
                )

            lines.append("")

        # Valuation Section (Lower = Better)
        if m.pe_ratio is not None or m.peg_ratio is not None:
            lines.append("VALUATION (Lower = Better)")
            lines.append("─" * 56)

            if m.pe_ratio is not None:
                # Scale PE to show relative to max (18)
                # Lower PE means more "headroom" - show how much of the budget is used
                bar = self._bar(
                    m.pe_ratio, self.PE_MAX * 1.5
                )  # Scale to 27 for visibility
                check = self._check(m.pe_ratio <= self.PE_MAX, c)
                lines.append(
                    f"P/E Ratio         {bar} {m.pe_ratio:5.1f}  {check} (max 18)"
                )

            if m.peg_ratio is not None:
                bar = self._bar(
                    m.peg_ratio, self.PEG_MAX * 2
                )  # Scale to 2.4 for visibility
                check = self._check(m.peg_ratio <= self.PEG_MAX, c)
                lines.append(
                    f"PEG Ratio         {bar} {m.peg_ratio:5.2f}  {check} (max 1.2)"
                )

            lines.append("")

        # Hard Fail Checks Section
        hard_checks = []
        if m.liquidity_pass is not None:
            check = self._check(m.liquidity_pass, c)
            val = m.liquidity_value or "N/A"
            hard_checks.append(f"  {check} Liquidity ({val} daily)")

        if m.analyst_pass is not None:
            check = self._check(m.analyst_pass, c)
            count = m.analyst_coverage if m.analyst_coverage is not None else "?"
            hard_checks.append(f"  {check} Analyst Coverage ({count} < 15)")

        if m.us_revenue_pass is not None:
            check = self._check(m.us_revenue_pass, c)
            val = m.us_revenue or "N/A"
            hard_checks.append(f"  {check} US Revenue ({val})")

        if hard_checks:
            lines.append("HARD FAIL CHECKS")
            lines.append("─" * 56)
            lines.extend(hard_checks)
            lines.append("")

        # Risk Tally Section
        if m.total_risk is not None:
            lines.append("RISK TALLY (Lower = Better)")
            lines.append("─" * 56)
            # Scale risk to show 0-3 range
            bar = self._bar(m.total_risk, 3.0)
            zone = self._zone(m.total_risk, c)
            lines.append(f"Risk Score        {bar} {m.total_risk:5.2f} → Zone: {zone}")
            lines.append("")

        # Decision Banner
        if m.decision:
            lines.append("━" * 56)
            if c:
                # Color the decision based on type
                decision_color = {
                    "BUY": c.BUY,
                    "HOLD": c.HOLD,
                    "SELL": c.SELL,
                }.get(m.decision, "")
                lines.append(f"DECISION: {c.BOLD}{decision_color}{m.decision}{c.RESET}")
            else:
                lines.append(f"DECISION: {m.decision}")

        if not use_color:
            lines.append("```")

        return "\n".join(lines)


def generate_thesis_visual(final_decision_text: str, use_color: bool = False) -> str:
    """
    Convenience function to generate thesis compliance visual.

    Args:
        final_decision_text: Portfolio Manager's final_trade_decision output
        use_color: If True, add colorblind-safe ANSI colors for terminal display.
                   Uses cyan (pass), yellow (warn), magenta (fail) - safe for
                   protanopia, deuteranopia, and tritanopia.

    Returns:
        ASCII visualization, or empty string if insufficient data.
        When use_color=False, wrapped in markdown code fences.
        When use_color=True, raw ANSI-colored output for terminal.
    """
    if not final_decision_text:
        return ""
    visualizer = ThesisVisualizer(final_decision_text)
    return visualizer.generate(use_color=use_color)
