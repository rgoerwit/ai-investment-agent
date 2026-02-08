"""Base classes for official filing API fetchers.

Provides FilingResult (standardized output) and FilingFetcher (ABC for
country-specific implementations like EDINET, DART, Companies House).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class FilingResult:
    """Standardized output from any filing API."""

    source: str  # e.g. "EDINET", "DART"
    ticker: str

    # Shareholders
    major_shareholders: list[dict] | None = None  # [{name, percent, type}]
    parent_company: dict | None = None  # {name, percent, relationship}

    # Segments
    segments: list[dict] | None = None  # [{name, revenue, op_profit, pct_of_total}]
    geographic_breakdown: list[dict] | None = None  # [{region, revenue, pct_of_total}]

    # Cash Flow
    operating_cash_flow: float | None = None
    ocf_period: str | None = None  # "FY2024", "H1 2025"
    ocf_currency: str | None = None  # "JPY", "KRW"

    # Metadata
    filing_date: str | None = None
    filing_type: str | None = None  # "Annual", "Quarterly"
    filing_url: str | None = None
    data_gaps: list[str] = field(default_factory=list)

    def to_report_string(self) -> str:
        """Format as structured text for LLM consumption.

        Matches the FLA output format: SEGMENT BREAKDOWN, OWNERSHIP STRUCTURE,
        FILING CASH FLOW sections.
        """
        sections = [f"### OFFICIAL FILING DATA ({self.source}) FOR {self.ticker}"]

        if self.filing_date:
            sections.append(f"Filing Date: {self.filing_date}")
        if self.filing_type:
            sections.append(f"Filing Type: {self.filing_type}")
        if self.filing_url:
            sections.append(f"Source: {self.filing_url}")

        sections.append("")

        # Segment Breakdown
        sections.append("**SEGMENT BREAKDOWN**")
        if self.segments:
            sections.append("| Segment | Revenue | Op. Profit | % of Total Rev |")
            sections.append("|---------|---------|-----------|----------------|")
            for seg in self.segments:
                name = seg.get("name", "Unknown")
                revenue = seg.get("revenue", "N/A")
                op_profit = seg.get("op_profit", "N/A")
                pct = seg.get("pct_of_total", "N/A")
                if isinstance(pct, int | float):
                    pct = f"{pct:.1f}%"
                sections.append(f"| {name} | {revenue} | {op_profit} | {pct} |")
            sections.append(f"Source: {self.source} official filing")
        else:
            sections.append("Segment data not found.")

        if self.geographic_breakdown:
            sections.append("\nGeographic Breakdown:")
            for geo in self.geographic_breakdown:
                region = geo.get("region", "Unknown")
                revenue = geo.get("revenue", "N/A")
                pct = geo.get("pct_of_total", "N/A")
                if isinstance(pct, int | float):
                    pct = f"{pct:.1f}%"
                sections.append(f"- {region}: {revenue} ({pct})")

        sections.append("")

        # Ownership Structure
        sections.append("**OWNERSHIP STRUCTURE**")
        if self.major_shareholders:
            for sh in self.major_shareholders:
                name = sh.get("name", "Unknown")
                pct = sh.get("percent", "N/A")
                sh_type = sh.get("type", "")
                if isinstance(pct, int | float):
                    pct = f"{pct:.2f}%"
                type_str = f" ({sh_type})" if sh_type else ""
                sections.append(f"- {name}: {pct}{type_str}")

        if self.parent_company:
            name = self.parent_company.get("name", "Unknown")
            pct = self.parent_company.get("percent", "N/A")
            rel = self.parent_company.get("relationship", "parent")
            if isinstance(pct, int | float):
                pct = f"{pct:.2f}%"
            sections.append(f"- Controlling Shareholder: {name} ({pct})")
            sections.append(f"- Parent Company: {name}")
            sections.append(f"- Relationship: {rel}")
        elif not self.major_shareholders:
            sections.append("Ownership data not found.")

        sections.append("")

        # Filing Cash Flow
        sections.append("**FILING CASH FLOW**")
        if self.operating_cash_flow is not None:
            currency = self.ocf_currency or ""
            period = self.ocf_period or "Unknown period"
            sections.append(
                f"- Operating Cash Flow (Filing): {currency} {self.operating_cash_flow:,.0f}"
            )
            sections.append(f"- Period: {period}")
            sections.append(f"- Source: {self.source} official filing")
        else:
            sections.append("Filing CF not found.")

        # Data gaps
        if self.data_gaps:
            sections.append("")
            sections.append(f"**DATA GAPS**: {', '.join(self.data_gaps)}")

        return "\n".join(sections)


class FilingFetcher(ABC):
    """Base class for country-specific filing API fetchers."""

    @property
    @abstractmethod
    def supported_suffixes(self) -> list[str]:
        """Ticker suffixes this fetcher handles, e.g. ['T'] for Japan."""

    @property
    @abstractmethod
    def source_name(self) -> str:
        """API name, e.g. 'EDINET'."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if API key present and service reachable."""

    @abstractmethod
    async def get_filing_data(self, ticker: str) -> FilingResult | None:
        """Fetch shareholders, segments, and cash flow from official filings."""
