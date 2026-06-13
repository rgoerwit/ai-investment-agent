"""History-backed ticker resolution suggestions.

This module deliberately suggests candidates only. Auto-adoption belongs to
verified live identity paths such as IBKR or explicit operator overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.ibkr.analysis_index import load_latest_analyses
from src.ibkr.models import AnalysisRecord
from src.ticker_policy import is_safe_symbol_crossmatch_base, split_ticker


@dataclass(frozen=True, slots=True)
class HistoricalCandidate:
    resolved_ticker: str
    currency: str
    exchange: str
    analysis_date: str

    def display_label(self) -> str:
        parts = [self.resolved_ticker]
        details = [
            value
            for value in (self.currency, self.exchange, self.analysis_date)
            if value
        ]
        if details:
            parts.append(f"({', '.join(details)})")
        return " ".join(parts)


def _analysis_age_days(
    analysis_date: str, *, now: datetime | None = None
) -> int | None:
    try:
        parsed = datetime.strptime(analysis_date, "%Y-%m-%d")
    except (TypeError, ValueError):
        return None
    reference = now or datetime.now()
    return (reference - parsed).days


def _is_recent(analysis_date: str, *, max_age_days: int) -> bool:
    age = _analysis_age_days(analysis_date)
    return age is not None and 0 <= age <= max_age_days


def _is_adoption_grade(record: AnalysisRecord, *, max_age_days: int) -> bool:
    data_quality = record.data_quality or {}
    return (
        data_quality.get("basics_ok") is True
        and data_quality.get("data_vacuum") is not True
        and record.current_price is not None
        and record.health_adj is not None
        and record.growth_adj is not None
        and bool(record.verdict)
        and bool(record.currency)
        and bool(record.exchange)
        and _is_recent(record.analysis_date, max_age_days=max_age_days)
    )


def historical_resolution_candidates(
    ticker: str,
    *,
    results_dir: Path,
    max_age_days: int = 120,
) -> list[HistoricalCandidate]:
    """Return reliable same-base historical candidates for operator review."""
    base, suffix = split_ticker(ticker)
    if not is_safe_symbol_crossmatch_base(base):
        return []

    candidates: list[HistoricalCandidate] = []
    analyses = load_latest_analyses(results_dir)
    for record in analyses.values():
        candidate_base, candidate_suffix = split_ticker(record.ticker)
        if candidate_base != base:
            continue
        if candidate_suffix == suffix or record.ticker.upper() == ticker.upper():
            continue
        if not _is_adoption_grade(record, max_age_days=max_age_days):
            continue
        candidates.append(
            HistoricalCandidate(
                resolved_ticker=record.ticker,
                currency=record.currency,
                exchange=record.exchange,
                analysis_date=record.analysis_date,
            )
        )

    if len({candidate.resolved_ticker for candidate in candidates}) > 1:
        return []

    return sorted(
        candidates,
        key=lambda candidate: (
            _analysis_age_days(candidate.analysis_date) or 9999,
            candidate.resolved_ticker,
        ),
    )
