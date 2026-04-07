from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

_MARKER_NAME = ".pipeline_last_run.json"
_DEFAULT_STALE_DAYS = 90


@dataclass(frozen=True)
class ScreeningFreshnessSummary:
    status: Literal["fresh", "stale", "missing"]
    screening_date: str | None = None
    completed_at: str | None = None
    age_days: int | None = None
    stale_after_days: int = _DEFAULT_STALE_DAYS
    candidate_count: int | None = None
    buy_count: int | None = None


def load_screening_freshness(
    results_dir: Path,
    *,
    stale_after_days: int = _DEFAULT_STALE_DAYS,
) -> ScreeningFreshnessSummary:
    """Load broad-screen completion metadata from the results directory."""
    marker = results_dir / _MARKER_NAME
    if not marker.exists():
        return ScreeningFreshnessSummary(
            status="missing",
            stale_after_days=stale_after_days,
        )

    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return ScreeningFreshnessSummary(
            status="missing",
            stale_after_days=stale_after_days,
        )

    screening_date = payload.get("screening_date")
    completed_at = payload.get("completed_at")
    anchor = screening_date or (
        completed_at[:10]
        if isinstance(completed_at, str) and len(completed_at) >= 10
        else None
    )
    if not anchor:
        return ScreeningFreshnessSummary(
            status="missing",
            stale_after_days=stale_after_days,
        )

    try:
        age_days = max((date.today() - date.fromisoformat(anchor)).days, 0)
    except ValueError:
        return ScreeningFreshnessSummary(
            status="missing",
            stale_after_days=stale_after_days,
        )

    return ScreeningFreshnessSummary(
        status="stale" if age_days > stale_after_days else "fresh",
        screening_date=screening_date,
        completed_at=completed_at,
        age_days=age_days,
        stale_after_days=stale_after_days,
        candidate_count=_coerce_int(payload.get("candidate_count")),
        buy_count=_coerce_int(payload.get("buy_count")),
    )


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
