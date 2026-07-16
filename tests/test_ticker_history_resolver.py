from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.ibkr.models import AnalysisRecord
from src.ticker_history_resolver import historical_resolution_candidates


def _record(
    ticker: str,
    *,
    currency: str = "EUR",
    exchange: str = "DE",
    analysis_date: str | None = None,
    data_quality: dict | None = None,
    current_price: float | None = 100.0,
    health_adj: float | None = 75.0,
    growth_adj: float | None = 65.0,
    verdict: str = "DO_NOT_INITIATE",
) -> AnalysisRecord:
    return AnalysisRecord(
        ticker=ticker,
        analysis_date=analysis_date or datetime.now().strftime("%Y-%m-%d"),
        verdict=verdict,
        health_adj=health_adj,
        growth_adj=growth_adj,
        current_price=current_price,
        currency=currency,
        exchange=exchange,
        data_quality=(
            {"basics_ok": True, "data_vacuum": False}
            if data_quality is None
            else data_quality
        ),
    )


def _patch_index(monkeypatch, records: list[AnalysisRecord]) -> None:
    monkeypatch.setattr(
        "src.ticker_history_resolver.load_latest_analyses",
        lambda _results_dir: {record.ticker: record for record in records},
    )


def test_krn_chooses_modern_reliable_candidate_over_legacy_record(monkeypatch):
    _patch_index(
        monkeypatch,
        [
            _record("KRN.DE", currency="EUR", exchange="DE"),
            _record(
                "KRN.SW",
                currency="CHF",
                exchange="SW",
                data_quality={},
            ),
        ],
    )

    candidates = historical_resolution_candidates("KRN", results_dir=Path("results"))

    assert [candidate.resolved_ticker for candidate in candidates] == ["KRN.DE"]


def test_ags_ambiguous_same_base_candidates_abstain(monkeypatch):
    _patch_index(
        monkeypatch,
        [
            _record("AGS.BR", currency="EUR", exchange="BR"),
            _record("AGS.SI", currency="SGD", exchange="SI"),
        ],
    )

    assert historical_resolution_candidates("AGS", results_dir=Path("results")) == []


def test_numeric_base_is_excluded(monkeypatch):
    _patch_index(
        monkeypatch,
        [_record("1264.TWO", currency="TWD", exchange="TWO")],
    )

    assert (
        historical_resolution_candidates("1264.TW", results_dir=Path("results")) == []
    )


def test_filters_stale_legacy_vacuum_and_incomplete_records(monkeypatch):
    _patch_index(
        monkeypatch,
        [
            _record("BAD1.DE", analysis_date="2000-01-01"),
            _record("BAD2.DE", data_quality={}),
            _record("BAD3.DE", data_quality={"basics_ok": True, "data_vacuum": True}),
            _record("BAD4.DE", current_price=None),
            _record("BAD5.DE", health_adj=None),
            _record("BAD6.DE", growth_adj=None),
            _record("BAD7.DE", verdict=""),
        ],
    )

    for ticker in ("BAD1", "BAD2", "BAD3", "BAD4", "BAD5", "BAD6", "BAD7"):
        assert (
            historical_resolution_candidates(ticker, results_dir=Path("results")) == []
        )


def test_minor_unit_currency_candidate_is_valid(monkeypatch):
    _patch_index(
        monkeypatch,
        [_record("MEGP.L", currency="GBX", exchange="L")],
    )

    candidates = historical_resolution_candidates("MEGP", results_dir=Path("results"))

    assert len(candidates) == 1
    assert candidates[0].resolved_ticker == "MEGP.L"
    assert candidates[0].currency == "GBX"
