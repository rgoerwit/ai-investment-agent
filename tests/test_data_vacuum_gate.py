"""Pre-LLM total-data-vacuum gate in run_analysis.

When startup company-name resolution fails AND a cheap no-LLM probe finds no
source with price/currency/identity, the analysis aborts before any LLM cost
(the 1264.TW delisted/migrated-ticker case). --force-data-vacuum bypasses.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.main import _is_total_data_vacuum, run_analysis


def _fetcher_returning(metrics):
    return SimpleNamespace(get_financial_metrics=AsyncMock(return_value=metrics))


class _HistoryCandidate:
    def __init__(self, label: str):
        self._label = label

    def display_label(self) -> str:
        return self._label


class TestVacuumProbe:
    @pytest.mark.asyncio
    async def test_total_vacuum_detected(self):
        with patch(
            "src.runtime_services.get_current_market_data_fetcher",
            return_value=_fetcher_returning({"error": "no data"}),
        ):
            assert await _is_total_data_vacuum("1264.TW") is True

    @pytest.mark.asyncio
    async def test_any_anchor_passes(self):
        for anchor in (
            {"currentPrice": 50.0},
            {"currency": "TWD"},
            {"longName": "Some Company"},
        ):
            with patch(
                "src.runtime_services.get_current_market_data_fetcher",
                return_value=_fetcher_returning(anchor),
            ):
                assert await _is_total_data_vacuum("2330.TW") is False

    @pytest.mark.asyncio
    async def test_probe_error_fails_open(self):
        fetcher = SimpleNamespace(
            get_financial_metrics=AsyncMock(side_effect=RuntimeError("probe down"))
        )
        with patch(
            "src.runtime_services.get_current_market_data_fetcher",
            return_value=fetcher,
        ):
            assert await _is_total_data_vacuum("2330.TW") is False


class TestGateInRunAnalysis:
    """Gate behavior inside run_analysis (graph construction mocked away)."""

    @staticmethod
    def _unresolved_name():
        return SimpleNamespace(is_resolved=False, preferred_display_name="1264.TW")

    @pytest.mark.asyncio
    async def test_vacuum_aborts_before_llm(self, capsys):
        with (
            patch(
                "src.ticker_utils.resolve_company_name",
                new=AsyncMock(return_value=self._unresolved_name()),
            ),
            patch(
                "src.main._is_total_data_vacuum", new=AsyncMock(return_value=True)
            ) as probe,
            patch(
                "src.ticker_history_resolver.historical_resolution_candidates",
                return_value=[],
            ),
            patch("src.graph.create_trading_graph") as graph_factory,
        ):
            result = await run_analysis("1264.TW", quick_mode=True)

        assert result is None
        probe.assert_awaited_once()
        graph_factory.assert_not_called()  # no LLM graph was built
        out = capsys.readouterr().out
        assert "data vacuum" in out
        assert "1264.TWO" in out  # sibling listing hint
        assert "--force-data-vacuum" in out

    @pytest.mark.asyncio
    async def test_force_flag_skips_probe(self):
        # run_analysis catches the sentinel internally; reaching the graph
        # factory (instead of the gate's early return) is the assertion.
        with (
            patch(
                "src.ticker_utils.resolve_company_name",
                new=AsyncMock(return_value=self._unresolved_name()),
            ),
            patch(
                "src.main._is_total_data_vacuum", new=AsyncMock(return_value=True)
            ) as probe,
            patch(
                "src.graph.create_trading_graph",
                side_effect=RuntimeError("stop here: graph reached"),
            ) as graph_factory,
        ):
            await run_analysis("1264.TW", quick_mode=True, force_data_vacuum=True)

        probe.assert_not_awaited()
        graph_factory.assert_called_once()

    @pytest.mark.asyncio
    async def test_vacuum_abort_prints_history_suggestions_without_redirect(
        self, capsys
    ):
        with (
            patch(
                "src.ticker_utils.resolve_company_name",
                new=AsyncMock(return_value=self._unresolved_name()),
            ),
            patch("src.main._is_total_data_vacuum", new=AsyncMock(return_value=True)),
            patch(
                "src.ticker_history_resolver.historical_resolution_candidates",
                return_value=[_HistoryCandidate("KRN.DE (EUR, DE, 2026-06-11)")],
            ),
            patch("src.graph.create_trading_graph") as graph_factory,
        ):
            result = await run_analysis("KRN", quick_mode=True)

        assert result is None
        graph_factory.assert_not_called()
        out = capsys.readouterr().out
        assert "Prior reliable same-base analysis(es): KRN.DE" in out
        assert "--force-data-vacuum" in out

    @pytest.mark.asyncio
    async def test_resolved_name_skips_probe_entirely(self):
        resolved = SimpleNamespace(is_resolved=True, preferred_display_name="TSMC")
        with (
            patch(
                "src.ticker_utils.resolve_company_name",
                new=AsyncMock(return_value=resolved),
            ),
            patch(
                "src.main._is_total_data_vacuum", new=AsyncMock(return_value=True)
            ) as probe,
            patch(
                "src.graph.create_trading_graph",
                side_effect=RuntimeError("stop here: graph reached"),
            ) as graph_factory,
        ):
            await run_analysis("2330.TW", quick_mode=True)

        probe.assert_not_awaited()
        graph_factory.assert_called_once()
