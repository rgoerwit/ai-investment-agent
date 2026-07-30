"""Tests for scripts/cost_report.py (B1): the cost-diff tool over saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.cost_report import (
    aggregate,
    diff_report,
    discover_runs,
    format_report,
    load_run,
)


def _write_run(
    dir_: Path,
    ticker: str,
    *,
    date: str = "2026-07-25T10:00:00",
    with_rollups: bool = True,
    agents: dict[str, float] | None = None,
    by_provider: dict[str, float] | None = None,
    by_model: dict[str, float] | None = None,
    by_tier: dict[str, float] | None = None,
    unpriced: list[str] | None = None,
) -> Path:
    agents = agents or {"Portfolio Manager": 0.10, "Consultant": 0.05}
    token_usage: dict = {
        "total_cost_usd": sum(agents.values()),
        "agents": {n: {"cost_usd": c, "total_tokens": 1000} for n, c in agents.items()},
    }
    if with_rollups:
        token_usage["by_provider"] = {
            k: {"cost_usd": v, "tokens": 1000, "calls": 1}
            for k, v in (by_provider or {"google": 0.10, "openai": 0.05}).items()
        }
        token_usage["by_model"] = {
            k: {"cost_usd": v, "tokens": 1000, "calls": 1}
            for k, v in (by_model or {"gemini-3.1-pro-preview": 0.10}).items()
        }
        token_usage["by_tier"] = {
            k: {"cost_usd": v, "tokens": 1000, "calls": 1}
            for k, v in (by_tier or {"flex": 0.15}).items()
        }
        token_usage["unpriced_models"] = unpriced or []
    payload = {
        "metadata": {"ticker": ticker, "analysis_date": date},
        "run_summary": {"quick_mode": False},
        "token_usage": token_usage,
    }
    path = dir_ / f"{ticker}_{date.replace(':', '').replace('-', '')}_analysis.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestLoadAndAggregate:
    def test_load_run_reads_rollups(self, tmp_path):
        p = _write_run(tmp_path, "AAA.T")
        run = load_run(p)
        assert run is not None
        assert run.ticker == "AAA.T"
        assert run.by_agent == {"Portfolio Manager": 0.10, "Consultant": 0.05}
        assert run.by_model is not None
        assert run.by_tier == {"flex": 0.15}
        assert run.approximate_provider is False

    def test_aggregate_sums_across_runs(self, tmp_path):
        _write_run(tmp_path, "AAA.T", agents={"PM": 0.10})
        _write_run(tmp_path, "BBB.T", date="2026-07-26T10:00:00", agents={"PM": 0.20})
        runs = discover_runs(tmp_path)
        totals, counted = aggregate(runs, "agent")
        assert counted == 2
        assert totals["PM"] == pytest.approx(0.30)

    def test_by_provider_reconciles_to_total(self, tmp_path):
        _write_run(
            tmp_path,
            "AAA.T",
            agents={"PM": 0.10, "Consultant": 0.05},
            by_provider={"google": 0.10, "openai": 0.05},
        )
        runs = discover_runs(tmp_path)
        totals, _ = aggregate(runs, "provider")
        assert sum(totals.values()) == sum(r.total_cost for r in runs)


class TestFilters:
    def test_since_filter(self, tmp_path):
        _write_run(tmp_path, "OLD.T", date="2026-07-01T10:00:00")
        _write_run(tmp_path, "NEW.T", date="2026-07-25T10:00:00")
        runs = discover_runs(tmp_path, since="2026-07-20")
        assert [r.ticker for r in runs] == ["NEW.T"]

    def test_ticker_filter(self, tmp_path):
        _write_run(tmp_path, "AAA.T")
        _write_run(tmp_path, "BBB.T", date="2026-07-26T10:00:00")
        runs = discover_runs(tmp_path, tickers={"BBB.T"})
        assert [r.ticker for r in runs] == ["BBB.T"]


class TestFallbackAndFlags:
    def test_pre_rollup_artifact_falls_back_to_agent_provider_map(self, tmp_path):
        _write_run(
            tmp_path,
            "OLD.T",
            with_rollups=False,
            agents={"Portfolio Manager": 0.10, "Consultant": 0.05},
        )
        run = discover_runs(tmp_path)[0]
        assert run.approximate_provider is True
        assert run.by_model is None  # can't derive per-model cost from old artifact
        # Consultant -> openai_compatible, PM -> google (keyword map)
        assert run.by_provider["openai_compatible"] == 0.05
        assert run.by_provider["google"] == 0.10

    def test_unpriced_models_surface_in_report(self, tmp_path):
        _write_run(tmp_path, "AAA.T", unpriced=["kimi-k9"])
        report = format_report(discover_runs(tmp_path), "agent")
        assert "kimi-k9" in report

    def test_model_dimension_absent_message(self, tmp_path):
        _write_run(tmp_path, "OLD.T", with_rollups=False)
        report = format_report(discover_runs(tmp_path), "model")
        assert "no artifacts carry this rollup" in report


class TestDiff:
    def test_ab_delta_math(self, tmp_path):
        base_dir = tmp_path / "off"
        cand_dir = tmp_path / "on"
        base_dir.mkdir()
        cand_dir.mkdir()
        # Flex OFF: full-rate call, standard tier. Flex ON: same work at half cost.
        _write_run(base_dir, "AAA.T", agents={"PM": 0.20}, by_tier={"standard": 0.20})
        _write_run(cand_dir, "AAA.T", agents={"PM": 0.10}, by_tier={"flex": 0.10})
        base = discover_runs(base_dir)
        cand = discover_runs(cand_dir)
        report = diff_report(base, cand, "tier")
        # candidate is cheaper by $0.10/run overall (the flex discount).
        assert "Δ total: $-0.1000/run" in report
        # ...and the spend moved out of the standard tier into flex.
        assert "standard" in report and "flex" in report
