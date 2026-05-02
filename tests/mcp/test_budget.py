from __future__ import annotations

from pathlib import Path

from src.mcp.budget import BudgetTracker
from src.mcp.config import MCPServerSpec


def test_budget_tracker_enforces_per_run_and_daily_limits(tmp_path: Path):
    tracker = BudgetTracker(str(tmp_path / "mcp_usage.db"))
    spec = MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url="https://example.test/mcp",
        daily_call_limit=2,
        per_run_limit=1,
    )

    assert tracker.can_call("fmp_remote", spec) is True
    tracker.record_upstream_consumption("fmp_remote")
    assert tracker.can_call("fmp_remote", spec) is False


def test_budget_tracker_persists_daily_counts(tmp_path: Path):
    db_path = tmp_path / "mcp_usage.db"
    tracker = BudgetTracker(str(db_path))
    spec = MCPServerSpec(
        id="fmp_remote",
        description="FMP",
        transport="streamable_http",
        base_url="https://example.test/mcp",
        daily_call_limit=1,
    )
    tracker.record_upstream_consumption("fmp_remote")

    fresh_tracker = BudgetTracker(str(db_path))
    assert fresh_tracker.can_call("fmp_remote", spec) is False
