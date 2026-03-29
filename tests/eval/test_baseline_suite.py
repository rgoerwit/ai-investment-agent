from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.eval.baseline_capture import BaselinePreflightResult, CaptureCleanupSummary
from src.eval.baseline_suite import build_arg_parser_for_suite, run_baseline_suite


def test_batch_baseline_parser_defaults_to_smoke():
    args = build_arg_parser_for_suite().parse_args([])
    assert args.suite == "smoke"
    assert args.json_output is None


@pytest.mark.asyncio
async def test_run_baseline_suite_uses_shared_default_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, bool, bool, bool]] = []
    preflight_calls: list[str] = []

    def fake_preflight():
        preflight_calls.append("called")
        return BaselinePreflightResult(
            git_clean=True,
            cleanup_summary=CaptureCleanupSummary(
                scanned=0,
                moved_to_rejected=0,
                removed_empty=0,
                rejected_paths=(),
            ),
        )

    async def fake_run_with_args(
        args,
        *,
        perform_capture_preflight: bool = True,
        capture_preflight_override=None,
    ):
        calls.append(
            (
                args.ticker,
                bool(args.quick),
                bool(args.strict),
                perform_capture_preflight,
            )
        )
        assert isinstance(capture_preflight_override, BaselinePreflightResult)
        return 0

    monkeypatch.setattr(
        "src.eval.baseline_suite._run_repo_level_preflight", fake_preflight
    )
    monkeypatch.setattr("src.eval.baseline_suite.run_with_args", fake_run_with_args)

    report = await run_baseline_suite(None)
    assert report.passed is True
    assert preflight_calls == ["called"]
    assert [item[0] for item in calls] == [
        "AAPL",
        "ASML.AS",
        "SAP.DE",
        "7203.T",
        "2330.TW",
        "0005.HK",
    ]
    assert all(item[3] is False for item in calls)


@pytest.mark.asyncio
async def test_run_baseline_suite_aggregates_failures(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_preflight():
        return BaselinePreflightResult(
            git_clean=True,
            cleanup_summary=CaptureCleanupSummary(
                scanned=0,
                moved_to_rejected=0,
                removed_empty=0,
                rejected_paths=(),
            ),
        )

    async def fake_run_with_args(
        args,
        *,
        perform_capture_preflight: bool = True,
        capture_preflight_override=None,
    ):
        return 1 if args.ticker == "SAP.DE" else 0

    monkeypatch.setattr(
        "src.eval.baseline_suite._run_repo_level_preflight", fake_preflight
    )
    monkeypatch.setattr("src.eval.baseline_suite.run_with_args", fake_run_with_args)

    report = await run_baseline_suite("smoke")
    assert report.passed is False
    failed = [item.ticker for item in report.scenario_reports if not item.passed]
    assert failed == ["SAP.DE"]


def test_repo_level_preflight_aborts_dirty_tree(monkeypatch: pytest.MonkeyPatch):
    class FakeManager:
        def __init__(self, config):
            self.config = config

        def cleanup_stale_inflight_runs(self):
            return CaptureCleanupSummary(
                scanned=1,
                moved_to_rejected=0,
                removed_empty=0,
                rejected_paths=(),
            )

        def preflight_git_clean(self):
            return False, ["Baseline capture requires a clean local git worktree."]

    monkeypatch.setattr("src.eval.baseline_suite.BaselineCaptureManager", FakeManager)

    from src.eval import baseline_suite as module

    with pytest.raises(SystemExit) as exc:
        module._run_repo_level_preflight()
    assert exc.value.code == 1
