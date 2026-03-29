from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.eval.baseline_capture import (
    BaselineCaptureConfig,
    BaselineCaptureManager,
    BaselinePreflightResult,
)
from src.eval.constants import CURRENT_CAPTURE_SCHEMA_VERSION
from src.eval.scenario_catalog import DEFAULT_SUITE_NAME, load_prompt_check_suite
from src.main import build_arg_parser, run_with_args


@dataclass(frozen=True)
class BaselineSuiteScenarioReport:
    ticker: str
    quick: bool
    strict: bool
    passed: bool


@dataclass(frozen=True)
class BaselineSuiteReport:
    suite: str
    passed: bool
    scenario_reports: tuple[BaselineSuiteScenarioReport, ...]


def build_arg_parser_for_suite() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed baseline captures for a shared prompt-check suite."
    )
    parser.add_argument(
        "--suite",
        default=DEFAULT_SUITE_NAME,
        help=f"Suite name under evals/prompt_check_suites/ (default: {DEFAULT_SUITE_NAME}).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable batch summary.",
    )
    return parser


async def run_baseline_suite(suite_name: str | None = None) -> BaselineSuiteReport:
    suite = load_prompt_check_suite(suite_name)
    preflight_override = _run_repo_level_preflight()

    scenario_reports: list[BaselineSuiteScenarioReport] = []
    for scenario in suite.scenarios:
        exit_code = await run_with_args(
            _build_capture_args(
                ticker=scenario.ticker,
                quick=scenario.quick,
                strict=scenario.strict,
            ),
            perform_capture_preflight=False,
            capture_preflight_override=preflight_override,
        )
        scenario_reports.append(
            BaselineSuiteScenarioReport(
                ticker=scenario.ticker,
                quick=scenario.quick,
                strict=scenario.strict,
                passed=exit_code == 0,
            )
        )

    return BaselineSuiteReport(
        suite=suite.name,
        passed=all(report.passed for report in scenario_reports),
        scenario_reports=tuple(scenario_reports),
    )


def _run_repo_level_preflight() -> BaselinePreflightResult:
    manager = BaselineCaptureManager(
        BaselineCaptureConfig(
            enabled=True,
            schema_version=CURRENT_CAPTURE_SCHEMA_VERSION,
            output_root=Path("evals") / "captures",
        )
    )
    cleanup_summary = manager.cleanup_stale_inflight_runs()
    if cleanup_summary.moved_to_rejected or cleanup_summary.removed_empty:
        print(
            "Cleaned "
            f"{cleanup_summary.moved_to_rejected} stale inflight capture(s)"
            f" and removed {cleanup_summary.removed_empty} empty inflight directory(ies)."
        )
    ok, errors = manager.preflight_git_clean()
    if not ok:
        for error in errors:
            print(error)
        raise SystemExit(1)
    return BaselinePreflightResult(
        git_clean=True,
        cleanup_summary=cleanup_summary,
    )


def _build_capture_args(
    *,
    ticker: str,
    quick: bool,
    strict: bool,
) -> argparse.Namespace:
    argv = ["--ticker", ticker, "--capture-baseline"]
    if quick:
        argv.append("--quick")
    if strict:
        argv.append("--strict")
    return build_arg_parser().parse_args(argv)


def _print_suite_report(report: BaselineSuiteReport) -> None:
    print(f"SUITE: {report.suite}")
    print(f"PASS: {sum(1 for item in report.scenario_reports if item.passed)}")
    print(f"FAIL: {sum(1 for item in report.scenario_reports if not item.passed)}")
    for scenario in report.scenario_reports:
        status = "PASS" if scenario.passed else "FAIL"
        print(
            f"  - [{status}] {scenario.ticker} "
            f"(quick={scenario.quick}, strict={scenario.strict})"
        )


def _write_json_report(path: Path, report: BaselineSuiteReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


async def _main_async() -> int:
    args = build_arg_parser_for_suite().parse_args()
    report = await run_baseline_suite(args.suite)
    _print_suite_report(report)
    if args.json_output is not None:
        _write_json_report(args.json_output, report)
    return 0 if report.passed else 1


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
