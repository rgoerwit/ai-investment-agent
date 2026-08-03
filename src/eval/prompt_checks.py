from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from src.agents.output_validation import validate_required_output
from src.charts.extractors.pm_block import extract_pm_block, extract_verdict_from_text
from src.charts.extractors.valuation import _extract_params
from src.config import config, validate_environment_variables
from src.data_block_utils import (
    has_parseable_data_block,
    normalize_legacy_data_block_report,
    normalize_structured_block_boundaries,
)
from src.eval.capture_contract import (
    NODE_CAPTURE_SPECS,
    NodeCaptureSpec,
    get_node_capture_spec,
)
from src.eval.scenario_catalog import (
    DEFAULT_SUITE_NAME,
    PromptCheckScenario,
    PromptCheckSuite,
    load_prompt_check_suite,
    load_prompt_check_suite_from_path,
)
from src.ibkr.order_builder import PARSEABLE_TRADE_ACTIONS, parse_trade_block
from src.main import run_analysis
from src.runtime_diagnostics import OPTIONAL_PUBLISHABLE_ARTIFACTS
from src.validators.red_flag_detector import RedFlagDetector


def _optional_prompt_keys() -> frozenset[str]:
    """Nodes whose absence the pipeline itself tolerates.

    Derived from the publication contract rather than hand-maintained: a node is
    optional here exactly when every artifact it owns is optional there. The old
    literal ``{"consultant"}`` had drifted — the Auditor and Valuation Calculator
    are optional for publication, yet a run with the OpenAI cross-check plane
    switched off (a supported configuration; consultant and auditor share one
    gate) reported the Auditor as a suite *failure*. That is the evaluator's model
    of the pipeline going stale, the same defect class as the TRADE_BLOCK action
    vocabulary.
    """
    return frozenset(
        spec.prompt_key
        for spec in NODE_CAPTURE_SPECS.values()
        if spec.prompt_key
        and spec.artifact_fields
        and set(spec.artifact_fields) <= OPTIONAL_PUBLISHABLE_ARTIFACTS
    )


_OPTIONAL_PROMPT_KEYS = _optional_prompt_keys()


@dataclass(frozen=True)
class PromptCheckNodeOutput:
    prompt_key: str
    node_name: str
    artifact_field: str
    artifact_text: str
    ran: bool
    artifact_status: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    scope: str
    passed: bool
    reason: str | None = None


@dataclass(frozen=True)
class NodeCheckReport:
    prompt_key: str
    node_name: str
    artifact_field: str
    source_node_name: str | None
    ran: bool
    skipped: bool
    passed: bool
    checks: tuple[CheckResult, ...]


@dataclass(frozen=True)
class RunCheckReport:
    passed: bool
    node_reports: tuple[NodeCheckReport, ...]


@dataclass(frozen=True)
class QuickFullComparison:
    ticker: str
    quick_verdict: str
    full_verdict: str
    verdict_diverged: bool
    quick_publishable: bool
    full_publishable: bool
    consultant_quick: str
    consultant_full: str


@dataclass(frozen=True)
class PromptCheckScenarioReport:
    ticker: str
    quick: bool
    strict: bool
    passed: bool
    run_report: RunCheckReport | None = None
    error: str | None = None
    quick_full_comparison: QuickFullComparison | None = None


@dataclass(frozen=True)
class PromptCheckSuiteReport:
    suite: str
    description: str
    passed: bool
    scenario_reports: tuple[PromptCheckScenarioReport, ...]


@dataclass(frozen=True)
class PromptCheckScenarioExecution:
    report: PromptCheckScenarioReport
    outputs: Mapping[str, PromptCheckNodeOutput]
    result: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class PromptCheckSuiteExecution:
    suite_report: PromptCheckSuiteReport
    scenario_executions: tuple[PromptCheckScenarioExecution, ...]


def _applicable_scopes(spec: NodeCaptureSpec) -> tuple[str, ...]:
    return tuple(
        scope for scope in spec.evaluator_scope if scope != "artifact_complete"
    )


def _iter_prompt_check_specs() -> list[NodeCaptureSpec]:
    from src.eval.capture_contract import iter_baseline_eligible_specs

    specs: list[NodeCaptureSpec] = []
    seen_prompt_keys: set[str] = set()
    for spec in iter_baseline_eligible_specs():
        scopes = _applicable_scopes(spec)
        if not scopes or not spec.prompt_key:
            continue
        if spec.prompt_key in seen_prompt_keys:
            continue
        seen_prompt_keys.add(spec.prompt_key)
        specs.append(spec)
    return specs


class PromptCheckCollector:
    """Capture node-level artifact outputs in memory during a live analysis run."""

    def __init__(self) -> None:
        self._records_by_prompt_key: dict[str, PromptCheckNodeOutput] = {}

    def wrap_node(self, node_name: str, node: Any) -> Any:
        spec = get_node_capture_spec(node_name)
        prompt_key = spec.prompt_key
        scopes = _applicable_scopes(spec)
        if not spec.baseline_eligible or not prompt_key or not scopes:
            return node

        async def wrapped(state, config):
            result = await node(state, config)
            record = self._record_for_result(spec, node_name, result)
            self._records_by_prompt_key[prompt_key] = record
            return result

        return wrapped

    def _record_for_result(
        self, spec: NodeCaptureSpec, node_name: str, result: Any
    ) -> PromptCheckNodeOutput:
        artifact_field = spec.artifact_fields[0]
        artifact_text = ""
        artifact_status: dict[str, Any] = {}
        if isinstance(result, Mapping):
            raw_text = result.get(artifact_field, "")
            artifact_text = (
                raw_text if isinstance(raw_text, str) else str(raw_text or "")
            )
            raw_statuses = result.get("artifact_statuses", {})
            if isinstance(raw_statuses, Mapping):
                status = raw_statuses.get(artifact_field)
                if isinstance(status, Mapping):
                    artifact_status = dict(status)
        return PromptCheckNodeOutput(
            prompt_key=spec.prompt_key or node_name,
            node_name=node_name,
            artifact_field=artifact_field,
            artifact_text=artifact_text,
            ran=True,
            artifact_status=artifact_status,
        )

    def collected_outputs(self) -> dict[str, PromptCheckNodeOutput]:
        return dict(self._records_by_prompt_key)


def check_data_block_present(text: str) -> tuple[bool, str | None]:
    normalized = normalize_legacy_data_block_report(text) or text
    normalized = normalize_structured_block_boundaries(normalized) or normalized
    ok = has_parseable_data_block(normalized)
    return ok, None if ok else "DATA_BLOCK unparseable after normalization"


def check_value_trap_block_present(text: str) -> tuple[bool, str | None]:
    metrics = RedFlagDetector.extract_value_trap_score(text)
    ok = bool(metrics.get("verdict"))
    return ok, None if ok else "VALUE_TRAP_BLOCK missing or verdict unparseable"


def check_value_trap_score_parseable(text: str) -> tuple[bool, str | None]:
    metrics = RedFlagDetector.extract_value_trap_score(text)
    ok = metrics.get("score") is not None
    return ok, None if ok else "VALUE_TRAP score missing or unparseable"


def check_pm_block_present(text: str) -> tuple[bool, str | None]:
    ok = bool(extract_pm_block(text).verdict)
    return ok, None if ok else "PM_BLOCK missing or unparseable"


def check_pm_verdict_present(text: str) -> tuple[bool, str | None]:
    block = extract_pm_block(text)
    verdict = block.verdict or extract_verdict_from_text(text)
    ok = bool(verdict)
    return ok, None if ok else "Portfolio Manager verdict missing"


def check_legal_json_valid(text: str) -> tuple[bool, str | None]:
    risks = RedFlagDetector.extract_legal_risks(text)
    ok = bool(risks.get("pfic_status")) and bool(risks.get("vie_structure"))
    return ok, None if ok else "legal JSON missing pfic_status or vie_structure"


def check_valuation_params_present(text: str) -> tuple[bool, str | None]:
    params = _extract_params(text)
    ok = params.method is not None
    return ok, None if ok else "VALUATION_PARAMS block missing or unparseable"


def check_trade_block_present(text: str) -> tuple[bool, str | None]:
    trade_block = parse_trade_block(text)
    ok = trade_block is not None and trade_block.action in PARSEABLE_TRADE_ACTIONS
    return ok, None if ok else "TRADE_BLOCK missing or ACTION not parseable"


def check_consultant_verdict_present(text: str) -> tuple[bool, str | None]:
    validation = validate_required_output("consultant", text)
    return (
        validation["ok"],
        None if validation["ok"] else "consultant verdict structure missing",
    )


def check_raw_data_wrapper_complete(text: str) -> tuple[bool, str | None]:
    ok = (
        "### TOOL 1: get_financial_metrics" in text
        and "### TOOL 2: get_fundamental_analysis" in text
        and "=== END RAW DATA ===" in text
    )
    return ok, None if ok else "raw fundamentals wrapper incomplete"


SCOPE_CHECKS: dict[str, Callable[[str], tuple[bool, str | None]]] = {
    "data_block_present": check_data_block_present,
    "value_trap_block_present": check_value_trap_block_present,
    "value_trap_score_parseable": check_value_trap_score_parseable,
    "pm_block_present": check_pm_block_present,
    "pm_verdict_present": check_pm_verdict_present,
    "legal_json_valid": check_legal_json_valid,
    "valuation_params_present": check_valuation_params_present,
    "trade_block_present": check_trade_block_present,
    "consultant_verdict_present": check_consultant_verdict_present,
    "raw_data_wrapper_complete": check_raw_data_wrapper_complete,
}


def run_prompt_checks_on_outputs(
    outputs: Mapping[str, PromptCheckNodeOutput],
) -> RunCheckReport:
    node_reports: list[NodeCheckReport] = []

    for spec in _iter_prompt_check_specs():
        prompt_key = spec.prompt_key or spec.node_name
        collected = outputs.get(prompt_key)
        artifact_field = spec.artifact_fields[0]
        scopes = _applicable_scopes(spec)

        if collected is None:
            skipped = prompt_key in _OPTIONAL_PROMPT_KEYS
            checks = (
                CheckResult(
                    scope="node_ran",
                    passed=skipped,
                    reason=None if skipped else "node did not run",
                ),
            )
            node_reports.append(
                NodeCheckReport(
                    prompt_key=prompt_key,
                    node_name=spec.node_name,
                    artifact_field=artifact_field,
                    source_node_name=None,
                    ran=False,
                    skipped=skipped,
                    passed=skipped,
                    checks=checks,
                )
            )
            continue

        check_results = []
        for scope in scopes:
            check_fn = SCOPE_CHECKS[scope]
            passed, reason = check_fn(collected.artifact_text)
            check_results.append(CheckResult(scope=scope, passed=passed, reason=reason))

        node_reports.append(
            NodeCheckReport(
                prompt_key=prompt_key,
                node_name=spec.node_name,
                artifact_field=artifact_field,
                source_node_name=collected.node_name,
                ran=collected.ran,
                skipped=False,
                passed=all(result.passed for result in check_results),
                checks=tuple(check_results),
            )
        )

    return RunCheckReport(
        passed=all(report.passed for report in node_reports if not report.skipped),
        node_reports=tuple(node_reports),
    )


def _load_suite_manifest(
    path: Path,
) -> tuple[str, str, tuple[PromptCheckScenario, ...]]:
    suite = load_prompt_check_suite_from_path(path)
    return suite.name, suite.description, suite.scenarios


def _extract_pm_verdict(result: Mapping[str, Any]) -> str:
    text = str(result.get("final_trade_decision") or "")
    block = extract_pm_block(text)
    return block.verdict or extract_verdict_from_text(text) or "UNKNOWN"


def _extract_consultant_verdict(outputs: Mapping[str, PromptCheckNodeOutput]) -> str:
    output = outputs.get("consultant")
    if output is None:
        return "UNKNOWN"
    return str(
        RedFlagDetector.parse_consultant_conditions(output.artifact_text).get("verdict")
        or "UNKNOWN"
    )


def _build_quick_full_comparison(
    quick_execution: PromptCheckScenarioExecution,
    full_execution: PromptCheckScenarioExecution,
) -> QuickFullComparison:
    quick_result = quick_execution.result
    full_result = full_execution.result
    quick_verdict = _extract_pm_verdict(quick_result)
    full_verdict = _extract_pm_verdict(full_result)
    return QuickFullComparison(
        ticker=quick_execution.report.ticker,
        quick_verdict=quick_verdict,
        full_verdict=full_verdict,
        verdict_diverged=quick_verdict != full_verdict,
        quick_publishable=bool(
            (quick_result.get("analysis_validity") or {}).get("publishable")
        ),
        full_publishable=bool(
            (full_result.get("analysis_validity") or {}).get("publishable")
        ),
        consultant_quick=_extract_consultant_verdict(quick_execution.outputs),
        consultant_full=_extract_consultant_verdict(full_execution.outputs),
    )


async def _run_single_prompt_check_scenario(
    scenario: PromptCheckScenario,
) -> PromptCheckScenarioExecution:
    collector = PromptCheckCollector()
    try:
        result = await run_analysis(
            scenario.ticker,
            scenario.quick,
            strict_mode=scenario.strict,
            skip_charts=True,
            node_observer=collector,
        )
    except Exception as exc:
        report = PromptCheckScenarioReport(
            ticker=scenario.ticker,
            quick=scenario.quick,
            strict=scenario.strict,
            passed=False,
            error=f"analysis raised {type(exc).__name__}: {exc}",
        )
        return PromptCheckScenarioExecution(report=report, outputs={})
    if not isinstance(result, dict):
        report = PromptCheckScenarioReport(
            ticker=scenario.ticker,
            quick=scenario.quick,
            strict=scenario.strict,
            passed=False,
            error="analysis returned no result",
        )
        return PromptCheckScenarioExecution(report=report, outputs={})

    outputs = collector.collected_outputs()
    run_report = run_prompt_checks_on_outputs(outputs)
    report = PromptCheckScenarioReport(
        ticker=scenario.ticker,
        quick=scenario.quick,
        strict=scenario.strict,
        passed=run_report.passed,
        run_report=run_report,
    )
    return PromptCheckScenarioExecution(report=report, outputs=outputs, result=result)


async def _run_prompt_check_scenario(
    scenario: PromptCheckScenario,
    *,
    compare_full: bool = False,
) -> PromptCheckScenarioExecution:
    quick_execution = await _run_single_prompt_check_scenario(scenario)
    if not compare_full or quick_execution.report.error:
        return quick_execution

    full_execution = await _run_single_prompt_check_scenario(
        replace(scenario, quick=False)
    )
    if full_execution.report.error:
        report = replace(
            quick_execution.report,
            passed=False,
            error=f"full comparison failed: {full_execution.report.error}",
        )
        return PromptCheckScenarioExecution(
            report=report,
            outputs=quick_execution.outputs,
            result=quick_execution.result,
        )

    comparison = _build_quick_full_comparison(quick_execution, full_execution)
    report = replace(
        quick_execution.report,
        passed=quick_execution.report.passed and full_execution.report.passed,
        quick_full_comparison=comparison,
    )
    return PromptCheckScenarioExecution(
        report=report,
        outputs=quick_execution.outputs,
        result=quick_execution.result,
    )


async def run_prompt_check_suite(
    suite: PromptCheckSuite,
    *,
    compare_full: bool = False,
) -> PromptCheckSuiteExecution:
    validate_environment_variables()
    scenario_executions = [
        await _run_prompt_check_scenario(scenario, compare_full=compare_full)
        for scenario in suite.scenarios
    ]
    suite_report = PromptCheckSuiteReport(
        suite=suite.name,
        description=suite.description,
        passed=all(execution.report.passed for execution in scenario_executions),
        scenario_reports=tuple(execution.report for execution in scenario_executions),
    )
    return PromptCheckSuiteExecution(
        suite_report=suite_report,
        scenario_executions=tuple(scenario_executions),
    )


async def run_prompt_check_suite_from_manifest(
    manifest_path: Path,
) -> PromptCheckSuiteReport:
    execution = await run_prompt_check_suite(
        load_prompt_check_suite_from_path(manifest_path)
    )
    return execution.suite_report


def _print_suite_report(report: PromptCheckSuiteReport) -> None:
    print(f"SUITE: {report.suite}")
    if report.description:
        print(report.description)
    print(f"PASS: {sum(1 for item in report.scenario_reports if item.passed)}")
    print(f"FAIL: {sum(1 for item in report.scenario_reports if not item.passed)}")

    for scenario in report.scenario_reports:
        status = "PASS" if scenario.passed else "FAIL"
        print(
            f"\n[{status}] {scenario.ticker} (quick={scenario.quick}, strict={scenario.strict})"
        )
        if scenario.error:
            print(f"  - {scenario.error}")
            continue
        if not scenario.run_report:
            continue
        if scenario.quick_full_comparison is not None:
            comparison = scenario.quick_full_comparison
            diverged = "diverged" if comparison.verdict_diverged else "same"
            print(
                "  - quick/full: "
                f"{comparison.quick_verdict} vs {comparison.full_verdict} "
                f"({diverged})"
            )
        for node_report in scenario.run_report.node_reports:
            if node_report.skipped:
                # Rendered in the same bulleted list as real failures, so say
                # plainly that it did not contribute one.
                print(
                    f"  - {node_report.node_name}: optional - not run "
                    "(excluded from pass/fail)"
                )
                continue
            for check in node_report.checks:
                if not check.passed:
                    print(
                        f"  - {node_report.node_name} / {check.scope}: "
                        f"{check.reason or 'failed'}"
                    )


def _write_json_report(
    path: Path,
    *,
    stage2_report: PromptCheckSuiteReport,
    stage3_report: Any | None = None,
) -> None:
    payload: dict[str, Any] = {"stage2": asdict(stage2_report)}
    if stage3_report is not None:
        payload["stage3"] = asdict(stage3_report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run opt-in deterministic prompt checks on live prompt outputs."
    )
    parser.add_argument(
        "--suite",
        required=False,
        default=None,
        help=f"Suite name under evals/prompt_check_suites/ (default: {DEFAULT_SUITE_NAME}).",
    )
    parser.add_argument(
        "--stage3",
        action="store_true",
        help=(
            "Also compare current outputs against accepted baselines. "
            "Adds up to 6 judge-model calls per ticker in the default setup."
        ),
    )
    parser.add_argument(
        "--compare-full",
        action="store_true",
        help=(
            "For each suite scenario, also run the same ticker in full mode "
            "and record compact quick/full verdict comparison fields."
        ),
    )
    parser.add_argument(
        "--allow-missing-baseline",
        action="store_true",
        help="Mark missing baselines as skipped instead of failing.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Optional model override for the Stage 3 semantic judge. "
            f"Defaults to QUICK_MODEL (currently: {config.quick_think_llm}). "
            "Use an OpenAI model name explicitly if you want the consultant-backed judge path."
        ),
    )
    parser.add_argument(
        "--baseline-warn-age-days",
        type=int,
        default=30,
        help="Warn when an accepted baseline is older than this many days.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON report.",
    )
    return parser


async def _main_async() -> int:
    args = build_arg_parser().parse_args()
    suite = load_prompt_check_suite(args.suite)
    execution = await run_prompt_check_suite(suite, compare_full=args.compare_full)
    report = execution.suite_report
    _print_suite_report(report)
    stage3_report = None
    if args.stage3:
        from src.eval.semantic_judge import (
            print_stage3_suite_report,
            run_stage3_suite,
        )

        stage3_report = await run_stage3_suite(
            execution,
            allow_missing_baseline=args.allow_missing_baseline,
            judge_model=args.judge_model,
            baseline_warn_age_days=args.baseline_warn_age_days,
        )
        print_stage3_suite_report(stage3_report)
    if args.json_output is not None:
        _write_json_report(
            args.json_output,
            stage2_report=report,
            stage3_report=stage3_report,
        )
    if stage3_report is not None:
        return 0 if report.passed and stage3_report.passed else 1
    return 0 if report.passed else 1


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
