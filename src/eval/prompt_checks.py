from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.agents.output_validation import validate_required_output
from src.charts.extractors.pm_block import extract_pm_block, extract_verdict_from_text
from src.charts.extractors.valuation import _extract_params
from src.config import validate_environment_variables
from src.data_block_utils import (
    has_parseable_data_block,
    normalize_legacy_data_block_report,
    normalize_structured_block_boundaries,
)
from src.eval.capture_contract import NodeCaptureSpec, get_node_capture_spec
from src.ibkr.order_builder import parse_trade_block
from src.main import run_analysis
from src.validators.red_flag_detector import RedFlagDetector

_OPTIONAL_PROMPT_KEYS = frozenset({"consultant"})


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
class PromptCheckScenario:
    ticker: str
    quick: bool = True
    strict: bool = False


@dataclass(frozen=True)
class PromptCheckScenarioReport:
    ticker: str
    quick: bool
    strict: bool
    passed: bool
    run_report: RunCheckReport | None = None
    error: str | None = None


@dataclass(frozen=True)
class PromptCheckSuiteReport:
    suite: str
    description: str
    passed: bool
    scenario_reports: tuple[PromptCheckScenarioReport, ...]


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
    ok = trade_block is not None and trade_block.action in {
        "BUY",
        "SELL",
        "HOLD",
        "REJECT",
    }
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    suite = str(payload.get("suite") or path.stem)
    description = str(payload.get("description") or "")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError(f"Suite manifest has no scenarios: {path}")

    scenarios = []
    for raw in raw_scenarios:
        if not isinstance(raw, Mapping) or not raw.get("ticker"):
            raise ValueError(f"Invalid scenario entry in {path}: {raw!r}")
        scenarios.append(
            PromptCheckScenario(
                ticker=str(raw["ticker"]),
                quick=bool(raw.get("quick", True)),
                strict=bool(raw.get("strict", False)),
            )
        )
    return suite, description, tuple(scenarios)


async def _run_prompt_check_scenario(
    scenario: PromptCheckScenario,
) -> PromptCheckScenarioReport:
    collector = PromptCheckCollector()
    result = await run_analysis(
        scenario.ticker,
        scenario.quick,
        strict_mode=scenario.strict,
        skip_charts=True,
        node_observer=collector,
    )
    if not isinstance(result, dict):
        return PromptCheckScenarioReport(
            ticker=scenario.ticker,
            quick=scenario.quick,
            strict=scenario.strict,
            passed=False,
            error="analysis returned no result",
        )

    run_report = run_prompt_checks_on_outputs(collector.collected_outputs())
    return PromptCheckScenarioReport(
        ticker=scenario.ticker,
        quick=scenario.quick,
        strict=scenario.strict,
        passed=run_report.passed,
        run_report=run_report,
    )


async def run_prompt_check_suite_from_manifest(
    manifest_path: Path,
) -> PromptCheckSuiteReport:
    validate_environment_variables()
    suite, description, scenarios = _load_suite_manifest(manifest_path)
    scenario_reports = [
        await _run_prompt_check_scenario(scenario) for scenario in scenarios
    ]
    return PromptCheckSuiteReport(
        suite=suite,
        description=description,
        passed=all(report.passed for report in scenario_reports),
        scenario_reports=tuple(scenario_reports),
    )


def _default_suite_dir() -> Path:
    return Path("evals") / "prompt_check_suites"


def _resolve_suite_manifest(suite_name: str) -> Path:
    candidate = _default_suite_dir() / f"{suite_name}.json"
    if not candidate.exists():
        raise FileNotFoundError(f"Prompt-check suite not found: {candidate}")
    return candidate


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
        for node_report in scenario.run_report.node_reports:
            if node_report.skipped:
                print(f"  - {node_report.node_name}: skipped")
                continue
            for check in node_report.checks:
                if not check.passed:
                    print(
                        f"  - {node_report.node_name} / {check.scope}: "
                        f"{check.reason or 'failed'}"
                    )


def _write_json_report(path: Path, report: PromptCheckSuiteReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run opt-in deterministic prompt checks on live prompt outputs."
    )
    parser.add_argument(
        "--suite", required=True, help="Suite name under evals/prompt_check_suites/"
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
    report = await run_prompt_check_suite_from_manifest(
        _resolve_suite_manifest(args.suite)
    )
    _print_suite_report(report)
    if args.json_output is not None:
        _write_json_report(args.json_output, report)
    return 0 if report.passed else 1


def main() -> int:
    return asyncio.run(_main_async())


if __name__ == "__main__":
    raise SystemExit(main())
