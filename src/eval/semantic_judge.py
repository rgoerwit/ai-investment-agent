from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from src.eval.capture_contract import get_node_capture_spec, iter_stage3_judge_specs
from src.eval.constants import CURRENT_CAPTURE_SCHEMA_VERSION
from src.eval.prompt_checks import (
    PromptCheckScenarioExecution,
    PromptCheckScenarioReport,
    PromptCheckSuiteExecution,
)
from src.llms import create_consultant_llm, create_quick_thinking_llm

JudgeVerdict = Literal["PASS", "SOFT_FAIL", "HARD_FAIL", "SKIPPED"]

_RUBRIC_PROMPTS: dict[str, str] = {
    "analysis_report": """Return JSON only.
Evaluate CURRENT against BASELINE for an investment analysis artifact.

PASS when:
- a company-specific thesis is still present
- at least two company-specific risks, caveats, or counterpoints remain
- the overall conclusion does not contradict the cited evidence

SOFT_FAIL when:
- the thesis remains but specificity is materially thinner
- one important risk/caveat is lost
- the conclusion is present but the evidence chain is weaker

HARD_FAIL when:
- no clear thesis remains
- the artifact becomes generic or ticker-agnostic
- the conclusion contradicts the cited evidence

Respond as JSON with keys: verdict, score, signals.""",
    "legal_structured": """Return JSON only.
Evaluate CURRENT against BASELINE for a legal/risk structured artifact.

PASS when:
- pfic_status is present and valid
- vie_structure is present and valid
- the explanation does not contradict the structured fields

SOFT_FAIL when:
- the structured fields remain but the explanation is materially thinner
- one non-critical clarification is lost

HARD_FAIL when:
- pfic_status or vie_structure is missing or invalid
- the explanation contradicts the structured fields

Respond as JSON with keys: verdict, score, signals.""",
    "risk_structured": """Return JSON only.
Evaluate CURRENT against BASELINE for a scored risk artifact.

PASS when:
- the score is parseable
- the verdict aligns with the score
- a concrete risk explanation remains present

SOFT_FAIL when:
- score and verdict still align but the explanation is thinner
- one risk explanation is lost without contradiction

HARD_FAIL when:
- score is not parseable
- verdict does not align with score
- the output no longer explains the risk call

Respond as JSON with keys: verdict, score, signals.""",
    "valuation_structured": """Return JSON only.
Evaluate CURRENT against BASELINE for a valuation artifact.

PASS when:
- valuation method is still present
- confidence is still present
- current-vs-target framing is coherent

SOFT_FAIL when:
- structure remains but the assumptions are thinner
- confidence or framing is less specific

HARD_FAIL when:
- method is missing
- upside/downside framing inverts without stated changed assumption
- the artifact is not valuation-usable

Respond as JSON with keys: verdict, score, signals.""",
    "trade_decision": """Return JSON only.
Evaluate CURRENT against BASELINE for a trade decision artifact.

PASS when:
- action remains parseable
- entry/stop/target are coherent
- if action changed, the changed premise is explicitly stated

SOFT_FAIL when:
- action is parseable but the reasoning is materially thinner
- one numeric field or one explanation is weaker but still usable

HARD_FAIL when:
- action is missing or unusable
- entry/stop/target relationship is incoherent
- action changed with no explicit changed premise

Respond as JSON with keys: verdict, score, signals.""",
    "portfolio_decision": """Return JSON only.
Evaluate CURRENT against BASELINE for a portfolio-manager artifact.

PASS when:
- PM verdict remains parseable
- PM_BLOCK remains coherent
- if verdict changed, the changed premise is explicit

SOFT_FAIL when:
- verdict remains parseable but justification is materially thinner
- one important supporting point is lost

HARD_FAIL when:
- PM verdict is missing
- PM_BLOCK is unusable
- verdict changed without explicit changed risk/valuation premise

Respond as JSON with keys: verdict, score, signals.""",
}


@dataclass(frozen=True)
class JudgeResult:
    prompt_key: str
    rubric_family: str
    verdict: JudgeVerdict
    score: float
    signals: tuple[str, ...] = ()
    baseline_run_id: str | None = None
    raw_response: str | None = None


@dataclass(frozen=True)
class Stage3ScenarioReport:
    stage2_report: PromptCheckScenarioReport
    judge_results: tuple[JudgeResult, ...]
    baseline_run_id: str | None
    baseline_trade_date: str | None
    baseline_age_days: int | None
    passed: bool
    warning: str | None = None
    error: str | None = None

    @property
    def ticker(self) -> str:
        return self.stage2_report.ticker

    @property
    def quick(self) -> bool:
        return self.stage2_report.quick

    @property
    def strict(self) -> bool:
        return self.stage2_report.strict


@dataclass(frozen=True)
class Stage3SuiteReport:
    suite: str
    passed: bool
    scenario_reports: tuple[Stage3ScenarioReport, ...]


@dataclass(frozen=True)
class BaselineArtifact:
    prompt_key: str
    node_name: str
    artifact_field: str
    artifact_text: str
    run_dir: Path
    run_id: str
    trade_date: str | None
    age_days: int | None


class SemanticJudge:
    def __init__(self, *, model: str | None = None) -> None:
        self._model = model
        self._llm = self._create_llm(model)

    def _create_llm(self, model: str | None):
        if model and _looks_like_openai_model(model):
            return create_consultant_llm(
                model=model,
                quick_mode=True,
                max_completion_tokens=2048,
            )
        return create_quick_thinking_llm(
            model=model,
            max_output_tokens=2048,
        )

    async def judge_artifact(
        self,
        *,
        prompt_key: str,
        rubric_family: str,
        baseline_text: str,
        current_text: str,
        baseline_run_id: str | None = None,
    ) -> JudgeResult:
        if not baseline_text.strip() or not current_text.strip():
            return JudgeResult(
                prompt_key=prompt_key,
                rubric_family=rubric_family,
                verdict="SKIPPED",
                score=0.0,
                signals=("baseline or current artifact text empty",),
                baseline_run_id=baseline_run_id,
            )

        rubric = _RUBRIC_PROMPTS.get(rubric_family)
        if rubric is None:
            return JudgeResult(
                prompt_key=prompt_key,
                rubric_family=rubric_family,
                verdict="SKIPPED",
                score=0.0,
                signals=(f"unknown rubric_family: {rubric_family}",),
                baseline_run_id=baseline_run_id,
            )

        user_text = (
            "BASELINE:\n"
            f"{_clip_text_for_judge(baseline_text)}\n\n"
            "CURRENT:\n"
            f"{_clip_text_for_judge(current_text)}"
        )
        response = await self._llm.ainvoke(
            [SystemMessage(content=rubric), HumanMessage(content=user_text)]
        )
        raw_response = _message_text(response)
        parsed = self._parse_response(raw_response)
        return JudgeResult(
            prompt_key=prompt_key,
            rubric_family=rubric_family,
            verdict=parsed["verdict"],
            score=parsed["score"],
            signals=parsed["signals"],
            baseline_run_id=baseline_run_id,
            raw_response=raw_response,
        )

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()
        try:
            payload = json.loads(text)
        except Exception as exc:
            return {
                "verdict": "HARD_FAIL",
                "score": 0.0,
                "signals": (f"judge response unparseable: {exc}",),
            }

        verdict = str(payload.get("verdict", "")).upper()
        if verdict not in {"PASS", "SOFT_FAIL", "HARD_FAIL", "SKIPPED"}:
            verdict = "HARD_FAIL"
        try:
            score = float(payload.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        raw_signals = payload.get("signals", [])
        if isinstance(raw_signals, list):
            signals = tuple(str(item) for item in raw_signals)
        else:
            signals = ("signals field missing or malformed",)
        return {"verdict": verdict, "score": score, "signals": signals}


async def run_stage3_suite(
    execution: PromptCheckSuiteExecution,
    *,
    allow_missing_baseline: bool = False,
    judge_model: str | None = None,
    baseline_warn_age_days: int = 30,
) -> Stage3SuiteReport:
    judge = SemanticJudge(model=judge_model)
    scenario_reports = [
        await _run_stage3_scenario(
            judge,
            scenario_execution,
            allow_missing_baseline=allow_missing_baseline,
            baseline_warn_age_days=baseline_warn_age_days,
        )
        for scenario_execution in execution.scenario_executions
    ]
    return Stage3SuiteReport(
        suite=execution.suite_report.suite,
        passed=all(report.passed for report in scenario_reports),
        scenario_reports=tuple(scenario_reports),
    )


async def _run_stage3_scenario(
    judge: SemanticJudge,
    execution: PromptCheckScenarioExecution,
    *,
    allow_missing_baseline: bool,
    baseline_warn_age_days: int,
) -> Stage3ScenarioReport:
    stage2_report = execution.report
    if stage2_report.error:
        return Stage3ScenarioReport(
            stage2_report=stage2_report,
            judge_results=(),
            baseline_run_id=None,
            baseline_trade_date=None,
            baseline_age_days=None,
            passed=False,
            error=stage2_report.error,
        )

    baseline_artifacts = _load_latest_baseline_artifacts(
        ticker=stage2_report.ticker,
        quick=stage2_report.quick,
        strict=stage2_report.strict,
    )
    if baseline_artifacts is None:
        message = (
            f"Missing accepted baseline for {stage2_report.ticker} "
            f"(quick={stage2_report.quick}, strict={stage2_report.strict}). "
            "Run: poetry run python -m src.eval.baseline_suite --suite smoke"
        )
        if allow_missing_baseline:
            return Stage3ScenarioReport(
                stage2_report=stage2_report,
                judge_results=(),
                baseline_run_id=None,
                baseline_trade_date=None,
                baseline_age_days=None,
                passed=True,
                warning=message,
            )
        return Stage3ScenarioReport(
            stage2_report=stage2_report,
            judge_results=(),
            baseline_run_id=None,
            baseline_trade_date=None,
            baseline_age_days=None,
            passed=False,
            error=message,
        )

    node_reports = {
        node_report.prompt_key: node_report
        for node_report in (
            stage2_report.run_report.node_reports if stage2_report.run_report else ()
        )
    }

    judge_results: list[JudgeResult] = []
    for spec in iter_stage3_judge_specs():
        prompt_key = spec.prompt_key
        if not prompt_key:
            continue
        node_report = node_reports.get(prompt_key)
        if node_report is None:
            judge_results.append(
                JudgeResult(
                    prompt_key=prompt_key,
                    rubric_family=spec.rubric_family or "unknown",
                    verdict="HARD_FAIL",
                    score=0.0,
                    signals=("stage2 node report missing for stage3-enabled prompt",),
                    baseline_run_id=baseline_artifacts["run_id"],
                )
            )
            continue
        if node_report.skipped:
            judge_results.append(
                JudgeResult(
                    prompt_key=prompt_key,
                    rubric_family=spec.rubric_family or "unknown",
                    verdict="SKIPPED",
                    score=0.0,
                    signals=("stage2 node skipped",),
                    baseline_run_id=baseline_artifacts["run_id"],
                )
            )
            continue
        if not node_report.passed:
            judge_results.append(
                JudgeResult(
                    prompt_key=prompt_key,
                    rubric_family=spec.rubric_family or "unknown",
                    verdict="HARD_FAIL",
                    score=0.0,
                    signals=("stage2 structural checks failed",),
                    baseline_run_id=baseline_artifacts["run_id"],
                )
            )
            continue

        current_output = execution.outputs.get(prompt_key)
        artifact = baseline_artifacts["artifacts"].get(prompt_key)
        if artifact is None:
            judge_results.append(
                JudgeResult(
                    prompt_key=prompt_key,
                    rubric_family=spec.rubric_family or "unknown",
                    verdict="HARD_FAIL",
                    score=0.0,
                    signals=("baseline_artifact_missing",),
                    baseline_run_id=baseline_artifacts["run_id"],
                )
            )
            continue
        current_text = current_output.artifact_text if current_output else ""
        judge_results.append(
            await judge.judge_artifact(
                prompt_key=prompt_key,
                rubric_family=spec.rubric_family or "unknown",
                baseline_text=artifact.artifact_text,
                current_text=current_text,
                baseline_run_id=artifact.run_id,
            )
        )

    warning = None
    baseline_age_days = baseline_artifacts["age_days"]
    if baseline_age_days is not None and baseline_age_days > baseline_warn_age_days:
        warning = (
            f"Accepted baseline is {baseline_age_days} day(s) old "
            f"(warn threshold: {baseline_warn_age_days})."
        )

    passed = all(result.verdict not in {"HARD_FAIL"} for result in judge_results)
    return Stage3ScenarioReport(
        stage2_report=stage2_report,
        judge_results=tuple(judge_results),
        baseline_run_id=baseline_artifacts["run_id"],
        baseline_trade_date=baseline_artifacts["trade_date"],
        baseline_age_days=baseline_age_days,
        passed=passed,
        warning=warning,
    )


def print_stage3_suite_report(report: Stage3SuiteReport) -> None:
    print("\nSTAGE 3: semantic baseline comparison")
    print(f"PASS: {sum(1 for item in report.scenario_reports if item.passed)}")
    print(f"FAIL: {sum(1 for item in report.scenario_reports if not item.passed)}")
    for scenario in report.scenario_reports:
        status = "PASS" if scenario.passed else "FAIL"
        print(f"\n[{status}] {scenario.ticker}")
        if scenario.error:
            print(f"  - {scenario.error}")
            continue
        if scenario.warning:
            print(f"  - warning: {scenario.warning}")
        for result in scenario.judge_results:
            if result.verdict == "PASS":
                continue
            signals = "; ".join(result.signals) if result.signals else "no signals"
            print(
                f"  - {result.prompt_key} / {result.rubric_family}: "
                f"{result.verdict} ({signals})"
            )


def _load_latest_baseline_artifacts(
    *,
    ticker: str,
    quick: bool,
    strict: bool,
) -> dict[str, Any] | None:
    accepted_root = (
        Path("evals") / "captures" / CURRENT_CAPTURE_SCHEMA_VERSION / "accepted"
    )
    if not accepted_root.exists():
        return None

    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    for manifest_path in accepted_root.glob("*/*/run_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("capture_status") != "accepted":
            continue
        if not manifest.get("usable_for_replay"):
            continue
        if manifest.get("ticker") != ticker:
            continue
        mode = manifest.get("mode", {})
        if bool(mode.get("quick")) != quick or bool(mode.get("strict")) != strict:
            continue
        sort_key = str(
            manifest.get("finalized_at_utc")
            or manifest.get("capture_timestamp_utc")
            or ""
        )
        candidates.append((sort_key, manifest_path.parent, manifest))

    if not candidates:
        return None

    _, run_dir, manifest = max(candidates, key=lambda item: item[0])
    artifacts: dict[str, BaselineArtifact] = {}
    agents_dir = run_dir / "agents"
    if agents_dir.exists():
        for agent_dir in agents_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            metadata_path = agent_dir / "metadata.json"
            output_path = agent_dir / "output.json"
            if not metadata_path.exists() or not output_path.exists():
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            output = json.loads(output_path.read_text(encoding="utf-8"))
            prompt = metadata.get("prompt", {})
            prompt_key = prompt.get("prompt_key")
            node_name = metadata.get("node_name")
            if not prompt_key or not isinstance(node_name, str):
                continue
            spec = get_node_capture_spec(node_name)
            if not spec.artifact_fields:
                continue
            artifact_field = spec.artifact_fields[0]
            raw_text = output.get(artifact_field, "")
            artifacts[prompt_key] = BaselineArtifact(
                prompt_key=prompt_key,
                node_name=node_name,
                artifact_field=artifact_field,
                artifact_text=raw_text
                if isinstance(raw_text, str)
                else str(raw_text or ""),
                run_dir=run_dir,
                run_id=str(manifest.get("run_id") or run_dir.name),
                trade_date=manifest.get("trade_date"),
                age_days=_baseline_age_days(manifest.get("trade_date")),
            )

    return {
        "run_id": str(manifest.get("run_id") or run_dir.name),
        "trade_date": manifest.get("trade_date"),
        "age_days": _baseline_age_days(manifest.get("trade_date")),
        "artifacts": artifacts,
    }


def _baseline_age_days(trade_date: str | None) -> int | None:
    if not trade_date:
        return None
    try:
        trade = datetime.fromisoformat(trade_date).date()
    except ValueError:
        return None
    return (datetime.now(UTC).date() - trade).days


def _clip_text_for_judge(text: str, *, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    head = text[:half]
    tail = text[-half:]
    return f"{head}\n\n[... truncated ...]\n\n{tail}"


def _looks_like_openai_model(model_name: str) -> bool:
    return model_name.startswith("gpt-") or model_name.startswith("o")


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)
