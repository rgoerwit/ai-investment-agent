from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.capture_contract import get_node_capture_spec, iter_stage3_judge_specs
from src.eval.constants import CURRENT_CAPTURE_SCHEMA_VERSION
from src.eval.prompt_checks import (
    CheckResult,
    NodeCheckReport,
    PromptCheckNodeOutput,
    PromptCheckScenarioExecution,
    PromptCheckScenarioReport,
    PromptCheckSuiteExecution,
    PromptCheckSuiteReport,
    RunCheckReport,
)
from src.eval.semantic_judge import JudgeResult, SemanticJudge, run_stage3_suite


def _write_accepted_capture(
    root: Path,
    *,
    ticker: str,
    quick: bool,
    strict: bool,
    run_id: str = "run-1",
    trade_date: str = "2026-03-28",
    prompt_key: str = "portfolio_manager",
    node_name: str = "Portfolio Manager",
    artifact_field: str = "final_trade_decision",
    artifact_text: str = "baseline artifact",
) -> Path:
    run_dir = root / trade_date / run_id
    agent_dir = run_dir / "agents" / "portfolio_manager"
    agent_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "capture_status": "accepted",
                "usable_for_replay": True,
                "run_id": run_id,
                "ticker": ticker,
                "trade_date": trade_date,
                "capture_timestamp_utc": f"{trade_date}T12:00:00Z",
                "finalized_at_utc": f"{trade_date}T12:05:00Z",
                "mode": {"quick": quick, "strict": strict},
            }
        ),
        encoding="utf-8",
    )
    (agent_dir / "metadata.json").write_text(
        json.dumps(
            {
                "node_name": node_name,
                "prompt": {"prompt_key": prompt_key},
            }
        ),
        encoding="utf-8",
    )
    (agent_dir / "output.json").write_text(
        json.dumps({artifact_field: artifact_text}),
        encoding="utf-8",
    )
    return run_dir


def _stage2_execution(
    *,
    ticker: str = "7203.T",
    prompt_key: str = "portfolio_manager",
    passed: bool = True,
    artifact_text: str = "current artifact",
) -> PromptCheckSuiteExecution:
    report = PromptCheckScenarioReport(
        ticker=ticker,
        quick=True,
        strict=False,
        passed=passed,
        run_report=RunCheckReport(
            passed=passed,
            node_reports=(
                NodeCheckReport(
                    prompt_key=prompt_key,
                    node_name="Portfolio Manager",
                    artifact_field="final_trade_decision",
                    source_node_name="Portfolio Manager",
                    ran=True,
                    skipped=False,
                    passed=passed,
                    checks=(CheckResult(scope="pm_block_present", passed=passed),),
                ),
            ),
        ),
    )
    execution = PromptCheckScenarioExecution(
        report=report,
        outputs={
            prompt_key: PromptCheckNodeOutput(
                prompt_key=prompt_key,
                node_name="Portfolio Manager",
                artifact_field="final_trade_decision",
                artifact_text=artifact_text,
                ran=True,
            )
        },
    )
    return PromptCheckSuiteExecution(
        suite_report=PromptCheckSuiteReport(
            suite="smoke",
            description="test",
            passed=passed,
            scenario_reports=(report,),
        ),
        scenario_executions=(execution,),
    )


def test_parse_response_accepts_valid_json():
    parsed = SemanticJudge._parse_response(
        json.dumps({"verdict": "PASS", "score": 0.93, "signals": []})
    )
    assert parsed["verdict"] == "PASS"
    assert parsed["score"] == pytest.approx(0.93)


def test_parse_response_strips_json_fences():
    parsed = SemanticJudge._parse_response(
        "```json\n"
        + json.dumps({"verdict": "SOFT_FAIL", "score": 0.7, "signals": ["thin"]})
        + "\n```"
    )
    assert parsed["verdict"] == "SOFT_FAIL"
    assert parsed["signals"] == ("thin",)


def test_parse_response_malformed_json_hard_fails():
    parsed = SemanticJudge._parse_response("not-json")
    assert parsed["verdict"] == "HARD_FAIL"
    assert parsed["signals"]


def test_semantic_judge_uses_quick_model_factory_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    sentinel = object()
    calls: list[tuple[str | None, int | None]] = []

    def fake_create_quick_thinking_llm(*, model=None, max_output_tokens=None):
        calls.append((model, max_output_tokens))
        return sentinel

    monkeypatch.setattr(
        "src.eval.semantic_judge.create_quick_thinking_llm",
        fake_create_quick_thinking_llm,
    )

    judge = SemanticJudge(model="gemini-2.0-flash")
    assert judge._llm is sentinel
    assert calls == [("gemini-2.0-flash", 2048)]


def test_semantic_judge_uses_consultant_factory_for_explicit_openai_override(
    monkeypatch: pytest.MonkeyPatch,
):
    sentinel = object()
    calls: list[tuple[str | None, bool, int | None]] = []

    def fake_create_consultant_llm(
        *,
        model=None,
        quick_mode=False,
        max_completion_tokens=None,
    ):
        calls.append((model, quick_mode, max_completion_tokens))
        return sentinel

    monkeypatch.setattr(
        "src.eval.semantic_judge.create_consultant_llm",
        fake_create_consultant_llm,
    )

    judge = SemanticJudge(model="gpt-4o-mini")
    assert judge._llm is sentinel
    assert calls == [("gpt-4o-mini", True, 2048)]


@pytest.mark.asyncio
async def test_run_stage3_suite_fails_when_baseline_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.eval.semantic_judge.SemanticJudge._create_llm",
        lambda self, model: object(),
    )
    monkeypatch.setattr(
        "src.eval.semantic_judge.iter_stage3_judge_specs",
        lambda: [get_node_capture_spec("Portfolio Manager")],
    )

    report = await run_stage3_suite(_stage2_execution(), allow_missing_baseline=False)
    scenario = report.scenario_reports[0]
    assert report.passed is False
    assert scenario.passed is False
    assert "Missing accepted baseline" in (scenario.error or "")


@pytest.mark.asyncio
async def test_run_stage3_suite_can_skip_missing_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.eval.semantic_judge.SemanticJudge._create_llm",
        lambda self, model: object(),
    )
    monkeypatch.setattr(
        "src.eval.semantic_judge.iter_stage3_judge_specs",
        lambda: [get_node_capture_spec("Portfolio Manager")],
    )

    report = await run_stage3_suite(_stage2_execution(), allow_missing_baseline=True)
    scenario = report.scenario_reports[0]
    assert report.passed is True
    assert scenario.passed is True
    assert "Missing accepted baseline" in (scenario.warning or "")


@pytest.mark.asyncio
async def test_run_stage3_suite_ignores_mismatched_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    accepted_root = (
        tmp_path / "evals" / "captures" / CURRENT_CAPTURE_SCHEMA_VERSION / "accepted"
    )
    _write_accepted_capture(
        accepted_root,
        ticker="7203.T",
        quick=False,
        strict=False,
    )
    monkeypatch.setattr(
        "src.eval.semantic_judge.SemanticJudge._create_llm",
        lambda self, model: object(),
    )
    monkeypatch.setattr(
        "src.eval.semantic_judge.iter_stage3_judge_specs",
        lambda: [get_node_capture_spec("Portfolio Manager")],
    )

    report = await run_stage3_suite(_stage2_execution(), allow_missing_baseline=False)
    assert "Missing accepted baseline" in (report.scenario_reports[0].error or "")


@pytest.mark.asyncio
async def test_run_stage3_suite_warns_on_stale_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    accepted_root = (
        tmp_path / "evals" / "captures" / CURRENT_CAPTURE_SCHEMA_VERSION / "accepted"
    )
    _write_accepted_capture(
        accepted_root,
        ticker="7203.T",
        quick=True,
        strict=False,
        trade_date="2026-02-01",
    )
    monkeypatch.setattr(
        "src.eval.semantic_judge.SemanticJudge._create_llm",
        lambda self, model: object(),
    )
    monkeypatch.setattr(
        "src.eval.semantic_judge.iter_stage3_judge_specs",
        lambda: [get_node_capture_spec("Portfolio Manager")],
    )

    async def fake_judge(self, **kwargs):
        return JudgeResult(
            prompt_key=kwargs["prompt_key"],
            rubric_family=kwargs["rubric_family"],
            verdict="PASS",
            score=1.0,
            signals=(),
            baseline_run_id=kwargs["baseline_run_id"],
            raw_response="{}",
        )

    monkeypatch.setattr(
        "src.eval.semantic_judge.SemanticJudge.judge_artifact", fake_judge
    )

    report = await run_stage3_suite(_stage2_execution(), baseline_warn_age_days=30)
    scenario = report.scenario_reports[0]
    assert scenario.passed is True
    assert "Accepted baseline is" in (scenario.warning or "")


@pytest.mark.asyncio
async def test_run_stage3_suite_hard_fails_when_stage2_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    accepted_root = (
        tmp_path / "evals" / "captures" / CURRENT_CAPTURE_SCHEMA_VERSION / "accepted"
    )
    _write_accepted_capture(
        accepted_root,
        ticker="7203.T",
        quick=True,
        strict=False,
    )
    monkeypatch.setattr(
        "src.eval.semantic_judge.SemanticJudge._create_llm",
        lambda self, model: object(),
    )
    monkeypatch.setattr(
        "src.eval.semantic_judge.iter_stage3_judge_specs",
        lambda: [get_node_capture_spec("Portfolio Manager")],
    )

    report = await run_stage3_suite(_stage2_execution(passed=False))
    result = report.scenario_reports[0].judge_results[0]
    assert result.verdict == "HARD_FAIL"
    assert "stage2 structural checks failed" in result.signals


@pytest.mark.asyncio
async def test_run_stage3_suite_hard_fails_when_baseline_artifact_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    accepted_root = (
        tmp_path / "evals" / "captures" / CURRENT_CAPTURE_SCHEMA_VERSION / "accepted"
    )
    _write_accepted_capture(
        accepted_root,
        ticker="7203.T",
        quick=True,
        strict=False,
        prompt_key="fundamentals_analyst",
        node_name="Fundamentals Analyst",
        artifact_field="fundamentals_report",
    )
    monkeypatch.setattr(
        "src.eval.semantic_judge.SemanticJudge._create_llm",
        lambda self, model: object(),
    )
    monkeypatch.setattr(
        "src.eval.semantic_judge.iter_stage3_judge_specs",
        lambda: [get_node_capture_spec("Portfolio Manager")],
    )

    report = await run_stage3_suite(_stage2_execution())
    result = report.scenario_reports[0].judge_results[0]
    assert result.verdict == "HARD_FAIL"
    assert "baseline_artifact_missing" in result.signals


def test_iter_stage3_specs_use_expected_rubric_families():
    families = {
        get_node_capture_spec("Fundamentals Analyst").rubric_family,
        get_node_capture_spec("Legal Counsel").rubric_family,
        get_node_capture_spec("Value Trap Detector").rubric_family,
        get_node_capture_spec("Valuation Calculator").rubric_family,
        get_node_capture_spec("Trader").rubric_family,
        get_node_capture_spec("Portfolio Manager").rubric_family,
        get_node_capture_spec("PM Fast-Fail").rubric_family,
    }
    assert families == {
        "analysis_report",
        "legal_structured",
        "risk_structured",
        "valuation_structured",
        "trade_decision",
        "portfolio_decision",
    }


def test_iter_stage3_specs_do_not_repeat_prompt_keys():
    prompt_keys = [spec.prompt_key for spec in iter_stage3_judge_specs()]
    assert len(prompt_keys) == len(set(prompt_keys))
