from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pc = importlib.import_module("src.eval.prompt_checks")
sc = importlib.import_module("src.eval.scenario_catalog")


VALID_DATA_BLOCK = """### --- START DATA_BLOCK ---
SECTOR: Information Technology
RAW_HEALTH_SCORE: 80
ADJUSTED_HEALTH_SCORE: 82
RAW_GROWTH_SCORE: 70
ADJUSTED_GROWTH_SCORE: 71
US_REVENUE_PERCENT: 5
PE_RATIO_TTM: 15
ADR_EXISTS: NO
### --- END DATA_BLOCK ---"""

LEGACY_DATA_BLOCK = """### DATA_BLOCK
SECTOR: Information Technology
RAW_HEALTH_SCORE: 80
ADJUSTED_HEALTH_SCORE: 82
RAW_GROWTH_SCORE: 70
ADJUSTED_GROWTH_SCORE: 71
US_REVENUE_PERCENT: 5
PE_RATIO_TTM: 15
ADR_EXISTS: NO

### FINANCIAL HEALTH DETAIL
Healthy enough
"""

VALID_PM_BLOCK = """### --- START PM_BLOCK ---
VERDICT: BUY
ZONE: LOW
POSITION_SIZE: 2.5
### --- END PM_BLOCK ---"""

VALID_VALUE_TRAP = """### --- START VALUE_TRAP_BLOCK ---
SCORE: 35
VERDICT: CAUTIOUS
TRAP_RISK: MEDIUM
### --- END VALUE_TRAP_BLOCK ---"""

VALID_LEGAL_JSON = json.dumps(
    {"pfic_status": "CLEAN", "vie_structure": "NO", "country": "JP", "sector": "IT"}
)

VALID_VALUATION_PARAMS = """### --- START VALUATION_PARAMS ---
METHOD: P/E_NORMALIZATION
SECTOR: Information Technology
SECTOR_MEDIAN_PE: 25
CURRENT_PE: 15
PEG_RATIO: N/A
GROWTH_SCORE_PCT: 65
CURRENT_PRICE: 150
CONFIDENCE: HIGH
### --- END VALUATION_PARAMS ---"""

VALID_TRADE_BLOCK = """TRADE_BLOCK:
ACTION: BUY
SIZE: 2.5%
ENTRY: 150.00
STOP: 135.00
"""

VALID_RAW_WRAPPER = """=== RAW FINANCIAL DATA FOR 0005.HK ===

### TOOL 1: get_financial_metrics
{"current_price": 87.4}

### TOOL 2: get_fundamental_analysis
No ADR found

=== END RAW DATA ===
"""

VALID_CONSULTANT = """### CONSULTANT REVIEW: APPROVED

### FINAL CONSULTANT VERDICT

Overall Assessment: APPROVED
"""


def test_data_block_present_accepts_fenced_output():
    passed, reason = pc.check_data_block_present(VALID_DATA_BLOCK)
    assert passed is True
    assert reason is None


def test_data_block_present_repairs_legacy_output():
    passed, reason = pc.check_data_block_present(LEGACY_DATA_BLOCK)
    assert passed is True
    assert reason is None


def test_data_block_present_rejects_unrecoverable_output():
    passed, reason = pc.check_data_block_present(
        "DATA_BLOCK:\nnot a real structured block"
    )
    assert passed is False
    assert reason == "DATA_BLOCK unparseable after normalization"


def test_pm_block_present_requires_real_pm_block():
    passed, _ = pc.check_pm_block_present(VALID_PM_BLOCK)
    assert passed is True

    passed, reason = pc.check_pm_block_present("PORTFOLIO MANAGER VERDICT: BUY")
    assert passed is False
    assert reason == "PM_BLOCK missing or unparseable"


def test_pm_verdict_present_allows_prose_fallback():
    passed, _ = pc.check_pm_verdict_present(VALID_PM_BLOCK)
    assert passed is True

    passed, reason = pc.check_pm_verdict_present("PORTFOLIO MANAGER VERDICT: BUY")
    assert passed is True
    assert reason is None


def test_value_trap_checks_require_structured_metrics():
    passed, _ = pc.check_value_trap_block_present(VALID_VALUE_TRAP)
    assert passed is True
    passed, _ = pc.check_value_trap_score_parseable(VALID_VALUE_TRAP)
    assert passed is True

    passed, reason = pc.check_value_trap_score_parseable("VERDICT: CAUTIOUS")
    assert passed is False
    assert reason == "VALUE_TRAP score missing or unparseable"


def test_legal_json_requires_pfic_and_vie_fields():
    passed, _ = pc.check_legal_json_valid(VALID_LEGAL_JSON)
    assert passed is True

    passed, reason = pc.check_legal_json_valid('{"pfic_status":"CLEAN"}')
    assert passed is False
    assert reason == "legal JSON missing pfic_status or vie_structure"


def test_valuation_params_present_requires_parseable_block():
    passed, _ = pc.check_valuation_params_present(VALID_VALUATION_PARAMS)
    assert passed is True

    passed, reason = pc.check_valuation_params_present("METHOD: P/E_NORMALIZATION")
    assert passed is False
    assert reason == "VALUATION_PARAMS block missing or unparseable"


def test_trade_block_present_requires_parseable_action():
    passed, _ = pc.check_trade_block_present(VALID_TRADE_BLOCK)
    assert passed is True

    passed, reason = pc.check_trade_block_present("ACTION: MAYBE")
    assert passed is False
    assert reason == "TRADE_BLOCK missing or ACTION not parseable"


def test_raw_data_wrapper_complete_requires_both_sections_and_end_marker():
    passed, _ = pc.check_raw_data_wrapper_complete(VALID_RAW_WRAPPER)
    assert passed is True

    passed, reason = pc.check_raw_data_wrapper_complete(
        VALID_RAW_WRAPPER.replace("=== END RAW DATA ===", "")
    )
    assert passed is False
    assert reason == "raw fundamentals wrapper incomplete"


def test_consultant_verdict_present_requires_expected_structure():
    passed, _ = pc.check_consultant_verdict_present(VALID_CONSULTANT)
    assert passed is True

    passed, reason = pc.check_consultant_verdict_present("CONSULTANT REVIEW only")
    assert passed is False
    assert reason == "consultant verdict structure missing"


def test_run_prompt_checks_on_outputs_skips_optional_consultant():
    outputs = {
        "fundamentals_analyst": pc.PromptCheckNodeOutput(
            prompt_key="fundamentals_analyst",
            node_name="Fundamentals Analyst",
            artifact_field="fundamentals_report",
            artifact_text=VALID_DATA_BLOCK,
            ran=True,
        ),
        "value_trap_detector": pc.PromptCheckNodeOutput(
            prompt_key="value_trap_detector",
            node_name="Value Trap Detector",
            artifact_field="value_trap_report",
            artifact_text=VALID_VALUE_TRAP,
            ran=True,
        ),
        "portfolio_manager": pc.PromptCheckNodeOutput(
            prompt_key="portfolio_manager",
            node_name="Portfolio Manager",
            artifact_field="final_trade_decision",
            artifact_text=VALID_PM_BLOCK,
            ran=True,
        ),
        "legal_counsel": pc.PromptCheckNodeOutput(
            prompt_key="legal_counsel",
            node_name="Legal Counsel",
            artifact_field="legal_report",
            artifact_text=VALID_LEGAL_JSON,
            ran=True,
        ),
        "valuation_calculator": pc.PromptCheckNodeOutput(
            prompt_key="valuation_calculator",
            node_name="Valuation Calculator",
            artifact_field="valuation_params",
            artifact_text=VALID_VALUATION_PARAMS,
            ran=True,
        ),
        "trader": pc.PromptCheckNodeOutput(
            prompt_key="trader",
            node_name="Trader",
            artifact_field="trader_investment_plan",
            artifact_text=VALID_TRADE_BLOCK,
            ran=True,
        ),
        "junior_fundamentals_analyst": pc.PromptCheckNodeOutput(
            prompt_key="junior_fundamentals_analyst",
            node_name="Junior Fundamentals Analyst",
            artifact_field="raw_fundamentals_data",
            artifact_text=VALID_RAW_WRAPPER,
            ran=True,
        ),
    }

    report = pc.run_prompt_checks_on_outputs(outputs)
    assert report.passed is True
    consultant_report = next(
        item for item in report.node_reports if item.prompt_key == "consultant"
    )
    assert consultant_report.skipped is True
    assert consultant_report.passed is True


@pytest.mark.asyncio
async def test_prompt_check_collector_records_wrapped_node_output():
    collector = pc.PromptCheckCollector()

    async def fake_node(state, config):
        return {"fundamentals_report": VALID_DATA_BLOCK, "artifact_statuses": {}}

    wrapped = collector.wrap_node("Fundamentals Analyst", fake_node)
    result = await wrapped({}, {})

    assert result["fundamentals_report"] == VALID_DATA_BLOCK
    outputs = collector.collected_outputs()
    assert outputs["fundamentals_analyst"].node_name == "Fundamentals Analyst"
    assert outputs["fundamentals_analyst"].artifact_text == VALID_DATA_BLOCK


@pytest.mark.asyncio
async def test_run_prompt_check_suite_from_manifest_uses_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path = tmp_path / "smoke.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite": "smoke",
                "description": "test suite",
                "scenarios": [{"ticker": "0005.HK", "quick": True}],
            }
        ),
        encoding="utf-8",
    )

    async def fake_run_suite(
        suite: sc.PromptCheckSuite,
    ) -> pc.PromptCheckSuiteExecution:
        report = pc.PromptCheckScenarioReport(
            ticker=suite.scenarios[0].ticker,
            quick=suite.scenarios[0].quick,
            strict=suite.scenarios[0].strict,
            passed=True,
            run_report=pc.RunCheckReport(passed=True, node_reports=()),
        )
        execution = pc.PromptCheckScenarioExecution(report=report, outputs={})
        return pc.PromptCheckSuiteExecution(
            suite_report=pc.PromptCheckSuiteReport(
                suite=suite.name,
                description=suite.description,
                passed=True,
                scenario_reports=(report,),
            ),
            scenario_executions=(execution,),
        )

    monkeypatch.setattr(
        "src.eval.prompt_checks.validate_environment_variables", lambda: None
    )
    monkeypatch.setattr("src.eval.prompt_checks.run_prompt_check_suite", fake_run_suite)

    report = await pc.run_prompt_check_suite_from_manifest(manifest_path)
    assert report.suite == "smoke"
    assert report.passed is True
    assert len(report.scenario_reports) == 1
    assert report.scenario_reports[0].ticker == "0005.HK"


def test_load_suite_manifest_rejects_empty_scenarios(tmp_path: Path):
    manifest_path = tmp_path / "broken.json"
    manifest_path.write_text(
        json.dumps({"suite": "broken", "scenarios": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="has no scenarios"):
        pc._load_suite_manifest(manifest_path)


def test_build_arg_parser_defaults_to_smoke_suite():
    args = pc.build_arg_parser().parse_args([])
    assert args.suite is None
    assert args.stage3 is False
    assert args.allow_missing_baseline is False
    assert args.baseline_warn_age_days == 30


@pytest.mark.asyncio
async def test_run_prompt_check_suite_returns_execution_objects(
    monkeypatch: pytest.MonkeyPatch,
):
    suite = sc.PromptCheckSuite(
        name="smoke",
        description="test suite",
        scenarios=(sc.PromptCheckScenario(ticker="AAPL", quick=True),),
    )

    async def fake_run_scenario(
        scenario: sc.PromptCheckScenario,
    ) -> pc.PromptCheckScenarioExecution:
        report = pc.PromptCheckScenarioReport(
            ticker=scenario.ticker,
            quick=scenario.quick,
            strict=scenario.strict,
            passed=True,
            run_report=pc.RunCheckReport(passed=True, node_reports=()),
        )
        outputs = {
            "portfolio_manager": pc.PromptCheckNodeOutput(
                prompt_key="portfolio_manager",
                node_name="Portfolio Manager",
                artifact_field="final_trade_decision",
                artifact_text=VALID_PM_BLOCK,
                ran=True,
            )
        }
        return pc.PromptCheckScenarioExecution(report=report, outputs=outputs)

    monkeypatch.setattr(
        "src.eval.prompt_checks.validate_environment_variables", lambda: None
    )
    monkeypatch.setattr(
        "src.eval.prompt_checks._run_prompt_check_scenario", fake_run_scenario
    )

    execution = await pc.run_prompt_check_suite(suite)
    assert execution.suite_report.passed is True
    assert execution.scenario_executions[0].report.ticker == "AAPL"
    assert "portfolio_manager" in execution.scenario_executions[0].outputs
