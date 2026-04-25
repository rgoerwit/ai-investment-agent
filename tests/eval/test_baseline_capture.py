import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.runtime import invoke_with_rate_limit_handling
from src.eval import (
    CURRENT_CAPTURE_SCHEMA_VERSION,
    NODE_CAPTURE_SPECS,
    BaselineCaptureConfig,
    BaselineCaptureManager,
    get_active_capture_manager,
    get_node_capture_spec,
    reset_active_capture_manager,
    set_active_capture_manager,
    validate_capture_bundle,
)
from src.eval.execution_profile import summarize_agent_llm_profile
from src.eval.prompt_provenance import compute_prompt_set_digest
from src.runtime_diagnostics import failure_artifact, success_artifact
from src.tooling.runtime import ToolExecutionService, ToolInvocation


class StaticRunnable:
    def __init__(self, response):
        self._response = response

    async def ainvoke(self, input_data):
        return self._response


class FlakyRunnable:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, input_data):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("temporary timeout")
        return AIMessage(content="consultant response")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _artifact_status(ok: bool) -> dict:
    return {
        "complete": True,
        "ok": ok,
        "content": "x",
        "error_kind": None if ok else "application_error",
        "provider": "google",
        "message": None,
        "retryable": False,
    }


def _clean_git_metadata(cwd=None) -> dict:
    return {
        "git_branch": "feature/test",
        "git_commit": "abc123",
        "dirty": False,
    }


def _dirty_git_metadata(cwd=None) -> dict:
    return {
        "git_branch": "feature/test",
        "git_commit": "abc123",
        "dirty": True,
    }


def _make_args(**overrides):
    base = {
        "ticker": "0005.HK",
        "quick": True,
        "strict": False,
        "no_memory": True,
        "capture_baseline": True,
        "retrospective_only": False,
        "output": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _make_manager(
    tmp_path: Path, monkeypatch, *, dirty: bool = False
) -> BaselineCaptureManager:
    monkeypatch.setattr(
        "src.eval.baseline_capture.get_git_metadata",
        _dirty_git_metadata if dirty else _clean_git_metadata,
    )
    manager = BaselineCaptureManager(
        BaselineCaptureConfig(
            enabled=True,
            schema_version=CURRENT_CAPTURE_SCHEMA_VERSION,
            output_root=tmp_path / "captures",
        )
    )
    manager.start_run(
        ticker="0005.HK",
        trade_date="2026-03-21",
        args=_make_args(),
        session_id="session-123",
    )
    return manager


async def _run_wrapped(
    manager: BaselineCaptureManager, node_name: str, node, *, state=None, config=None
):
    wrapped = manager.wrap_node(node_name, node)
    token = set_active_capture_manager(manager)
    try:
        return await wrapped(
            state or {"company_of_interest": "0005.HK"},
            config
            or {
                "configurable": {
                    "context": {
                        "ticker": "0005.HK",
                        "trade_date": "2026-03-21",
                        "quick_mode": True,
                        "enable_memory": False,
                    }
                }
            },
        )
    finally:
        reset_active_capture_manager(token)


async def _create_accepted_rm_capture(tmp_path: Path, monkeypatch) -> Path:
    manager = _make_manager(tmp_path, monkeypatch)
    llm = StaticRunnable(AIMessage(content="investment plan body"))

    async def fake_rm_node(state, config):
        await invoke_with_rate_limit_handling(
            llm,
            [HumanMessage(content="Provide Investment Plan.")],
            context="Research Manager",
            provider="google",
            model_name="test-model",
        )
        return success_artifact(
            "investment_plan",
            "Investment plan body",
            provider="google",
        )

    await _run_wrapped(manager, "Research Manager", fake_rm_node)
    run_dir = manager.finalize_run(
        {
            "analysis_validity": {"publishable": True},
            "artifact_statuses": {"investment_plan": _artifact_status(True)},
            "run_summary": {
                "quick_model": "gemini-2.5-flash",
                "deep_model": "gemini-2.5-pro",
                "llm_provider": "google",
                "llm_providers_used": ["google"],
            },
        }
    )
    assert run_dir is not None
    return run_dir


def test_node_capture_contract_covers_live_prompt_nodes():
    expected_eligible = {
        "Market Analyst",
        "Sentiment Analyst",
        "News Analyst",
        "Junior Fundamentals Analyst",
        "Foreign Language Analyst",
        "Legal Counsel",
        "Value Trap Detector",
        "Auditor",
        "Fundamentals Analyst",
        "Bull Researcher R1",
        "Bull Researcher R2",
        "Bear Researcher R1",
        "Bear Researcher R2",
        "Research Manager",
        "Valuation Calculator",
        "Consultant",
        "Trader",
        "Risky Analyst",
        "Safe Analyst",
        "Neutral Analyst",
        "Portfolio Manager",
        "PM Fast-Fail",
    }
    actual_eligible = {
        node_name
        for node_name, spec in NODE_CAPTURE_SPECS.items()
        if spec.baseline_eligible
    }
    assert actual_eligible == expected_eligible
    assert (
        get_node_capture_spec("Fundamentals Analyst").prompt_key
        == "fundamentals_analyst"
    )

    for node_name in (
        "Dispatcher",
        "Sync Check",
        "Fundamentals Sync Check",
        "Debate Sync R1",
        "Debate Sync Final",
        "Financial Validator",
        "Chart Generator",
        "State Cleaner",
    ):
        assert get_node_capture_spec(node_name).baseline_eligible is False


def test_baseline_eligible_specs_define_evaluator_scope():
    for spec in NODE_CAPTURE_SPECS.values():
        if not spec.baseline_eligible:
            continue
        assert spec.evaluator_scope

    assert (
        "pm_verdict_present"
        not in get_node_capture_spec("Fundamentals Analyst").evaluator_scope
    )
    assert (
        "pm_verdict_present"
        in get_node_capture_spec("Portfolio Manager").evaluator_scope
    )


def test_stage3_rubric_family_annotations_are_valid():
    allowed = {
        None,
        "analysis_report",
        "legal_structured",
        "risk_structured",
        "valuation_structured",
        "trade_decision",
        "portfolio_decision",
    }
    for spec in NODE_CAPTURE_SPECS.values():
        assert spec.rubric_family in allowed
        if spec.rubric_family is None:
            continue
        assert spec.baseline_eligible is True
        assert spec.capture_role == "agent"

    assert (
        get_node_capture_spec("Portfolio Manager").rubric_family == "portfolio_decision"
    )
    assert get_node_capture_spec("PM Fast-Fail").rubric_family == "portfolio_decision"


def test_prompt_set_digest_changes_when_prompt_digest_changes():
    digest_one = compute_prompt_set_digest(
        {
            "market_analyst": {"digest": "sha256:a"},
            "research_manager": {"digest": "sha256:b"},
        }
    )
    digest_two = compute_prompt_set_digest(
        {
            "market_analyst": {"digest": "sha256:a"},
            "research_manager": {"digest": "sha256:c"},
        }
    )

    assert digest_one != digest_two


def test_prompt_set_digest_is_stable_when_non_prompt_fields_change():
    digest_one = compute_prompt_set_digest(
        {
            "research_manager": {
                "digest": "sha256:b",
                "system_message": "one",
                "version": "v1",
            }
        }
    )
    digest_two = compute_prompt_set_digest(
        {
            "research_manager": {
                "digest": "sha256:b",
                "system_message": "two",
                "version": "v2",
            }
        }
    )

    assert digest_one == digest_two


@pytest.mark.asyncio
async def test_baseline_capture_accepts_clean_run_and_validator_passes(
    tmp_path, monkeypatch
):
    run_dir = await _create_accepted_rm_capture(tmp_path, monkeypatch)

    manifest = _read_json(run_dir / "run_manifest.json")
    assert manifest["capture_status"] == "accepted"
    assert manifest["storage_tier"] == "accepted"
    assert manifest["usable_for_replay"] is True
    assert manifest["schema_version"] == CURRENT_CAPTURE_SCHEMA_VERSION
    assert manifest["prompt_set_digest"].startswith("sha256:")
    assert "research_manager" in manifest["prompts"]["used"]
    assert manifest["models"]["defaults"]["quick_model"] == "gemini-2.5-flash"
    assert manifest["models"]["defaults"]["deep_model"] == "gemini-2.5-pro"
    assert manifest["models"]["observed_models"] == ["test-model"]
    assert manifest["models"]["observed_providers"] == ["google"]
    assert "/accepted/2026-03-21/" in str(run_dir)

    agent_dir = run_dir / "agents" / "research_manager"
    metadata = _read_json(agent_dir / "metadata.json")
    config_snapshot = _read_json(agent_dir / "config_snapshot.json")
    output = _read_json(agent_dir / "output.json")

    assert metadata["prompt"]["digest"].startswith("sha256:")
    assert metadata["prompt"]["prompt_key"] == "research_manager"
    assert metadata["artifact_fields"] == ["investment_plan"]
    assert metadata["usable_for_replay"] is True
    assert metadata["evaluator_scope"] == ["artifact_complete"]
    assert metadata["llm_call_sequence_ids"] == [1]
    assert metadata["effective_models"] == ["test-model"]
    assert metadata["effective_providers"] == ["google"]
    assert metadata["primary_model"] == "test-model"
    assert metadata["primary_provider"] == "google"
    assert metadata["primary_status"] == "success"
    assert metadata["last_attempt_model"] == "test-model"
    assert metadata["last_attempt_status"] == "success"
    assert metadata["llm_call_count"] == 1
    assert config_snapshot["configurable"]["context"]["ticker"] == "0005.HK"
    assert output["investment_plan"] == "Investment plan body"

    llm_rows = (run_dir / "llm_calls.jsonl").read_text().strip().splitlines()
    assert len(llm_rows) == 1
    llm_row = json.loads(llm_rows[0])
    assert llm_row["node_name"] == "Research Manager"
    assert llm_row["reasoning_level"] is None
    assert llm_row["input_tokens"] is None
    assert llm_row["output_tokens"] is None

    report = validate_capture_bundle(run_dir)
    assert report.passed is True


@pytest.mark.asyncio
async def test_llm_retry_and_tool_events_are_linked_to_node(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    runnable = FlakyRunnable()
    hook = manager.make_tool_hook()
    service = ToolExecutionService(hooks=[hook])

    async def fake_consultant_node(state, config):
        await invoke_with_rate_limit_handling(
            runnable,
            [HumanMessage(content="consultant prompt")],
            context="Consultant",
            provider="openai",
            model_name="test-model",
        )
        await service.execute(
            ToolInvocation(
                name="spot_check_metric_alt",
                args={"ticker": "0005.HK"},
                source="consultant",
                agent_key="consultant",
            ),
            runner=lambda args: _async_return({"checked": args["ticker"]}),
        )
        return success_artifact(
            "consultant_review",
            "Consultant review",
            provider="openai",
        )

    async def _no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr("src.agents.runtime.asyncio.sleep", _no_sleep)
    await _run_wrapped(manager, "Consultant", fake_consultant_node)

    run_dir = manager.finalize_run(
        {
            "analysis_validity": {"publishable": True},
            "artifact_statuses": {"consultant_review": _artifact_status(True)},
            "run_summary": {
                "quick_model": "gemini-2.5-flash",
                "deep_model": "gemini-2.5-pro",
                "llm_provider": "openai",
                "llm_providers_used": ["openai"],
            },
        }
    )

    assert run_dir is not None
    metadata = _read_json(run_dir / "agents" / "consultant" / "metadata.json")
    assert metadata["llm_call_sequence_ids"] == [1, 2]
    assert metadata["tool_call_sequence_ids"] == [1, 2]
    assert metadata["effective_models"] == ["test-model"]
    assert metadata["primary_model"] == "test-model"
    assert metadata["primary_status"] == "success"
    assert metadata["last_attempt_model"] == "test-model"
    assert metadata["last_attempt_status"] == "success"

    llm_rows = [
        json.loads(line)
        for line in (run_dir / "llm_calls.jsonl").read_text().strip().splitlines()
    ]
    assert [row["status"] for row in llm_rows] == ["failure", "success"]
    assert all(row["node_name"] == "Consultant" for row in llm_rows)


@pytest.mark.asyncio
async def test_llm_call_rows_include_reasoning_and_token_usage_when_available(
    tmp_path, monkeypatch
):
    manager = _make_manager(tmp_path, monkeypatch)
    response = AIMessage(
        content="investment plan body",
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 45,
            "total_tokens": 165,
        },
    )
    llm = StaticRunnable(response)
    llm.thinking_level = "low"

    async def fake_rm_node(state, config):
        await invoke_with_rate_limit_handling(
            llm,
            [HumanMessage(content="Provide Investment Plan.")],
            context="Research Manager",
            provider="google",
            model_name="gemini-3-flash-preview",
        )
        return success_artifact(
            "investment_plan",
            "Investment plan body",
            provider="google",
        )

    await _run_wrapped(manager, "Research Manager", fake_rm_node)
    run_dir = manager.finalize_run(
        {
            "analysis_validity": {"publishable": True},
            "artifact_statuses": {"investment_plan": _artifact_status(True)},
            "run_summary": {
                "quick_model": "gemini-3-flash-preview",
                "deep_model": "gemini-3-pro-preview",
                "llm_provider": "google",
                "llm_providers_used": ["google"],
            },
        }
    )

    assert run_dir is not None
    llm_rows = [
        json.loads(line)
        for line in (run_dir / "llm_calls.jsonl").read_text().strip().splitlines()
    ]
    assert llm_rows[0]["reasoning_level"] == "low"
    assert llm_rows[0]["thinking_config_raw"]["name"] == "thinking_level"
    assert llm_rows[0]["input_tokens"] == 120
    assert llm_rows[0]["output_tokens"] == 45
    assert llm_rows[0]["total_tokens"] == 165


def test_summarize_agent_llm_profile_sets_primary_to_none_when_all_calls_fail():
    profile = summarize_agent_llm_profile(
        [1, 2],
        {
            1: {
                "sequence_id": 1,
                "status": "failure",
                "provider": "openai",
                "model": "gpt-5-mini",
                "reasoning_level": "medium",
            },
            2: {
                "sequence_id": 2,
                "status": "failure",
                "provider": "openai",
                "model": "gpt-5-mini",
                "reasoning_level": "medium",
            },
        },
    )

    assert profile["effective_models"] == ["gpt-5-mini"]
    assert profile["primary_model"] is None
    assert profile["primary_status"] is None
    assert profile["last_attempt_model"] == "gpt-5-mini"
    assert profile["last_attempt_status"] == "failure"


@pytest.mark.asyncio
async def test_validate_capture_bundle_fails_when_config_snapshot_missing(
    tmp_path, monkeypatch
):
    run_dir = await _create_accepted_rm_capture(tmp_path, monkeypatch)
    (run_dir / "agents" / "research_manager" / "config_snapshot.json").unlink()

    report = validate_capture_bundle(run_dir)
    assert report.passed is False
    assert any(
        "missing_file:config_snapshot.json" in reason
        for agent in report.agent_reports
        for reason in agent.reasons
    )


@pytest.mark.asyncio
async def test_validate_capture_bundle_fails_when_llm_link_is_broken(
    tmp_path, monkeypatch
):
    run_dir = await _create_accepted_rm_capture(tmp_path, monkeypatch)
    metadata_path = run_dir / "agents" / "research_manager" / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata["llm_call_sequence_ids"] = [999]
    _write_json(metadata_path, metadata)

    report = validate_capture_bundle(run_dir)
    assert report.passed is False
    assert any(
        "missing_llm_event:999" in reason
        for agent in report.agent_reports
        for reason in agent.reasons
    )


@pytest.mark.asyncio
async def test_validate_capture_bundle_fails_when_prompt_digest_mismatches(
    tmp_path, monkeypatch
):
    run_dir = await _create_accepted_rm_capture(tmp_path, monkeypatch)
    metadata_path = run_dir / "agents" / "research_manager" / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata["prompt"]["digest"] = "sha256:not-real"
    _write_json(metadata_path, metadata)

    report = validate_capture_bundle(run_dir)
    assert report.passed is False
    assert any(
        "prompt_digest_mismatch" in reason
        for agent in report.agent_reports
        for reason in agent.reasons
    )


def test_reset_active_capture_manager_restores_previous_manager(tmp_path, monkeypatch):
    manager_one = _make_manager(tmp_path / "one", monkeypatch)
    manager_two = _make_manager(tmp_path / "two", monkeypatch)

    token_one = set_active_capture_manager(manager_one)
    assert get_active_capture_manager() is manager_one

    token_two = set_active_capture_manager(manager_two)
    assert get_active_capture_manager() is manager_two

    reset_active_capture_manager(token_two)
    assert get_active_capture_manager() is manager_one

    reset_active_capture_manager(token_one)
    assert get_active_capture_manager() is None


@pytest.mark.asyncio
async def test_parallel_wrapped_nodes_keep_event_links_isolated(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch)
    service = ToolExecutionService(hooks=[manager.make_tool_hook()])
    llm = StaticRunnable(AIMessage(content="parallel response"))
    ready = asyncio.Event()
    release = asyncio.Event()

    async def market_node(state, config):
        await invoke_with_rate_limit_handling(
            llm,
            [HumanMessage(content="market prompt")],
            context="Market Analyst",
            provider="google",
            model_name="test-model",
        )
        ready.set()
        await release.wait()
        await service.execute(
            ToolInvocation(
                name="get_market_snapshot",
                args={"ticker": "0005.HK"},
                source="market",
                agent_key="market_analyst",
            ),
            runner=lambda args: _async_return({"market": args["ticker"]}),
        )
        return success_artifact("market_report", "Market report", provider="google")

    async def sentiment_node(state, config):
        await ready.wait()
        await invoke_with_rate_limit_handling(
            llm,
            [HumanMessage(content="sentiment prompt")],
            context="Sentiment Analyst",
            provider="google",
            model_name="test-model",
        )
        manager.record_memory_event(
            {"event": "sentiment_memory_lookup", "ticker": "0005.HK"}
        )
        release.set()
        return success_artifact(
            "sentiment_report", "Sentiment report", provider="google"
        )

    market_wrapped = manager.wrap_node("Market Analyst", market_node)
    sentiment_wrapped = manager.wrap_node("Sentiment Analyst", sentiment_node)
    config = {
        "configurable": {
            "context": {
                "ticker": "0005.HK",
                "trade_date": "2026-03-21",
                "quick_mode": True,
                "enable_memory": False,
            }
        }
    }
    token = set_active_capture_manager(manager)
    try:
        await asyncio.gather(
            asyncio.create_task(
                market_wrapped({"company_of_interest": "0005.HK"}, config)
            ),
            asyncio.create_task(
                sentiment_wrapped({"company_of_interest": "0005.HK"}, config)
            ),
        )
    finally:
        reset_active_capture_manager(token)

    run_dir = manager.finalize_run(
        {
            "analysis_validity": {"publishable": True},
            "artifact_statuses": {
                "market_report": _artifact_status(True),
                "sentiment_report": _artifact_status(True),
            },
            "run_summary": {
                "quick_model": "gemini-2.5-flash",
                "deep_model": "gemini-2.5-pro",
                "llm_provider": "google",
                "llm_providers_used": ["google"],
            },
        }
    )
    assert run_dir is not None

    market_meta = _read_json(run_dir / "agents" / "market_analyst" / "metadata.json")
    sentiment_meta = _read_json(
        run_dir / "agents" / "sentiment_analyst" / "metadata.json"
    )

    assert market_meta["llm_call_sequence_ids"] == [1]
    assert market_meta["tool_call_sequence_ids"] == [1, 2]
    assert market_meta["memory_event_sequence_ids"] == []

    assert sentiment_meta["llm_call_sequence_ids"] == [2]
    assert sentiment_meta["tool_call_sequence_ids"] == []
    assert sentiment_meta["memory_event_sequence_ids"] == [1]

    llm_rows = [
        json.loads(line)
        for line in (run_dir / "llm_calls.jsonl").read_text().strip().splitlines()
    ]
    assert llm_rows[0]["node_name"] == "Market Analyst"
    assert llm_rows[1]["node_name"] == "Sentiment Analyst"


@pytest.mark.asyncio
async def test_capture_rejected_after_invalidating_error_and_downstream_bundles_skipped(
    tmp_path, monkeypatch
):
    manager = _make_manager(tmp_path, monkeypatch)

    async def failing_consultant(state, config):
        return failure_artifact(
            "consultant_review",
            "tooling failure",
            provider="openai",
        )

    async def clean_rm_node(state, config):
        return success_artifact(
            "investment_plan",
            "Investment plan body",
            provider="google",
        )

    await _run_wrapped(manager, "Consultant", failing_consultant)
    await _run_wrapped(manager, "Research Manager", clean_rm_node)

    run_dir = manager.finalize_run(
        {
            "analysis_validity": {"publishable": False},
            "artifact_statuses": {
                "consultant_review": _artifact_status(False),
                "investment_plan": _artifact_status(True),
            },
            "run_summary": {},
        }
    )

    assert run_dir is not None
    rejected = _read_json(run_dir / "capture_rejected.json")
    assert rejected["capture_status"] == "rejected"
    assert rejected["storage_tier"] == "rejected"
    assert rejected["usable_for_replay"] is False
    assert "/rejected/2026-03-21/" in str(run_dir)
    assert not (run_dir / "agents").exists()

    node_rows = [
        json.loads(line)
        for line in (run_dir / "node_events.jsonl").read_text().strip().splitlines()
    ]
    rm_row = next(row for row in node_rows if row["node_name"] == "Research Manager")
    assert rm_row["eligible"] is False
    assert rm_row["rejected_before"] is True


@pytest.mark.asyncio
async def test_dirty_worktree_rejects_capture(tmp_path, monkeypatch):
    manager = _make_manager(tmp_path, monkeypatch, dirty=True)
    llm = StaticRunnable(AIMessage(content="investment plan body"))

    async def fake_rm_node(state, config):
        await invoke_with_rate_limit_handling(
            llm,
            [HumanMessage(content="Provide Investment Plan.")],
            context="Research Manager",
            provider="google",
            model_name="test-model",
        )
        return success_artifact(
            "investment_plan",
            "Investment plan body",
            provider="google",
        )

    await _run_wrapped(manager, "Research Manager", fake_rm_node)
    run_dir = manager.finalize_run(
        {
            "analysis_validity": {"publishable": True},
            "artifact_statuses": {"investment_plan": _artifact_status(True)},
            "run_summary": {
                "quick_model": "gemini-2.5-flash",
                "deep_model": "gemini-2.5-pro",
                "llm_provider": "google",
                "llm_providers_used": ["google"],
            },
        }
    )

    assert run_dir is not None
    rejected = _read_json(run_dir / "capture_rejected.json")
    assert "git_dirty" in rejected["rejection_reasons"]
    assert not (run_dir / "agents").exists()


def test_cleanup_stale_inflight_moves_capture_to_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.eval.baseline_capture.get_git_metadata", _clean_git_metadata
    )
    manager = BaselineCaptureManager(
        BaselineCaptureConfig(
            enabled=True,
            schema_version=CURRENT_CAPTURE_SCHEMA_VERSION,
            output_root=tmp_path / "captures",
        )
    )
    stale_dir = (
        tmp_path
        / "captures"
        / CURRENT_CAPTURE_SCHEMA_VERSION
        / "inflight"
        / "stale-run"
    )
    stale_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        stale_dir / "run_manifest.json",
        {
            "schema_version": CURRENT_CAPTURE_SCHEMA_VERSION,
            "run_id": "stale-run",
            "trade_date": "2026-03-20",
            "capture_status": "inflight",
            "rejection_reasons": [],
        },
    )

    summary = manager.cleanup_stale_inflight_runs()

    assert summary.moved_to_rejected == 1
    rejected_manifest = _read_json(
        tmp_path
        / "captures"
        / CURRENT_CAPTURE_SCHEMA_VERSION
        / "rejected"
        / "2026-03-20"
        / "stale-run"
        / "run_manifest.json"
    )
    assert rejected_manifest["capture_status"] == "rejected"
    assert "interrupted_run" in rejected_manifest["rejection_reasons"]


def test_capture_flag_is_parsed_from_cli():
    from src.cli import build_arg_parser

    parser = build_arg_parser()
    args = parser.parse_args(["--ticker", "0005.HK", "--capture-baseline"])
    assert args.capture_baseline is True


async def _async_return(value):
    return value
