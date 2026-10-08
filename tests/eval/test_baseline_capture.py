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


async def _finalize_with_statuses(tmp_path, monkeypatch, statuses: dict) -> Path | None:
    """Run one clean node, then finalize with a caller-supplied status set."""
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
            "investment_plan", "Investment plan body", provider="google"
        )

    await _run_wrapped(manager, "Research Manager", fake_rm_node)
    return manager.finalize_run(
        {
            "analysis_validity": {"publishable": True},
            "artifact_statuses": {
                "investment_plan": _artifact_status(True),
                **statuses,
            },
            "run_summary": {
                "quick_model": "gemini-2.5-flash",
                "deep_model": "gemini-2.5-pro",
                "llm_provider": "google",
                "llm_providers_used": ["google"],
            },
        }
    )


@pytest.mark.asyncio
async def test_failing_optional_node_does_not_reject_the_capture(tmp_path, monkeypatch):
    """The live path: a *wrapped* optional node that actually fails.

    Node-level invalidation runs first and latches, so exempting optional
    artifacts only at finalization is dead code. Every 2026-08-03 capture
    recorded ``first_rejection_stage='Auditor'`` — not 'finalize'. This test
    wraps a genuinely failing Auditor rather than injecting a failed status at
    the end, which is what the finalize-only test could not catch.
    """
    manager = _make_manager(tmp_path, monkeypatch)

    async def failing_auditor_node(state, config):
        return failure_artifact(
            "auditor_report",
            "'llm:Global Forensic Accountant:openai:kimi-k3' exceeded hard timeout",
            provider="openai",
        )

    await _run_wrapped(manager, "Auditor", failing_auditor_node)
    assert manager.rejected is False, (
        "a failed optional seat must not reject the run at node level"
    )

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
            "investment_plan", "Investment plan body", provider="google"
        )

    await _run_wrapped(manager, "Research Manager", fake_rm_node)
    run_dir = manager.finalize_run(
        {
            "analysis_validity": {"publishable": True},
            "artifact_statuses": {
                "investment_plan": _artifact_status(True),
                "auditor_report": _artifact_status(False),
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
    manifest = _read_json(run_dir / "run_manifest.json")
    assert manifest["capture_status"] == "accepted"
    assert "/accepted/" in str(run_dir)


@pytest.mark.asyncio
async def test_optional_failure_text_does_not_reject_via_error_marker_scan(
    tmp_path, monkeypatch
):
    """Skipping artifact_failed is not enough on its own.

    The generic recursive scan still reaches the optional artifact's body and its
    status message, so a failure rendered with a "TOOL_ERROR:" prefix — which the
    auditor/consultant loops emit verbatim — re-latches the rejection through a
    different door.
    """
    manager = _make_manager(tmp_path, monkeypatch)

    async def failing_auditor_node(state, config):
        return failure_artifact(
            "auditor_report",
            "TOOL_ERROR: TimeoutError",
            provider="openai",
        )

    await _run_wrapped(manager, "Auditor", failing_auditor_node)
    assert manager.rejected is False


@pytest.mark.asyncio
async def test_unrelated_error_marker_still_rejects(tmp_path, monkeypatch):
    """The exemption is path-scoped, not a blanket disabling of the scan."""
    manager = _make_manager(tmp_path, monkeypatch)

    async def poisoned_rm_node(state, config):
        return {
            **success_artifact(
                "investment_plan", "[SYSTEM ERROR] upstream", provider="google"
            ),
            "investment_plan": "[SYSTEM ERROR] upstream",
        }

    await _run_wrapped(manager, "Research Manager", poisoned_rm_node)
    assert manager.rejected is True


@pytest.mark.asyncio
async def test_failed_optional_record_is_not_replay_eligible(tmp_path, monkeypatch):
    """Accepting the run must not promote a failed seat to reference material."""
    manager = _make_manager(tmp_path, monkeypatch)

    async def failing_auditor_node(state, config):
        return failure_artifact("auditor_report", "timed out", provider="openai")

    await _run_wrapped(manager, "Auditor", failing_auditor_node)

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
            "investment_plan", "Investment plan body", provider="google"
        )

    await _run_wrapped(manager, "Research Manager", fake_rm_node)
    run_dir = manager.finalize_run(
        {
            "analysis_validity": {"publishable": True},
            "artifact_statuses": {
                "investment_plan": _artifact_status(True),
                "auditor_report": _artifact_status(False),
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
    auditor_meta = _read_json(run_dir / "agents" / "auditor" / "metadata.json")
    rm_meta = _read_json(run_dir / "agents" / "research_manager" / "metadata.json")
    assert auditor_meta["usable_for_replay"] is False
    assert auditor_meta["replay_ineligible_reason"] == "artifact_failed"
    assert rm_meta["usable_for_replay"] is True


@pytest.mark.asyncio
async def test_failing_required_node_still_rejects_at_node_level(tmp_path, monkeypatch):
    """The exemption is scoped: a required seat failing still latches a rejection."""
    manager = _make_manager(tmp_path, monkeypatch)

    async def failing_fundamentals(state, config):
        return failure_artifact("fundamentals_report", "boom", provider="google")

    await _run_wrapped(manager, "Fundamentals Analyst", failing_fundamentals)
    assert manager.rejected is True


@pytest.mark.asyncio
async def test_node_exception_is_persisted_as_safe_structured_summary(
    tmp_path, monkeypatch
):
    manager = _make_manager(tmp_path, monkeypatch)

    async def raising_node(state, config):
        raise RuntimeError("request failed for https://user:secret@example.test/data")

    with pytest.raises(RuntimeError):
        await _run_wrapped(manager, "Fundamentals Analyst", raising_node)

    run_dir = manager.finalize_run(
        {
            "analysis_validity": {"publishable": False},
            "artifact_statuses": {},
            "run_summary": {},
        }
    )

    assert run_dir is not None
    rows = [
        json.loads(line)
        for line in (run_dir / "node_events.jsonl").read_text().splitlines()
    ]
    event = next(row for row in rows if row["node_name"] == "Fundamentals Analyst")
    assert isinstance(event["error"], dict)
    assert event["error"]["error_type"] == "RuntimeError"
    assert "user:secret" not in json.dumps(event)


@pytest.mark.asyncio
async def test_failed_optional_artifact_does_not_reject_the_capture(
    tmp_path, monkeypatch
):
    """An artifact the pipeline publishes without must not veto its capture.

    On 2026-08-03 all six smoke-suite tickers produced publishable analyses and
    all six captures were rejected for `artifact_failed:auditor_report` alone —
    the optional cross-check seat timing out made a clean corpus uncapturable.
    """
    run_dir = await _finalize_with_statuses(
        tmp_path,
        monkeypatch,
        {
            "auditor_report": _artifact_status(False),
            "consultant_review": _artifact_status(False),
        },
    )

    assert run_dir is not None
    manifest = _read_json(run_dir / "run_manifest.json")
    assert manifest["capture_status"] == "accepted"
    assert "/accepted/" in str(run_dir)


@pytest.mark.asyncio
async def test_failed_required_artifact_still_rejects_the_capture(
    tmp_path, monkeypatch
):
    """The relaxation is scoped: a required artifact's failure is still fatal."""
    run_dir = await _finalize_with_statuses(
        tmp_path, monkeypatch, {"fundamentals_report": _artifact_status(False)}
    )

    assert run_dir is not None
    manifest = _read_json(run_dir / "run_manifest.json")
    assert manifest["capture_status"] == "rejected"
    assert "artifact_failed:fundamentals_report" in manifest["rejection_reasons"]


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
    # Uses a *required* seat: an optional one (Consultant/Auditor) deliberately
    # no longer invalidates, since the pipeline publishes without it. The
    # downstream-skip machinery under test is unchanged.
    manager = _make_manager(tmp_path, monkeypatch)

    async def failing_fundamentals(state, config):
        return failure_artifact(
            "fundamentals_report",
            "tooling failure",
            provider="google",
        )

    async def clean_rm_node(state, config):
        return success_artifact(
            "investment_plan",
            "Investment plan body",
            provider="google",
        )

    await _run_wrapped(manager, "Fundamentals Analyst", failing_fundamentals)
    await _run_wrapped(manager, "Research Manager", clean_rm_node)

    run_dir = manager.finalize_run(
        {
            "analysis_validity": {"publishable": False},
            "artifact_statuses": {
                "fundamentals_report": _artifact_status(False),
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


async def _create_junior_cycle_capture(
    tmp_path, monkeypatch, *, omit_selector=False, failed=False, empty_failure=False
):
    from unittest.mock import AsyncMock, MagicMock

    from langchain_core.runnables import RunnableLambda

    from src.agents.analyst_nodes import create_analyst_node
    from src.eval.prompt_checks import (
        PromptCheckCollector,
        run_prompt_checks_on_outputs,
    )
    from src.eval.serialization import normalize_for_json
    from src.forensic_budget import graph_research_budget_policies
    from src.graph.tool_nodes import create_agent_tool_node
    from src.runtime_services import RuntimeServices, use_runtime_services
    from src.tooling.inspection_service import InspectionService

    manager = _make_manager(tmp_path, monkeypatch)
    # No local settings are needed to exercise the real capture seam.
    monkeypatch.setattr(
        manager,
        "_config_snapshot",
        lambda config: normalize_for_json({"configurable": config["configurable"]}),
    )
    calls = [
        {"name": name, "args": {"ticker": "0005.HK"}, "id": f"call-{index}"}
        for index, name in enumerate(
            ("get_financial_metrics", "get_fundamental_analysis")
        )
    ]
    response = AIMessage(content="", tool_calls=calls)

    class Selector(RunnableLambda):
        def bind_tools(self, tools):
            return self

    selector = Selector(lambda _: response)
    tools = []
    for name, value in (
        ("get_financial_metrics", '{"sector":"Industrials","operatingCashflow":123}'),
        ("get_fundamental_analysis", "" if failed else "inspected qualitative fact"),
    ):
        tool = MagicMock()
        tool.name = name
        tool.ainvoke = AsyncMock(return_value=value)
        tools.append(tool)
    junior = "junior_fundamentals_analyst"
    policy = graph_research_budget_policies(quick_mode=True)[junior]
    analyst = create_analyst_node(
        selector, junior, tools, "raw_fundamentals_data", research_budget_policy=policy
    )
    collector = PromptCheckCollector()
    analyst = collector.wrap_node("Junior Fundamentals Analyst", analyst)
    state = {"company_of_interest": "0005.HK", "messages": []}
    config = {
        "configurable": {
            "context": SimpleNamespace(ticker="0005.HK", trade_date="2026-03-21")
        }
    }
    with use_runtime_services(
        RuntimeServices(
            tool_service=ToolExecutionService([manager.make_tool_hook()]),
            inspection_service=InspectionService(),
        )
    ):
        selected = await _run_wrapped(
            manager, "Junior Fundamentals Analyst", analyst, state=state, config=config
        )
        acquired = await _run_wrapped(
            manager,
            "junior_fund_tools",
            create_agent_tool_node(tools, junior, budget_policy=policy),
            state={**state, **selected},
            config=config,
        )
        if empty_failure:
            from src.runtime_diagnostics import failure_artifact

            acquired.update(
                failure_artifact(
                    "raw_fundamentals_data",
                    "Code acquisition capped",
                    provider="code",
                    error_kind="output_cap_exhausted",
                )
            )
        terminal = await _run_wrapped(
            manager,
            "Junior Fundamentals Analyst",
            analyst,
            state={**state, **acquired},
            config=config,
        )
    assert terminal["raw_fundamentals_data"] == acquired["raw_fundamentals_data"]
    assert terminal["artifact_statuses"] == acquired["artifact_statuses"]
    outputs = collector.collected_outputs()
    assert outputs[junior].artifact_text == terminal["raw_fundamentals_data"]
    checks = run_prompt_checks_on_outputs(outputs)
    junior_checks = [
        check for check in checks.node_reports if check.prompt_key == junior
    ]
    # Structural wrapper completeness is distinct from acquisition health.
    assert junior_checks and all(check.passed for check in junior_checks) is (
        not empty_failure
    )
    assert outputs[junior].artifact_status["ok"] is (not (failed or empty_failure))
    if omit_selector:
        manager._llm_calls.clear()
    run_dir = manager.finalize_run(
        {
            **terminal,
            "analysis_validity": {"publishable": True},
            "run_summary": {"quick_model": "test-quick", "deep_model": "test-deep"},
        }
    )
    return manager, run_dir, state, terminal


@pytest.mark.asyncio
async def test_junior_selection_tools_code_completion_capture_and_collector(
    tmp_path, monkeypatch
):
    manager, run_dir, state, terminal = await _create_junior_cycle_capture(
        tmp_path, monkeypatch
    )
    assert manager.final_status == "accepted"
    assert validate_capture_bundle(run_dir).passed
    agent_dir = run_dir / "agents/junior_fundamentals_analyst"
    metadata = _read_json(agent_dir / "metadata.json")
    assert metadata["llm_call_count"] == 1
    assert len(metadata["llm_call_sequence_ids"]) == 1
    assert len(metadata["tool_call_sequence_ids"]) == 4
    assert metadata["artifact_producer"] == "code"
    assert _read_json(agent_dir / "input_state.json")["messages"] == []
    assert (
        _read_json(agent_dir / "output.json")["raw_fundamentals_data"]
        == terminal["raw_fundamentals_data"]
    )
    before = metadata.copy()
    manager._coalesce_junior_completion()
    assert manager._agent_records[0]["metadata"] == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    [
        "selector",
        "tool",
        "node",
        "foreign_selector",
        "mixed_selector_ids",
        "missing_cycle",
    ],
)
async def test_persisted_junior_cycle_rejects_corrupt_observed_links(
    tmp_path, monkeypatch, case
):
    _, run_dir, _, _ = await _create_junior_cycle_capture(tmp_path, monkeypatch)
    agent_dir = run_dir / "agents/junior_fundamentals_analyst"
    metadata = _read_json(agent_dir / "metadata.json")
    if case in {"selector", "tool", "node", "foreign_selector"}:
        path = (
            run_dir
            / {
                "selector": "llm_calls.jsonl",
                "tool": "tool_calls.jsonl",
                "node": "node_events.jsonl",
                "foreign_selector": "llm_calls.jsonl",
            }[case]
        )
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if case == "foreign_selector":
            rows[0]["node_name"] = "Research Manager"
        else:
            rows.pop(0)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        metadata["selector_node_sequence_ids"] = (
            [1, "bad"] if case == "mixed_selector_ids" else []
        )
        _write_json(agent_dir / "metadata.json", metadata)
    assert not validate_capture_bundle(run_dir).passed


@pytest.mark.asyncio
async def test_in_memory_junior_capture_cannot_replace_missing_selector(
    tmp_path, monkeypatch
):
    manager, _, _, _ = await _create_junior_cycle_capture(
        tmp_path, monkeypatch, omit_selector=True
    )
    assert manager.final_status == "rejected"
    assert any(
        "selector" in reason or "llm" in reason for reason in manager._rejection_reasons
    )


@pytest.mark.asyncio
async def test_failed_junior_completion_preserves_failure_and_capture_ineligibility(
    tmp_path, monkeypatch
):
    manager, _, _, terminal = await _create_junior_cycle_capture(
        tmp_path, monkeypatch, failed=True
    )
    assert not terminal["artifact_statuses"]["raw_fundamentals_data"]["ok"]
    assert manager.final_status == "rejected"


@pytest.mark.asyncio
async def test_empty_terminal_failure_closes_truthfully_and_cannot_be_captured_as_success(
    tmp_path, monkeypatch
):
    manager, run_dir, _, terminal = await _create_junior_cycle_capture(
        tmp_path, monkeypatch, empty_failure=True
    )
    assert terminal["raw_fundamentals_data"] == ""
    assert (
        terminal["artifact_statuses"]["raw_fundamentals_data"]["error_kind"]
        == "output_cap_exhausted"
    )
    assert not terminal["messages"][0].tool_calls
    assert manager.final_status == "rejected"
    validation = validate_capture_bundle(run_dir)
    assert not validation.passed
    rejected = _read_json(run_dir / "capture_rejected.json")
    assert "artifact_failed:raw_fundamentals_data" in rejected["rejection_reasons"]
    assert not (run_dir / "run_output.json").exists()
    assert not (run_dir / "agents/junior_fundamentals_analyst/output.json").exists()
    events = [
        json.loads(line)
        for line in (run_dir / "node_events.jsonl").read_text().splitlines()
    ]
    terminal_event = events[-1]
    assert terminal_event["node_name"] == "Junior Fundamentals Analyst"
    assert {"raw_fundamentals_data", "artifact_statuses", "messages", "sender"} <= set(
        terminal_event["output_keys"]
    )
    output = next(
        record["output"]
        for record in manager._agent_records
        if record["node_name"] == "Junior Fundamentals Analyst"
    )
    assert output["raw_fundamentals_data"] == ""
    assert output["artifact_statuses"] == terminal["artifact_statuses"]
