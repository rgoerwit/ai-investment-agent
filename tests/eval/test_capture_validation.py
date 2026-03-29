import json
from pathlib import Path

from src.eval.capture_validation import validate_capture_bundle
from src.eval.constants import CURRENT_CAPTURE_SCHEMA_VERSION
from src.eval.prompt_digest import prompt_digest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _build_capture(
    tmp_path: Path,
    *,
    node_name: str = "Research Manager",
    prompt: dict | None = None,
    metadata_overrides: dict | None = None,
    config_snapshot: dict | None = None,
    input_state: dict | None = None,
    input_sources: dict | None = None,
    output: dict | None = None,
    node_events: list[dict] | None = None,
    llm_rows: list[dict] | None = None,
    tool_rows: list[dict] | None = None,
    memory_rows: list[dict] | None = None,
    run_manifest: dict | None = None,
    include_agent_bundle: bool = True,
) -> Path:
    run_dir = tmp_path / "capture"
    prompt_payload = {
        "agent_key": "research_manager",
        "agent_name": "Research Manager",
        "version": "v1",
        "system_message": "hello",
        "category": "research",
        "requires_tools": False,
    }
    prompt = prompt or {
        **prompt_payload,
        "digest": prompt_digest(prompt_payload),
    }
    manifest_defaults = {
        "schema_version": CURRENT_CAPTURE_SCHEMA_VERSION,
        "capture_status": "accepted",
    }
    if CURRENT_CAPTURE_SCHEMA_VERSION == "schema_v4":
        manifest_defaults.update(
            {
                "prompt_set_digest": "sha256:test-prompt-set",
                "models": {
                    "defaults": {
                        "quick_model": "gemini-3-flash-preview",
                        "deep_model": "gemini-3-pro-preview",
                    }
                },
            }
        )
    _write_json(
        run_dir / "run_manifest.json",
        {**manifest_defaults, **(run_manifest or {})},
    )

    agent_dir = run_dir / "agents" / "research_manager"
    if include_agent_bundle:
        metadata_defaults = {
            "node_name": node_name,
            "prompt": prompt,
            "llm_call_sequence_ids": [1],
            "tool_call_sequence_ids": [],
            "memory_event_sequence_ids": [],
        }
        if CURRENT_CAPTURE_SCHEMA_VERSION == "schema_v4":
            metadata_defaults.update(
                {
                    "evaluator_scope": ["artifact_complete"],
                    "effective_models": ["test-model"],
                    "effective_providers": ["google"],
                    "effective_reasoning_levels": [],
                    "primary_model": "test-model",
                    "primary_provider": "google",
                    "primary_reasoning_level": None,
                    "primary_status": "success",
                    "last_attempt_model": "test-model",
                    "last_attempt_provider": "google",
                    "last_attempt_reasoning_level": None,
                    "last_attempt_status": "success",
                    "llm_call_count": 1,
                }
            )
        _write_json(
            agent_dir / "metadata.json",
            {**metadata_defaults, **(metadata_overrides or {})},
        )
        _write_json(
            agent_dir / "config_snapshot.json",
            config_snapshot or {"configurable": {"context": {"ticker": "0005.HK"}}},
        )
        _write_json(
            agent_dir / "input_state.json",
            input_state or {"company_of_interest": "0005.HK"},
        )
        _write_json(
            agent_dir / "input_sources.json",
            input_sources
            or {"company_of_interest": {"source_type": "runtime_context"}},
        )
        _write_json(
            agent_dir / "output.json",
            output or {"investment_plan": "body"},
        )

    (run_dir / "node_events.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                node_events
                or [
                    {
                        "node_name": node_name,
                        "eligible": True,
                        "baseline_eligible": True,
                    }
                ]
            )
        )
    )
    (run_dir / "llm_calls.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                llm_rows
                or [
                    {
                        "sequence_id": 1,
                        "status": "success",
                        "provider": "google",
                        "model": "test-model",
                        "input": [{"content": "hello"}],
                        "response": {"content": "world"},
                    }
                ]
            )
        )
    )
    (run_dir / "tool_calls.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (tool_rows or []))
    )
    (run_dir / "memory_events.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (memory_rows or []))
    )
    _write_json(
        run_dir / "run_output.json", {"analysis_validity": {"publishable": True}}
    )
    return run_dir


def test_validate_capture_bundle_fails_on_prompt_digest_mismatch(tmp_path):
    payload = {
        "agent_key": "research_manager",
        "agent_name": "Research Manager",
        "version": "v1",
        "system_message": "hello",
        "category": "research",
        "requires_tools": False,
    }
    run_dir = _build_capture(
        tmp_path,
        prompt={**payload, "digest": "sha256:not-real"},
    )

    report = validate_capture_bundle(run_dir)

    assert report.passed is False
    assert report.agent_reports[0].reasons == ("prompt_digest_mismatch",)
    assert prompt_digest(payload).startswith("sha256:")


def test_validate_capture_bundle_fails_on_unsupported_schema_version(tmp_path):
    run_dir = _build_capture(
        tmp_path,
        run_manifest={"schema_version": "schema_v2"},
    )

    report = validate_capture_bundle(run_dir)

    assert report.passed is False
    assert report.reasons == ("unsupported_schema_version:schema_v2",)


def test_validate_capture_bundle_fails_when_expected_output_field_missing(tmp_path):
    run_dir = _build_capture(tmp_path, output={"unexpected_field": "body"})

    report = validate_capture_bundle(run_dir)

    assert report.passed is False
    assert report.agent_reports[0].reasons == ("missing_expected_output_field",)


def test_schema_v4_capture_requires_prompt_set_digest(tmp_path):
    run_dir = _build_capture(tmp_path, run_manifest={"prompt_set_digest": ""})

    report = validate_capture_bundle(run_dir)

    assert report.passed is False
    assert "missing_prompt_set_digest" in report.reasons


def test_schema_v4_capture_requires_default_models(tmp_path):
    run_dir = _build_capture(
        tmp_path,
        run_manifest={"models": {"defaults": {"quick_model": "", "deep_model": ""}}},
    )

    report = validate_capture_bundle(run_dir)

    assert report.passed is False
    assert "missing_default_quick_model" in report.reasons
    assert "missing_default_deep_model" in report.reasons


def test_schema_v4_capture_requires_agent_execution_summary_for_llm_nodes(tmp_path):
    run_dir = _build_capture(
        tmp_path,
        metadata_overrides={
            "effective_models": None,
            "primary_model": None,
        },
    )

    report = validate_capture_bundle(run_dir)

    assert report.passed is False
    assert "missing_effective_models" in report.agent_reports[0].reasons


def test_schema_v4_capture_allows_null_reasoning_level(tmp_path):
    run_dir = _build_capture(
        tmp_path,
        metadata_overrides={
            "primary_reasoning_level": None,
            "last_attempt_reasoning_level": None,
        },
    )

    report = validate_capture_bundle(run_dir)

    assert report.passed is True


def test_validate_capture_bundle_fails_on_serialization_fallbacks(tmp_path):
    fallback = {
        "__serialization_fallback__": True,
        "type": "OpaqueValue",
        "repr": "<x>",
    }
    run_dir = _build_capture(
        tmp_path,
        config_snapshot={
            "configurable": {"context": {"ticker": "0005.HK"}},
            "bad": fallback,
        },
        input_state={"company_of_interest": "0005.HK", "bad": fallback},
        output={"investment_plan": "body", "bad": fallback},
        llm_rows=[{"sequence_id": 1, "input": fallback, "response": fallback}],
    )

    report = validate_capture_bundle(run_dir)

    assert report.passed is False
    assert report.reasons == (
        "llm_input_contains_serialization_fallback:1",
        "llm_response_contains_serialization_fallback:1",
    )
    assert set(report.agent_reports[0].reasons) == {
        "config_snapshot_contains_serialization_fallback",
        "input_state_contains_serialization_fallback",
        "output_contains_serialization_fallback",
    }


def test_validate_capture_bundle_fails_on_broken_tool_and_memory_links(tmp_path):
    run_dir = _build_capture(
        tmp_path,
        metadata_overrides={
            "tool_call_sequence_ids": [7],
            "memory_event_sequence_ids": [9],
        },
    )

    report = validate_capture_bundle(run_dir)

    assert report.passed is False
    assert set(report.agent_reports[0].reasons) == {
        "missing_tool_event:7",
        "missing_memory_event:9",
    }


def test_validate_capture_bundle_fails_when_expected_llm_calls_are_missing(tmp_path):
    run_dir = _build_capture(tmp_path, metadata_overrides={"llm_call_sequence_ids": []})

    report = validate_capture_bundle(run_dir)

    assert report.passed is False
    assert report.agent_reports[0].reasons == ("missing_llm_calls",)


def test_validate_capture_bundle_fails_when_eligible_agent_bundle_is_missing(tmp_path):
    run_dir = _build_capture(tmp_path, include_agent_bundle=False)
    (run_dir / "agents").mkdir(parents=True, exist_ok=True)

    report = validate_capture_bundle(run_dir)

    assert report.passed is False
    assert report.reasons == ("missing_agent_bundle:Research Manager",)


def test_validate_agent_bundle_passes_for_non_eligible_node(tmp_path):
    prompt = {
        "agent_key": None,
        "agent_name": "Dispatcher",
        "version": None,
        "system_message": "",
        "category": None,
        "requires_tools": False,
        "digest": "sha256:ignored",
    }
    run_dir = _build_capture(
        tmp_path,
        node_name="Dispatcher",
        prompt=prompt,
        metadata_overrides={"llm_call_sequence_ids": []},
        output={},
        node_events=[
            {
                "node_name": "Dispatcher",
                "eligible": False,
                "baseline_eligible": False,
            }
        ],
    )

    report = validate_capture_bundle(run_dir)

    assert report.passed is True
    assert report.agent_reports[0].passed is True
