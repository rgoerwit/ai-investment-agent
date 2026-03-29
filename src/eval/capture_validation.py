from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .capture_contract import get_node_capture_spec
from .constants import CURRENT_CAPTURE_SCHEMA_VERSION
from .prompt_digest import prompt_digest
from .serialization import contains_serialization_fallback


@dataclass(frozen=True)
class AgentValidationReport:
    node_name: str
    passed: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CaptureValidationReport:
    passed: bool
    reasons: tuple[str, ...]
    agent_reports: tuple[AgentValidationReport, ...]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _index_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    index: dict[int, dict[str, Any]] = {}
    for row in rows:
        sequence_id = row.get("sequence_id")
        if isinstance(sequence_id, int):
            index[sequence_id] = row
    return index


def _output_satisfies_fields(
    output: dict[str, Any], expected_fields: tuple[str, ...]
) -> bool:
    return any(field in output for field in expected_fields)


def _failed_before_llm(
    output: dict[str, Any], expected_fields: tuple[str, ...]
) -> bool:
    artifact_statuses = output.get("artifact_statuses")
    if not isinstance(artifact_statuses, dict):
        return False
    for field in expected_fields:
        status = artifact_statuses.get(field)
        if (
            isinstance(status, dict)
            and status.get("complete")
            and not status.get("ok", False)
        ):
            return True
    return False


def validate_agent_bundle(
    agent_dir: Path,
    run_manifest: dict[str, Any],
    event_indexes: dict[str, dict[int, dict[str, Any]]],
) -> AgentValidationReport:
    metadata = _read_json(agent_dir / "metadata.json")
    node_name = metadata.get("node_name", agent_dir.name)
    spec = get_node_capture_spec(node_name)
    reasons: list[str] = []

    if not spec.baseline_eligible:
        return AgentValidationReport(node_name=node_name, passed=True, reasons=())

    for filename in (
        "metadata.json",
        "config_snapshot.json",
        "input_state.json",
        "input_sources.json",
        "output.json",
    ):
        if not (agent_dir / filename).exists():
            reasons.append(f"missing_file:{filename}")

    if reasons:
        return AgentValidationReport(
            node_name=node_name, passed=False, reasons=tuple(reasons)
        )

    prompt = metadata.get("prompt")
    if not isinstance(prompt, dict):
        reasons.append("missing_prompt")
    else:
        digest = prompt.get("digest")
        prompt_payload = {
            key: prompt.get(key)
            for key in (
                "agent_key",
                "agent_name",
                "version",
                "system_message",
                "category",
                "requires_tools",
            )
        }
        if not digest:
            reasons.append("missing_prompt_digest")
        elif digest != prompt_digest(prompt_payload):
            reasons.append("prompt_digest_mismatch")

    config_snapshot = _read_json(agent_dir / "config_snapshot.json")
    configurable = config_snapshot.get("configurable")
    if not isinstance(configurable, dict):
        reasons.append("missing_configurable")
    elif not isinstance(configurable.get("context"), dict):
        reasons.append("missing_context")
    if contains_serialization_fallback(config_snapshot):
        reasons.append("config_snapshot_contains_serialization_fallback")

    input_state = _read_json(agent_dir / "input_state.json")
    if contains_serialization_fallback(input_state):
        reasons.append("input_state_contains_serialization_fallback")

    output = _read_json(agent_dir / "output.json")
    if not _output_satisfies_fields(output, spec.artifact_fields):
        reasons.append("missing_expected_output_field")
    if contains_serialization_fallback(output):
        reasons.append("output_contains_serialization_fallback")

    llm_ids = metadata.get("llm_call_sequence_ids", [])
    tool_ids = metadata.get("tool_call_sequence_ids", [])
    memory_ids = metadata.get("memory_event_sequence_ids", [])

    for llm_id in llm_ids:
        if llm_id not in event_indexes["llm"]:
            reasons.append(f"missing_llm_event:{llm_id}")
    for tool_id in tool_ids:
        if tool_id not in event_indexes["tool"]:
            reasons.append(f"missing_tool_event:{tool_id}")
    for memory_id in memory_ids:
        if memory_id not in event_indexes["memory"]:
            reasons.append(f"missing_memory_event:{memory_id}")

    if (
        spec.expects_llm_calls
        and not llm_ids
        and not _failed_before_llm(output, spec.artifact_fields)
    ):
        reasons.append("missing_llm_calls")

    if run_manifest.get("schema_version") == "schema_v4":
        evaluator_scope = metadata.get("evaluator_scope")
        if not isinstance(evaluator_scope, list):
            reasons.append("missing_evaluator_scope")

        if spec.expects_llm_calls:
            effective_models = metadata.get("effective_models")
            if not isinstance(effective_models, list):
                reasons.append("missing_effective_models")

            if "primary_model" not in metadata:
                reasons.append("missing_primary_model")

            if "primary_status" not in metadata:
                reasons.append("missing_primary_status")

            if "llm_call_count" not in metadata or not isinstance(
                metadata.get("llm_call_count"), int
            ):
                reasons.append("missing_llm_call_count")

            if llm_ids and "last_attempt_model" not in metadata:
                reasons.append("missing_last_attempt_model")

    return AgentValidationReport(
        node_name=node_name,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def validate_capture_bundle(run_dir: Path) -> CaptureValidationReport:
    manifest = _read_json(run_dir / "run_manifest.json")
    reasons: list[str] = []
    agent_reports: list[AgentValidationReport] = []

    schema_version = manifest.get("schema_version")
    if schema_version != CURRENT_CAPTURE_SCHEMA_VERSION:
        reasons.append(f"unsupported_schema_version:{schema_version}")
        return CaptureValidationReport(False, tuple(reasons), ())

    if manifest.get("capture_status") != "accepted":
        reasons.append("capture_not_accepted")
        return CaptureValidationReport(False, tuple(reasons), ())

    if schema_version == "schema_v4":
        prompt_set_digest = manifest.get("prompt_set_digest")
        if not isinstance(prompt_set_digest, str) or not prompt_set_digest.strip():
            reasons.append("missing_prompt_set_digest")

        defaults = (manifest.get("models") or {}).get("defaults") or {}
        if not isinstance(defaults.get("quick_model"), str) or not defaults.get(
            "quick_model"
        ):
            reasons.append("missing_default_quick_model")
        if not isinstance(defaults.get("deep_model"), str) or not defaults.get(
            "deep_model"
        ):
            reasons.append("missing_default_deep_model")

    agent_dirs_root = run_dir / "agents"
    if not agent_dirs_root.exists():
        reasons.append("missing_agents_dir")
        return CaptureValidationReport(False, tuple(reasons), ())

    node_events = _read_jsonl(run_dir / "node_events.jsonl")
    expected_nodes = {
        event["node_name"]
        for event in node_events
        if event.get("eligible") and event.get("baseline_eligible")
    }

    llm_rows = _read_jsonl(run_dir / "llm_calls.jsonl")
    tool_rows = _read_jsonl(run_dir / "tool_calls.jsonl")
    memory_rows = _read_jsonl(run_dir / "memory_events.jsonl")
    for row in llm_rows:
        sequence_id = row.get("sequence_id", "unknown")
        if contains_serialization_fallback(row.get("input")):
            reasons.append(f"llm_input_contains_serialization_fallback:{sequence_id}")
        if contains_serialization_fallback(row.get("response")):
            reasons.append(
                f"llm_response_contains_serialization_fallback:{sequence_id}"
            )
    event_indexes = {
        "llm": _index_rows(llm_rows),
        "tool": _index_rows(tool_rows),
        "memory": _index_rows(memory_rows),
    }

    present_nodes: set[str] = set()
    for agent_dir in sorted(
        path for path in agent_dirs_root.iterdir() if path.is_dir()
    ):
        report = validate_agent_bundle(agent_dir, manifest, event_indexes)
        agent_reports.append(report)
        present_nodes.add(report.node_name)

    for node_name in sorted(expected_nodes - present_nodes):
        reasons.append(f"missing_agent_bundle:{node_name}")

    if not (run_dir / "run_output.json").exists():
        reasons.append("missing_run_output")

    passed = not reasons and all(report.passed for report in agent_reports)
    return CaptureValidationReport(
        passed=passed,
        reasons=tuple(reasons),
        agent_reports=tuple(agent_reports),
    )
