from __future__ import annotations

import contextvars
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import structlog

from src.runtime_diagnostics import get_optional_publishable_artifacts

from .capture_contract import NodeCaptureSpec, get_node_capture_spec
from .capture_validation import validate_capture_bundle
from .execution_profile import summarize_agent_llm_profile
from .git_meta import get_git_metadata
from .prompt_digest import prompt_digest
from .prompt_provenance import compute_prompt_set_digest
from .serialization import normalize_for_json

logger = structlog.get_logger(__name__)

_ACTIVE_CAPTURE_MANAGER: contextvars.ContextVar[BaselineCaptureManager | None] = (
    contextvars.ContextVar("baseline_capture_manager", default=None)
)
_ACTIVE_CAPTURE_NODE: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("baseline_capture_node", default=None)
)

_ERROR_MARKERS = (
    "[SYSTEM ERROR]",
    "[ERROR]:",
    "Error: Missing prompt",
    "TOOL_ERROR:",
)

_RUNTIME_CONTEXT_FIELDS = {
    "company_of_interest",
    "company_name",
    "company_name_resolved",
    "trade_date",
    "sender",
}
_UPSTREAM_ARTIFACT_FIELDS = {
    "market_report",
    "sentiment_report",
    "news_report",
    "raw_fundamentals_data",
    "foreign_language_report",
    "legal_report",
    "fundamentals_report",
    "auditor_report",
    "value_trap_report",
    "investment_debate_state",
    "investment_plan",
    "valuation_params",
    "consultant_review",
    "trader_investment_plan",
    "risk_debate_state",
    "final_trade_decision",
    "red_flags",
    "pre_screening_result",
    "chart_paths",
}


def get_active_capture_manager() -> BaselineCaptureManager | None:
    return _ACTIVE_CAPTURE_MANAGER.get()


def set_active_capture_manager(
    manager: BaselineCaptureManager | None,
) -> contextvars.Token:
    """Install the active capture manager and return a reset token.

    For cleanup, use `reset_active_capture_manager(token)`. Do not clear the
    context by calling `set_active_capture_manager(None)`.
    """
    return _ACTIVE_CAPTURE_MANAGER.set(manager)


def reset_active_capture_manager(token: contextvars.Token) -> None:
    """Restore the previous active capture manager using the token returned by set()."""
    _ACTIVE_CAPTURE_MANAGER.reset(token)


def get_active_capture_node() -> dict[str, Any] | None:
    return _ACTIVE_CAPTURE_NODE.get()


def set_active_capture_node(node_context: dict[str, Any] | None) -> contextvars.Token:
    return _ACTIVE_CAPTURE_NODE.set(node_context)


@dataclass(frozen=True)
class BaselineCaptureConfig:
    enabled: bool
    schema_version: str
    output_root: Path


@dataclass(frozen=True)
class CaptureCleanupSummary:
    scanned: int
    moved_to_rejected: int
    removed_empty: int
    rejected_paths: tuple[str, ...]


@dataclass(frozen=True)
class BaselinePreflightResult:
    git_clean: bool
    cleanup_summary: CaptureCleanupSummary | None = None


class BaselineCaptureToolHook:
    def __init__(self, manager: BaselineCaptureManager) -> None:
        self.manager = manager

    async def before(self, call):
        self.manager.record_tool_call(
            {
                "stage": "before",
                "name": call.name,
                "args": normalize_for_json(call.args),
                "source": call.source,
                "agent_key": call.agent_key,
            }
        )
        return call

    async def after(self, call, result):
        self.manager.record_tool_call(
            {
                "stage": "after",
                "name": call.name,
                "args": normalize_for_json(call.args),
                "source": call.source,
                "agent_key": call.agent_key,
                "blocked": result.blocked,
                "findings": normalize_for_json(result.findings or []),
                "value": normalize_for_json(result.value),
            }
        )
        return result


class BaselineCaptureManager:
    def __init__(self, config: BaselineCaptureConfig) -> None:
        self.config = config
        self._started = False
        self._rejected = False
        self._final_status: str | None = None
        self._first_rejection_reason: str | None = None
        self._first_rejection_stage: str | None = None
        self._rejection_reasons: list[str] = []
        self._node_events: list[dict[str, Any]] = []
        self._tool_calls: list[dict[str, Any]] = []
        self._memory_events: list[dict[str, Any]] = []
        self._llm_calls: list[dict[str, Any]] = []
        self._agent_records: list[dict[str, Any]] = []
        self._agent_records_by_sequence: dict[int, dict[str, Any]] = {}
        self._node_counter = 0
        self._tool_counter = 0
        self._memory_counter = 0
        self._llm_counter = 0
        self._run_dir: Path | None = None
        self._run_id: str | None = None
        self._run_manifest: dict[str, Any] = {}
        self._cleanup_summary: CaptureCleanupSummary | None = None
        self._preflight_git_clean: bool | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def rejected(self) -> bool:
        return self._rejected

    @property
    def final_status(self) -> str | None:
        return self._final_status

    @property
    def cleanup_summary(self) -> CaptureCleanupSummary | None:
        return self._cleanup_summary

    def apply_preflight_result(
        self,
        *,
        git_clean: bool,
        cleanup_summary: CaptureCleanupSummary | None = None,
    ) -> None:
        """Apply a pre-computed preflight result, typically from a batch runner."""
        self._preflight_git_clean = git_clean
        self._cleanup_summary = cleanup_summary

    def _schema_root(self) -> Path:
        return self.config.output_root / self.config.schema_version

    def _inflight_root(self) -> Path:
        return self._schema_root() / "inflight"

    def _accepted_root(self) -> Path:
        return self._schema_root() / "accepted"

    def _rejected_root(self) -> Path:
        return self._schema_root() / "rejected"

    def _now_iso(self) -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")

    def _load_json(self, path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as handle:
            return cast(dict[str, Any], json.load(handle))

    def _final_destination(self, *, status: str, trade_date: str) -> Path:
        root = self._accepted_root() if status == "accepted" else self._rejected_root()
        return root / trade_date / (self._run_id or "unknown_run")

    def _move_run_dir(self, destination: Path) -> None:
        current = self._run_path()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(current), str(destination))
        self._run_dir = destination

    def preflight_git_clean(self) -> tuple[bool, list[str]]:
        code_meta = get_git_metadata(Path.cwd())
        self._preflight_git_clean = not bool(code_meta.get("dirty"))
        if self._preflight_git_clean:
            return True, []
        return False, [
            "Baseline capture requires a clean local git worktree.",
            "Both modified tracked files and untracked files count as dirty.",
            "Run `git status --short`, then commit, stash, remove, or rerun without `--capture-baseline`.",
        ]

    def cleanup_stale_inflight_runs(self) -> CaptureCleanupSummary:
        inflight_root = self._inflight_root()
        inflight_root.mkdir(parents=True, exist_ok=True)
        moved: list[str] = []
        removed_empty = 0
        scanned = 0

        for run_dir in sorted(
            path for path in inflight_root.iterdir() if path.is_dir()
        ):
            scanned += 1
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.exists():
                if not any(run_dir.iterdir()):
                    shutil.rmtree(run_dir)
                    removed_empty += 1
                    continue
                manifest: dict[str, Any] = {
                    "schema_version": self.config.schema_version,
                    "run_id": run_dir.name,
                    "capture_status": "rejected",
                    "storage_tier": "rejected",
                    "usable_for_replay": False,
                    "usable_for_promotion": False,
                    "rejection_reasons": ["abandoned_inflight"],
                    "first_rejection_reason": "abandoned_inflight",
                    "first_rejection_stage": "cleanup",
                    "finalized_at_utc": self._now_iso(),
                }
                trade_date = "unknown_date"
            else:
                manifest = self._load_json(manifest_path)
                trade_date = manifest.get("trade_date", "unknown_date")
                reasons = list(manifest.get("rejection_reasons", []))
                if "interrupted_run" not in reasons:
                    reasons.append("interrupted_run")
                manifest["capture_status"] = "rejected"
                manifest["storage_tier"] = "rejected"
                manifest["usable_for_replay"] = False
                manifest["usable_for_promotion"] = False
                manifest["rejection_reasons"] = reasons
                manifest["first_rejection_reason"] = manifest.get(
                    "first_rejection_reason", "interrupted_run"
                )
                manifest["first_rejection_stage"] = manifest.get(
                    "first_rejection_stage", "cleanup"
                )
                manifest["finalized_at_utc"] = self._now_iso()

            destination = self._rejected_root() / trade_date / run_dir.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(run_dir), str(destination))
            with open(
                destination / "run_manifest.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    manifest, handle, indent=2, sort_keys=True, ensure_ascii=False
                )
            with open(
                destination / "capture_rejected.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    manifest, handle, indent=2, sort_keys=True, ensure_ascii=False
                )
            moved.append(str(destination))

        summary = CaptureCleanupSummary(
            scanned=scanned,
            moved_to_rejected=len(moved),
            removed_empty=removed_empty,
            rejected_paths=tuple(moved),
        )
        self._cleanup_summary = summary
        return summary

    def start_run(
        self,
        *,
        ticker: str,
        trade_date: str,
        args: Any,
        session_id: str | None,
    ) -> None:
        if not self.enabled or self._started:
            return
        timestamp = datetime.now(UTC)
        ts_slug = timestamp.strftime("%Y%m%dT%H%M%SZ")
        mode_bits = [
            "quick" if getattr(args, "quick", False) else "deep",
            "strict" if getattr(args, "strict", False) else "normal",
            "memoff" if getattr(args, "no_memory", False) else "memon",
        ]
        self._run_id = f"{ticker}_{'_'.join(mode_bits)}_{ts_slug}"
        self._inflight_root().mkdir(parents=True, exist_ok=True)
        self._accepted_root().mkdir(parents=True, exist_ok=True)
        self._rejected_root().mkdir(parents=True, exist_ok=True)
        self._run_dir = self._inflight_root() / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._started = True
        self._run_manifest = {
            "schema_version": self.config.schema_version,
            "capture_status": "inflight",
            "usable_for_replay": False,
            "usable_for_promotion": False,
            "storage_tier": "inflight",
            "capture_timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
            "finalized_at_utc": None,
            "run_id": self._run_id,
            "ticker": ticker,
            "trade_date": trade_date,
            "mode": {
                "quick": bool(getattr(args, "quick", False)),
                "strict": bool(getattr(args, "strict", False)),
                "memory_enabled": not bool(getattr(args, "no_memory", False)),
            },
            "runtime_args": {
                key: normalize_for_json(value)
                for key, value in vars(args).items()
                if not (key == "capture_baseline" and not value)
            },
            "code": get_git_metadata(Path.cwd()),
            "tracing": {"session_id": session_id},
            "rejection_reasons": [],
            "preflight": {
                "git_clean": self._preflight_git_clean,
                "cleanup_performed": bool(self._cleanup_summary),
                "cleanup_results": None
                if self._cleanup_summary is None
                else {
                    "scanned": self._cleanup_summary.scanned,
                    "moved_to_rejected": self._cleanup_summary.moved_to_rejected,
                    "removed_empty": self._cleanup_summary.removed_empty,
                    "rejected_paths": list(self._cleanup_summary.rejected_paths),
                },
            },
        }
        self._write_json(self._run_dir / "run_manifest.json", self._run_manifest)

    def _run_path(self) -> Path:
        if self._run_dir is None:
            raise RuntimeError("baseline capture run not started")
        return self._run_dir

    def make_tool_hook(self) -> BaselineCaptureToolHook:
        return BaselineCaptureToolHook(self)

    def _active_node_details(self) -> dict[str, Any]:
        active_node = get_active_capture_node() or {}
        return {
            "node_name": active_node.get("node_name"),
            "node_sequence_id": active_node.get("sequence_id"),
            "baseline_eligible": bool(active_node.get("baseline_eligible")),
            "eligible": not self.rejected and bool(active_node.get("eligible", True)),
        }

    def _attach_event_link(self, event_kind: str, sequence_id: int) -> None:
        active_node = get_active_capture_node() or {}
        node_sequence_id = active_node.get("sequence_id")
        if not isinstance(node_sequence_id, int):
            return
        record = self._agent_records_by_sequence.get(node_sequence_id)
        if not record:
            return
        metadata = record["metadata"]
        link_field = {
            "tool": "tool_call_sequence_ids",
            "memory": "memory_event_sequence_ids",
            "llm": "llm_call_sequence_ids",
        }[event_kind]
        ids = metadata.setdefault(link_field, [])
        ids.append(sequence_id)

    def record_memory_event(self, payload: dict[str, Any]) -> int | None:
        if not self.enabled or not self._started:
            return None
        self._memory_counter += 1
        node_details = self._active_node_details()
        event = {
            "sequence_id": self._memory_counter,
            **node_details,
            **normalize_for_json(payload),
        }
        self._memory_events.append(event)
        self._attach_event_link("memory", self._memory_counter)
        return self._memory_counter

    def record_tool_call(self, payload: dict[str, Any]) -> int | None:
        if not self.enabled or not self._started:
            return None
        self._tool_counter += 1
        node_details = self._active_node_details()
        event = {
            "sequence_id": self._tool_counter,
            **node_details,
            **normalize_for_json(payload),
        }
        self._tool_calls.append(event)
        self._attach_event_link("tool", self._tool_counter)
        return self._tool_counter

    def record_llm_call(self, payload: dict[str, Any]) -> int | None:
        if not self.enabled or not self._started:
            return None
        self._llm_counter += 1
        node_details = self._active_node_details()
        event = {
            "sequence_id": self._llm_counter,
            **node_details,
            **normalize_for_json(payload),
        }
        self._llm_calls.append(event)
        self._attach_event_link("llm", self._llm_counter)
        return self._llm_counter

    def _state_input_sources(self, state: dict[str, Any]) -> dict[str, Any]:
        sources: dict[str, Any] = {}
        for key in state:
            if key in _RUNTIME_CONTEXT_FIELDS:
                sources[key] = {"source_type": "runtime_context"}
            elif key in _UPSTREAM_ARTIFACT_FIELDS:
                sources[key] = {"source_type": "upstream_artifact"}
            elif key in {"prompts_used", "artifact_statuses", "tools_called"}:
                sources[key] = {"source_type": "derived_local"}
        return sources

    def _config_snapshot(self, config: Any) -> dict[str, Any]:
        configurable = {}
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
        return {"configurable": normalize_for_json(configurable)}

    def _prompt_info(self, spec: NodeCaptureSpec) -> dict[str, Any] | None:
        prompt_key = spec.prompt_key
        if not prompt_key:
            return None
        from src.prompts import get_prompt

        prompt = get_prompt(prompt_key)
        if not prompt:
            return {
                "prompt_key": prompt_key,
                "missing": True,
            }
        payload = {
            "agent_key": prompt.agent_key,
            "agent_name": prompt.agent_name,
            "version": prompt.version,
            "system_message": prompt.system_message,
            "category": prompt.category,
            "requires_tools": prompt.requires_tools,
        }
        return {
            "prompt_key": prompt_key,
            **payload,
            "digest": prompt_digest(payload),
        }

    def _optional_artifact_fields(self) -> frozenset[str]:
        """Optional-by-publication-contract fields, for this run's mode.

        The manager knows its own mode from the manifest, so node-level and
        finalize-level invalidation agree — they must, since node-level rejection
        latches and finalize can never revisit it.
        """
        quick = bool((self._run_manifest.get("mode") or {}).get("quick"))
        return get_optional_publishable_artifacts(
            {"run_summary": {"quick_mode": quick}}
        )

    def _detect_invalidating_reasons(self, result: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        artifact_statuses = result.get("artifact_statuses")
        # An artifact the pipeline publishes without must not veto the capture —
        # and this is the load-bearing site, not finalize_run(): a failing node
        # rejects here first and rejection latches, so exempting only at
        # finalization is dead code for every real optional-seat failure (the
        # 2026-08-03 captures all record first_rejection_stage='Auditor').
        optional_fields = self._optional_artifact_fields()
        if isinstance(artifact_statuses, dict):
            for field, status in artifact_statuses.items():
                if field in optional_fields:
                    continue
                if (
                    isinstance(status, dict)
                    and status.get("complete")
                    and not status.get("ok", False)
                ):
                    reasons.append(f"artifact_failed:{field}")

        prompts_used = result.get("prompts_used")
        if isinstance(prompts_used, dict):
            for key, value in prompts_used.items():
                if isinstance(value, dict) and not value.get("version"):
                    reasons.append(f"prompt_metadata_missing:{key}")

        # Skipping the artifact_failed reason above is not sufficient on its own:
        # this generic scan would still reach the same optional artifact's body and
        # its status message, so a failure rendered as "TOOL_ERROR: ..." (the
        # auditor/consultant loops emit exactly that prefix) re-latches the
        # rejection through a different door. Exclude only the known failed-optional
        # subtrees; unrelated error markers anywhere else still invalidate.
        statuses = artifact_statuses if isinstance(artifact_statuses, dict) else {}
        failed_optional = {
            field
            for field in optional_fields
            if isinstance(statuses.get(field), dict)
            and not statuses[field].get("ok", False)
        }
        exempt_paths = {f"result.{field}" for field in failed_optional} | {
            f"result.artifact_statuses.{field}" for field in failed_optional
        }

        def walk(value: Any, path: str) -> None:
            if path in exempt_paths:
                return
            if isinstance(value, str):
                if value.startswith(_ERROR_MARKERS):
                    reasons.append(f"error_marker:{path}")
            elif isinstance(value, dict):
                for child_key, child_value in value.items():
                    walk(child_value, f"{path}.{child_key}")
            elif isinstance(value, list):
                for index, child_value in enumerate(value):
                    walk(child_value, f"{path}[{index}]")

        walk(result, "result")
        return sorted(set(reasons))

    def reject_run(
        self,
        reasons: list[str],
        *,
        stage: str,
        partial: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or not self._started:
            return
        normalized = [str(reason) for reason in reasons if reason]
        if not normalized:
            return
        if not self._rejected:
            self._first_rejection_reason = normalized[0]
            self._first_rejection_stage = stage
        self._rejected = True
        for reason in normalized:
            if reason not in self._rejection_reasons:
                self._rejection_reasons.append(reason)
        logger.warning(
            "baseline_capture_rejected",
            stage=stage,
            first_reason=self._first_rejection_reason,
            reasons=self._rejection_reasons,
        )
        if partial:
            self._run_manifest["rejection_partial"] = normalize_for_json(partial)

    def wrap_node(self, node_name: str, node: Any) -> Any:
        if not self.enabled:
            return node

        spec = get_node_capture_spec(node_name)

        async def wrapped(state, config):
            rejected_before = self.rejected
            input_slice = None if rejected_before else normalize_for_json(dict(state))
            input_sources = (
                {} if rejected_before else self._state_input_sources(dict(state))
            )
            config_snapshot = {} if rejected_before else self._config_snapshot(config)
            prompt_info = self._prompt_info(spec)
            self._node_counter += 1
            sequence_id = self._node_counter

            pending_record: dict[str, Any] | None = None
            if spec.baseline_eligible and not rejected_before:
                pending_record = {
                    "node_name": node_name,
                    "metadata": {
                        "sequence_id": sequence_id,
                        "node_name": node_name,
                        "capture_role": spec.capture_role,
                        "baseline_eligible": spec.baseline_eligible,
                        "usable_for_replay": False,
                        "expects_llm_calls": spec.expects_llm_calls,
                        "artifact_fields": list(spec.artifact_fields),
                        "evaluator_scope": list(spec.evaluator_scope),
                        "prompt": normalize_for_json(prompt_info),
                        "llm_call_sequence_ids": [],
                        "tool_call_sequence_ids": [],
                        "memory_event_sequence_ids": [],
                    },
                    "config_snapshot": config_snapshot,
                    "input_state": input_slice,
                    "input_sources": normalize_for_json(input_sources),
                }
                self._agent_records_by_sequence[sequence_id] = pending_record

            node_token = set_active_capture_node(
                {
                    "sequence_id": sequence_id,
                    "node_name": node_name,
                    "baseline_eligible": spec.baseline_eligible,
                    "eligible": not rejected_before,
                }
            )
            try:
                result = await node(state, config)
            except Exception as exc:
                self._node_events.append(
                    {
                        "sequence_id": sequence_id,
                        "node_name": node_name,
                        "capture_role": spec.capture_role,
                        "baseline_eligible": spec.baseline_eligible,
                        "eligible": False,
                        "rejected_before": rejected_before,
                        "raised": True,
                        "error": str(exc),
                        "prompt": normalize_for_json(prompt_info),
                    }
                )
                self.reject_run([f"node_exception:{node_name}"], stage=node_name)
                raise
            finally:
                _ACTIVE_CAPTURE_NODE.reset(node_token)

            result_norm = normalize_for_json(result)
            reasons = self._detect_invalidating_reasons(
                result if isinstance(result, dict) else {}
            )
            if prompt_info and prompt_info.get("missing"):
                reasons.append(f"prompt_metadata_missing:{node_name}")
            if reasons:
                self.reject_run(reasons, stage=node_name, partial={"node": node_name})

            event = {
                "sequence_id": sequence_id,
                "node_name": node_name,
                "capture_role": spec.capture_role,
                "baseline_eligible": spec.baseline_eligible,
                "eligible": not self.rejected and not rejected_before,
                "rejected_before": rejected_before,
                "output_keys": sorted(result.keys())
                if isinstance(result, dict)
                else [],
                "artifact_fields": list(spec.artifact_fields),
                "invalidating_reasons": reasons,
                "prompt": normalize_for_json(prompt_info),
            }
            self._node_events.append(event)

            if pending_record is not None:
                pending_record["output"] = result_norm
                self._agent_records.append(pending_record)
            return result

        return wrapped

    def _collect_used_prompts(self) -> dict[str, Any]:
        prompts: dict[str, Any] = {}
        for event in self._node_events:
            prompt = event.get("prompt")
            if not isinstance(prompt, dict):
                continue
            prompt_key = prompt.get("prompt_key")
            if prompt_key:
                prompts[prompt_key] = prompt
        return prompts

    def _agent_dir_name(self, node_name: str) -> str:
        return node_name.lower().replace(" ", "_").replace("/", "_")

    def _write_agent_records(self) -> None:
        agents_dir = self._run_path() / "agents"
        agents_dir.mkdir(parents=True, exist_ok=True)
        for record in self._agent_records:
            metadata = record["metadata"]
            if not metadata.get("baseline_eligible"):
                continue
            # Accepting the *run* despite a failed optional seat must not promote
            # that seat's output to reference material. `valuation_params` feeds
            # the `valuation_structured` rubric and, in quick mode,
            # `value_trap_report` feeds `risk_structured` — so an empty or failed
            # optional output could otherwise become the baseline that future
            # good output is judged against.
            spec = get_node_capture_spec(record["node_name"])
            statuses = (record.get("output") or {}).get("artifact_statuses")
            failed = False
            if isinstance(statuses, dict):
                failed = any(
                    isinstance(statuses.get(field), dict)
                    and not statuses[field].get("ok", False)
                    for field in spec.artifact_fields
                )
            metadata["usable_for_replay"] = not failed
            if failed:
                metadata["replay_ineligible_reason"] = "artifact_failed"
            agent_dir = agents_dir / self._agent_dir_name(record["node_name"])
            agent_dir.mkdir(parents=True, exist_ok=True)
            self._write_json(agent_dir / "metadata.json", metadata)
            self._write_json(
                agent_dir / "config_snapshot.json", record["config_snapshot"]
            )
            self._write_json(agent_dir / "input_state.json", record["input_state"])
            self._write_json(agent_dir / "input_sources.json", record["input_sources"])
            self._write_json(agent_dir / "output.json", record["output"])

    def _write_json(self, path: Path, payload: Any) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    def _write_event_streams(self, run_dir: Path) -> None:
        self._write_jsonl(run_dir / "node_events.jsonl", self._node_events)
        self._write_jsonl(run_dir / "tool_calls.jsonl", self._tool_calls)
        self._write_jsonl(run_dir / "memory_events.jsonl", self._memory_events)
        self._write_jsonl(run_dir / "llm_calls.jsonl", self._llm_calls)

    def _build_llm_index(self) -> dict[int, dict[str, Any]]:
        return {
            row["sequence_id"]: row
            for row in self._llm_calls
            if isinstance(row.get("sequence_id"), int)
        }

    def _attach_agent_execution_profiles(self) -> None:
        llm_index = self._build_llm_index()
        for record in self._agent_records:
            metadata = record["metadata"]
            if not metadata.get("baseline_eligible"):
                continue
            metadata.update(
                summarize_agent_llm_profile(
                    metadata.get("llm_call_sequence_ids", []),
                    llm_index,
                )
            )

    def _observed_models(self) -> list[str]:
        models: list[str] = []
        for row in self._llm_calls:
            model = row.get("response_model") or row.get("model")
            if isinstance(model, str) and model and model not in models:
                models.append(model)
        return models

    def _observed_providers(self) -> list[str]:
        providers: list[str] = []
        for row in self._llm_calls:
            provider = row.get("provider")
            if isinstance(provider, str) and provider and provider not in providers:
                providers.append(provider)
        return providers

    def _validate_pending_capture(self) -> list[str]:
        reasons: list[str] = []
        code_meta = self._run_manifest.get("code", {})
        if code_meta.get("dirty"):
            reasons.append("git_dirty")

        for record in self._agent_records:
            metadata = record["metadata"]
            if not metadata.get("baseline_eligible"):
                continue
            node_name = metadata["node_name"]
            prompt = metadata.get("prompt")
            if not isinstance(prompt, dict):
                reasons.append(f"capture_missing_prompt:{node_name}")
            else:
                if not prompt.get("digest"):
                    reasons.append(f"capture_missing_prompt_digest:{node_name}")
            config_snapshot = record.get("config_snapshot", {})
            configurable = config_snapshot.get("configurable")
            if not isinstance(configurable, dict) or not isinstance(
                configurable.get("context"), dict
            ):
                reasons.append(f"capture_missing_context:{node_name}")
            if record.get("input_state") is None:
                reasons.append(f"capture_missing_input_state:{node_name}")
            if record.get("output") is None:
                reasons.append(f"capture_missing_output:{node_name}")

            spec = get_node_capture_spec(node_name)
            llm_ids = metadata.get("llm_call_sequence_ids", [])
            if spec.expects_llm_calls and not llm_ids:
                output = record.get("output", {})
                artifact_statuses = output.get("artifact_statuses", {})
                failure_without_llm = False
                if isinstance(artifact_statuses, dict):
                    for field in spec.artifact_fields:
                        status = artifact_statuses.get(field)
                        if (
                            isinstance(status, dict)
                            and status.get("complete")
                            and not status.get("ok", False)
                        ):
                            failure_without_llm = True
                            break
                if not failure_without_llm:
                    reasons.append(f"capture_missing_llm_calls:{node_name}")

        known_llm_ids = {row["sequence_id"] for row in self._llm_calls}
        known_tool_ids = {row["sequence_id"] for row in self._tool_calls}
        known_memory_ids = {row["sequence_id"] for row in self._memory_events}
        for record in self._agent_records:
            metadata = record["metadata"]
            if not metadata.get("baseline_eligible"):
                continue
            node_name = metadata["node_name"]
            for llm_id in metadata.get("llm_call_sequence_ids", []):
                if llm_id not in known_llm_ids:
                    reasons.append(f"capture_bad_llm_link:{node_name}:{llm_id}")
            for tool_id in metadata.get("tool_call_sequence_ids", []):
                if tool_id not in known_tool_ids:
                    reasons.append(f"capture_bad_tool_link:{node_name}:{tool_id}")
            for memory_id in metadata.get("memory_event_sequence_ids", []):
                if memory_id not in known_memory_ids:
                    reasons.append(f"capture_bad_memory_link:{node_name}:{memory_id}")

        return sorted(set(reasons))

    def _cleanup_accepted_outputs(self, run_dir: Path) -> None:
        run_output = run_dir / "run_output.json"
        if run_output.exists():
            run_output.unlink()
        agents_dir = run_dir / "agents"
        if agents_dir.exists():
            shutil.rmtree(agents_dir)

    def _mark_rejected_manifest(
        self,
        manifest: dict[str, Any],
        *,
        storage_tier: str,
        finalized_at: str,
    ) -> dict[str, Any]:
        manifest["capture_status"] = "rejected"
        manifest["usable_for_replay"] = False
        manifest["usable_for_promotion"] = False
        manifest["storage_tier"] = storage_tier
        manifest["finalized_at_utc"] = finalized_at
        manifest["first_rejection_reason"] = self._first_rejection_reason
        manifest["first_rejection_stage"] = self._first_rejection_stage
        manifest["rejection_reasons"] = self._rejection_reasons
        return manifest

    def finalize_run(self, result: dict[str, Any]) -> Path | None:
        if not self.enabled or not self._started:
            return None

        manifest = dict(self._run_manifest)
        prompts_used = self._collect_used_prompts()
        self._attach_agent_execution_profiles()
        manifest["models"] = {
            "defaults": {
                "quick_model": result.get("run_summary", {}).get("quick_model"),
                "deep_model": result.get("run_summary", {}).get("deep_model"),
                "llm_provider": result.get("run_summary", {}).get("llm_provider"),
                "llm_providers_used": normalize_for_json(
                    result.get("run_summary", {}).get("llm_providers_used", [])
                ),
            },
            "observed_models": self._observed_models(),
            "observed_providers": self._observed_providers(),
        }
        manifest["prompts"] = {"used": prompts_used}
        manifest["prompt_set_digest"] = compute_prompt_set_digest(prompts_used)
        manifest["analysis_validity"] = normalize_for_json(
            result.get("analysis_validity", {})
        )
        manifest["artifact_statuses"] = normalize_for_json(
            result.get("artifact_statuses", {})
        )

        acceptance_reasons: list[str] = []
        if not isinstance(result, dict):
            acceptance_reasons.append("result_not_dict")
        elif not result.get("analysis_validity", {}).get("publishable", False):
            acceptance_reasons.append("analysis_not_publishable")
        # Optionality follows the publication contract, not a second opinion: an
        # artifact the pipeline is willing to publish without must not veto the
        # capture of an otherwise-complete run. On 2026-08-03 all six smoke-suite
        # tickers produced publishable analyses and all six captures were rejected
        # for `artifact_failed:auditor_report` alone — the optional cross-check
        # seat timing out made a clean corpus uncapturable.
        optional_fields = self._optional_artifact_fields()
        for field, status in (result.get("artifact_statuses", {}) or {}).items():
            if field in optional_fields:
                continue
            if (
                isinstance(status, dict)
                and status.get("complete")
                and not status.get("ok", False)
            ):
                acceptance_reasons.append(f"artifact_failed:{field}")

        acceptance_reasons.extend(self._validate_pending_capture())
        if acceptance_reasons:
            self.reject_run(acceptance_reasons, stage="finalize", partial=result)

        run_dir = self._run_path()
        finalized_at = self._now_iso()
        inflight_bundle_written = False
        if not self.rejected:
            self._write_json(run_dir / "run_output.json", normalize_for_json(result))
            self._write_agent_records()
            accepted_manifest = {
                **manifest,
                "capture_status": "accepted",
                "usable_for_replay": True,
                "usable_for_promotion": True,
                "storage_tier": "inflight",
                "finalized_at_utc": finalized_at,
            }
            self._write_json(run_dir / "run_manifest.json", accepted_manifest)
            self._write_event_streams(run_dir)
            inflight_bundle_written = True
            validation = validate_capture_bundle(run_dir)
            if not validation.passed:
                self.reject_run(
                    list(validation.reasons)
                    + [
                        f"agent_validation_failed:{report.node_name}:{reason}"
                        for report in validation.agent_reports
                        for reason in report.reasons
                    ],
                    stage="capture_validation",
                )
                self._cleanup_accepted_outputs(run_dir)

        if self.rejected:
            manifest = self._mark_rejected_manifest(
                manifest,
                storage_tier="inflight",
                finalized_at=finalized_at,
            )
        else:
            manifest["capture_status"] = "accepted"
            manifest["usable_for_replay"] = True
            manifest["usable_for_promotion"] = True
            manifest["storage_tier"] = "inflight"
            manifest["finalized_at_utc"] = finalized_at

        if not inflight_bundle_written or self.rejected:
            self._write_json(run_dir / "run_manifest.json", manifest)
            self._write_event_streams(run_dir)
        destination = self._final_destination(
            status=manifest["capture_status"],
            trade_date=manifest.get("trade_date", "unknown_date"),
        )
        self._move_run_dir(destination)
        manifest["storage_tier"] = manifest["capture_status"]
        self._write_json(destination / "run_manifest.json", manifest)
        if manifest["capture_status"] == "rejected":
            self._write_json(destination / "capture_rejected.json", manifest)
        self._final_status = manifest["capture_status"]
        return destination
