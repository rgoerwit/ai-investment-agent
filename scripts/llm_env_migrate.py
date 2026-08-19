#!/usr/bin/env python3
"""Create a non-destructive multi-provider .env candidate from legacy keys."""

import argparse
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.config import Settings
from src.llm_runtime.bindings import resolve_binding_plan
from src.llm_runtime.identities import sanitize_endpoint_host

LEGACY_LLM_KEYS = {
    "LLM_PROVIDER",
    "QUICK_MODEL",
    "DEEP_MODEL",
    "APEX_MODEL",
    "APEX_QUICK_MODEL",
    "APEX_THINKING_LEVEL",
    "CONSULTANT_MODEL",
    "CONSULTANT_QUICK_MODEL",
    "AUDITOR_MODEL",
    "AUDITOR_QUICK_MODEL",
    "AUDITOR_ESCALATION_MODEL",
    "EDITOR_MODEL",
    "WRITER_MODEL",
    "CLAUDE_KEY",
    "ENABLE_CONSULTANT",
    "ENABLE_APAC_SPECIALIST",
    "GEMINI_RPM_LIMIT",
    "GEMINI_SERVICE_TIER",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_RPM_LIMIT",
    "APAC_SPECIALIST_MODEL",
    "APAC_SPECIALIST_BASE_URL",
    "APAC_SPECIALIST_API_KEY",
}

KNOWN_ENDPOINT_PROVIDERS = {
    "api.openai.com": "openai",
    "api.deepseek.com": "deepseek",
    "api.z.ai": "zai",
    "api.moonshot.cn": "moonshot",
    "api.moonshot.ai": "moonshot",
}


class MigrationChoiceRequired(ValueError):
    pass


@dataclass(frozen=True)
class Choices:
    review_provider: str | None = None
    regional_provider: str | None = None


def _clean_value(raw: str) -> str:
    value = raw.strip()
    if value and value[0] in {'"', "'"} and value[-1:] == value[0]:
        return value[1:-1]
    return re.split(r"\s+#", value, maxsplit=1)[0].strip()


def parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*)$", line)
        if match:
            values[match.group(1)] = _clean_value(match.group(2))
    return values


def infer_legacy_provider(
    base_url: str | None, *, default: str, explicit: str | None
) -> str:
    if explicit:
        return explicit.strip().lower()
    if not base_url:
        return default
    host = sanitize_endpoint_host(base_url)
    try:
        return KNOWN_ENDPOINT_PROVIDERS[str(host)]
    except KeyError as exc:
        raise MigrationChoiceRequired(
            f"unknown endpoint host {host!r}; provide an explicit provider"
        ) from exc


def _enabled_mode(raw: str | None) -> str:
    return (
        "auto"
        if str(raw or "").strip().lower() in {"1", "true", "yes", "on"}
        else "off"
    )


def migrate_llm_values(
    values: Mapping[str, str], choices: Choices = Choices()
) -> dict[str, str]:
    review = infer_legacy_provider(
        values.get("OPENAI_API_BASE"),
        default="openai",
        explicit=choices.review_provider,
    )
    regional = infer_legacy_provider(
        values.get("APAC_SPECIALIST_BASE_URL"),
        default="deepseek",
        explicit=choices.regional_provider,
    )
    if review not in {"google", "openai", "moonshot"}:
        raise MigrationChoiceRequired(
            f"review endpoint resolves to {review!r}, but review seats require a "
            "reviewed tool-capable Google, OpenAI, or Moonshot Kimi K3 binding"
        )
    if regional not in {"deepseek", "zai"}:
        raise MigrationChoiceRequired(
            f"regional endpoint resolves to unsupported provider {regional!r}"
        )

    quick = values.get("QUICK_MODEL", "gemini-3-flash-preview")
    deep = values.get("DEEP_MODEL", "gemini-3.1-pro-preview")
    apex = values.get("APEX_MODEL", "")
    apex_quick = values.get("APEX_QUICK_MODEL", "") or quick
    consultant = values.get("CONSULTANT_MODEL", "gpt-5.4")
    overrides = {
        "fundamentals_analyst": apex or quick,
        "portfolio_manager": apex or deep,
        "consultant": consultant,
        "forensic_auditor": values.get("AUDITOR_MODEL", consultant),
        "forensic_auditor_escalation": values.get(
            "AUDITOR_ESCALATION_MODEL", values.get("AUDITOR_MODEL", consultant)
        ),
        "article_editor": values.get("EDITOR_MODEL", consultant),
        "article_writer_review_fallback": values.get("EDITOR_MODEL", consultant),
        "apac_regional_specialist": values.get(
            "APAC_SPECIALIST_MODEL", "deepseek-v4-pro"
        ),
        "apac_regional_specialist_direct_retry": values.get(
            "APAC_SPECIALIST_MODEL", "deepseek-v4-pro"
        ),
    }
    quick_overrides = {
        "fundamentals_analyst": apex_quick if apex else quick,
        "portfolio_manager": apex_quick if apex else quick,
        "consultant": values.get("CONSULTANT_QUICK_MODEL", consultant),
        "forensic_auditor": values.get(
            "AUDITOR_QUICK_MODEL", values.get("AUDITOR_MODEL", consultant)
        ),
    }
    writer_provider = (
        "anthropic"
        if values.get("CLAUDE_KEY")
        else review
        if values.get("OPENAI_API_KEY")
        else "google"
    )
    output = {
        "LLM_BASE_PROVIDER": "google",
        "LLM_REVIEW_PROVIDER": review,
        "LLM_REGIONAL_PROVIDER": regional,
        "LLM_WRITER_PROVIDER": writer_provider,
        "LLM_OPERATIONAL_PROVIDER": "google",
        "LLM_JUDGE_PROVIDER": "google",
        "GOOGLE_LLM_FAST_MODEL": quick,
        "GOOGLE_LLM_REASONING_MODEL": deep,
        "GOOGLE_LLM_CRITICAL_MODEL": apex or deep,
        "LLM_CONSULTANT_MODE": _enabled_mode(values.get("ENABLE_CONSULTANT")),
        "LLM_AUDITOR_MODE": _enabled_mode(values.get("ENABLE_CONSULTANT")),
        "LLM_EDITOR_MODE": _enabled_mode(values.get("ENABLE_CONSULTANT")),
        "LLM_APAC_MODE": _enabled_mode(values.get("ENABLE_APAC_SPECIALIST")),
        "LLM_SEAT_MODEL_OVERRIDES": json.dumps(overrides, sort_keys=True),
        "LLM_SEAT_QUICK_MODEL_OVERRIDES": json.dumps(quick_overrides, sort_keys=True),
    }
    if apex:
        apex_reasoning = values.get("APEX_THINKING_LEVEL", "high")
        apex_seats = {
            "fundamentals_analyst": apex_reasoning,
            "portfolio_manager": apex_reasoning,
        }
        output["LLM_SEAT_REASONING_OVERRIDES"] = json.dumps(apex_seats, sort_keys=True)
        output["LLM_SEAT_QUICK_REASONING_OVERRIDES"] = json.dumps(
            apex_seats, sort_keys=True
        )
    for unchanged_key in ("OPENAI_API_KEY", "OPENAI_RPM_LIMIT"):
        if unchanged_key in values:
            output[unchanged_key] = values[unchanged_key]
    if "CLAUDE_KEY" in values:
        output["ANTHROPIC_API_KEY"] = values["CLAUDE_KEY"]
    output["GOOGLE_RPM_LIMIT"] = values.get("GEMINI_RPM_LIMIT", "15")
    output["GOOGLE_SERVICE_TIER"] = values.get("GEMINI_SERVICE_TIER", "standard")
    if review == "moonshot":
        output.pop("OPENAI_API_KEY", None)
        output.pop("OPENAI_RPM_LIMIT", None)
        output.update(
            MOONSHOT_LLM_FAST_MODEL=values.get("CONSULTANT_QUICK_MODEL", consultant),
            MOONSHOT_LLM_REASONING_MODEL=consultant,
            MOONSHOT_LLM_CRITICAL_MODEL=values.get("EDITOR_MODEL", consultant),
            MOONSHOT_LLM_ESCALATION_MODEL=values.get(
                "AUDITOR_ESCALATION_MODEL", values.get("AUDITOR_MODEL", consultant)
            ),
            MOONSHOT_API_BASE=values.get(
                "OPENAI_API_BASE", "https://api.moonshot.ai/v1"
            ),
            MOONSHOT_API_KEY=values.get("OPENAI_API_KEY", ""),
        )
        if values.get("OPENAI_RPM_LIMIT"):
            output["MOONSHOT_RPM_LIMIT"] = values["OPENAI_RPM_LIMIT"]
    if regional == "zai":
        output.update(
            ZAI_LLM_REASONING_MODEL=values.get("APAC_SPECIALIST_MODEL", "glm-5.2"),
            ZAI_API_BASE=values.get(
                "APAC_SPECIALIST_BASE_URL", "https://api.z.ai/api/paas/v4/"
            ),
            ZAI_API_KEY=values.get("APAC_SPECIALIST_API_KEY", ""),
        )
    else:
        output.update(
            DEEPSEEK_LLM_REASONING_MODEL=values.get(
                "APAC_SPECIALIST_MODEL", "deepseek-v4-pro"
            ),
            DEEPSEEK_API_BASE=values.get(
                "APAC_SPECIALIST_BASE_URL", "https://api.deepseek.com"
            ),
            DEEPSEEK_API_KEY=values.get("APAC_SPECIALIST_API_KEY", ""),
        )
    if values.get("CLAUDE_KEY"):
        output["ANTHROPIC_LLM_PROSE_MODEL"] = values.get(
            "WRITER_MODEL", "claude-opus-4-6"
        )
    return output


def render_migration(source_text: str, migrated: Mapping[str, str]) -> str:
    rendered: list[str] = []
    for line in source_text.splitlines():
        match = re.match(r"^(\s*)(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=", line)
        if match and match.group(2) in LEGACY_LLM_KEYS:
            disposition = (
                "retired metadata-only key"
                if match.group(2) == "LLM_PROVIDER"
                else "migrated legacy key"
            )
            rendered.append(f"{match.group(1)}# {disposition}: {line.strip()}")
        else:
            rendered.append(line)
    rendered.extend(["", "# Multi-provider LLM bindings (generated)"])
    for key, value in migrated.items():
        rendered.append(f"{key}={value}")
    return "\n".join(rendered) + "\n"


def _validate_candidate(values: Mapping[str, str]) -> None:
    aliases: dict[str, str] = {}
    for name, field in Settings.model_fields.items():
        alias = field.validation_alias
        if isinstance(alias, str):
            aliases[alias] = name
        else:
            for choice in getattr(alias, "choices", ()):
                if isinstance(choice, str):
                    aliases[choice] = name
    kwargs = {
        aliases[key]: value
        for key, value in values.items()
        if key in aliases and isinstance(aliases[key], str)
    }
    for field_name in (
        "llm_seat_model_overrides",
        "llm_seat_quick_model_overrides",
        "llm_seat_reasoning_overrides",
        "llm_seat_quick_reasoning_overrides",
    ):
        raw_value = kwargs.get(field_name)
        if isinstance(raw_value, str):
            kwargs[field_name] = json.loads(raw_value)
    resolve_binding_plan(Settings(_env_file=None, **kwargs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--review-provider")
    parser.add_argument("--regional-provider")
    args = parser.parse_args()
    if not args.check and args.output is None:
        parser.error("--output is required unless --check is used")
    if args.output and args.source.resolve() == args.output.resolve():
        parser.error("refusing to overwrite the source .env")

    source_text = args.source.read_text(encoding="utf-8")
    source_values = parse_env(source_text)
    try:
        migrated = migrate_llm_values(
            source_values,
            Choices(args.review_provider, args.regional_provider),
        )
        candidate_values = {
            key: value
            for key, value in source_values.items()
            if key not in LEGACY_LLM_KEYS
        }
        candidate_values.update(migrated)
        _validate_candidate(candidate_values)
    except ValueError as exc:
        parser.exit(2, f"migration error: {exc}\n")
    if args.check:
        print("migration candidate validates; no file written")
        return 0
    assert args.output is not None
    args.output.write_text(render_migration(source_text, migrated), encoding="utf-8")
    print(f"wrote migration candidate: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
