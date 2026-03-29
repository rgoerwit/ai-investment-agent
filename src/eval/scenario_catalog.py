from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SUITE_NAME = "smoke"
_SUITE_DIR = Path("evals") / "prompt_check_suites"


@dataclass(frozen=True)
class PromptCheckScenario:
    ticker: str
    quick: bool = True
    strict: bool = False


@dataclass(frozen=True)
class PromptCheckSuite:
    name: str
    description: str
    scenarios: tuple[PromptCheckScenario, ...]


def load_prompt_check_suite(name: str | None = None) -> PromptCheckSuite:
    return load_prompt_check_suite_from_path(resolve_prompt_check_suite_path(name))


def load_prompt_check_suite_from_path(path: Path) -> PromptCheckSuite:
    payload = json.loads(path.read_text(encoding="utf-8"))
    suite_name = str(payload.get("suite") or path.stem)
    description = str(payload.get("description") or "")
    raw_scenarios = payload.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError(f"Suite manifest has no scenarios: {path}")

    scenarios: list[PromptCheckScenario] = []
    seen: set[tuple[str, bool, bool]] = set()
    for raw in raw_scenarios:
        if not isinstance(raw, Mapping) or not raw.get("ticker"):
            raise ValueError(f"Invalid scenario entry in {path}: {raw!r}")
        scenario = PromptCheckScenario(
            ticker=str(raw["ticker"]),
            quick=bool(raw.get("quick", True)),
            strict=bool(raw.get("strict", False)),
        )
        dedupe_key = (scenario.ticker, scenario.quick, scenario.strict)
        if dedupe_key in seen:
            raise ValueError(f"Duplicate scenario in {path}: {dedupe_key}")
        seen.add(dedupe_key)
        scenarios.append(scenario)

    return PromptCheckSuite(
        name=suite_name,
        description=description,
        scenarios=tuple(scenarios),
    )


def resolve_prompt_check_suite_path(name: str | None = None) -> Path:
    suite_name = name or DEFAULT_SUITE_NAME
    path = _SUITE_DIR / f"{suite_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Prompt-check suite not found: {path}")
    return path
