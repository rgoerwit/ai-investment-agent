#!/usr/bin/env python3
"""Validate or record the frozen LLM judge replay fixture.

Default mode is offline validation only. ``--record`` intentionally calls the
live judge model, rewrites the replay fixture with raw judge responses, and must
be run manually with a real API key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "tests" / "fixtures" / "injection_payloads" / "corpus.json"
REPLAY_PATH = ROOT / "tests" / "fixtures" / "judge_replay.json"


def _semantic_cases() -> list[dict[str, Any]]:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return [case for case in corpus if case.get("expectation") == "semantic_replay"]


def _envelope(case: dict[str, Any]):
    from src.tooling.inspector import InspectionEnvelope, SourceKind

    return InspectionEnvelope(
        content_text=case["payload"],
        raw_content=case["payload"],
        source_kind=SourceKind(case["source_kind"]),
        source_name="corpus",
    )


def _missing_replay_keys() -> list[str]:
    from src.tooling.llm_judge_inspector import LLMJudgeInspector

    replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    judge = LLMJudgeInspector(llm=object())
    missing: list[str] = []
    for case in _semantic_cases():
        key = judge._build_cache_key(_envelope(case))
        if key not in replay:
            missing.append(f"{case['id']} {key}")
    return missing


async def _record() -> dict[str, str]:
    from src.agents.message_utils import extract_string_content
    from src.agents.runtime import invoke_with_rate_limit_handling
    from src.tooling.llm_judge_inspector import LLMJudgeInspector

    recorded: dict[str, str] = {}
    current_key: str | None = None

    async def recording_invoker(llm, messages):
        nonlocal current_key
        response = await invoke_with_rate_limit_handling(
            llm,
            messages,
            context="Prompt Injection Judge Fixture Refresh",
        )
        raw = extract_string_content(getattr(response, "content", response))
        if current_key is None:
            raise RuntimeError("recording key was not set")
        recorded[current_key] = raw
        return response

    judge = LLMJudgeInspector(invoker=recording_invoker)
    for case in _semantic_cases():
        envelope = _envelope(case)
        current_key = judge._build_cache_key(envelope)
        await judge.inspect(envelope)
    return recorded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record",
        action="store_true",
        help="call the live judge and rewrite tests/fixtures/judge_replay.json",
    )
    args = parser.parse_args()

    if not args.record:
        missing = _missing_replay_keys()
        if missing:
            print("Missing judge replay keys:")
            print("\n".join(missing))
            return 1
        replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
        print(f"Judge replay fixture is complete: {len(replay)} entries")
        return 0

    from src.config import get_env_value

    api_key = get_env_value("GOOGLE_API_KEY")
    if not api_key or api_key == "test-key":
        raise SystemExit("--record requires a real GOOGLE_API_KEY")

    recorded = asyncio.run(_record())
    REPLAY_PATH.write_text(
        json.dumps(recorded, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Recorded {len(recorded)} judge replay responses into {REPLAY_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
