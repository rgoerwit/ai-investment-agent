from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.tooling.inspector import SourceKind

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "injection_payloads"
CORPUS_PATH = FIXTURE_DIR / "corpus.json"
FLOORS_PATH = FIXTURE_DIR / "floors.json"

_ACTIONS = {"allow", "sanitize", "block", "degrade"}
_EXPECTATIONS = {
    "must_block",
    "block_or_sanitize",
    "floor_only",
    "semantic_replay",
    "current_policy",
}
_REQUIRED = {
    "id",
    "payload",
    "source_kind",
    "category",
    "expected_action",
    "expectation",
    "lang",
    "origin",
    "license",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{path}: invalid JSON: {exc}") from exc


def _validate_entry(entry: object, *, seen: set[str]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise AssertionError(f"{CORPUS_PATH}: corpus entry is not an object")

    missing = sorted(_REQUIRED.difference(entry))
    case_id = str(entry.get("id", "<missing-id>"))
    if missing:
        raise AssertionError(f"{CORPUS_PATH}: {case_id} missing fields: {missing}")

    if not isinstance(entry["id"], str) or not entry["id"].strip():
        raise AssertionError(f"{CORPUS_PATH}: entry has empty id")
    if entry["id"] in seen:
        raise AssertionError(f"{CORPUS_PATH}: duplicate corpus id {entry['id']}")
    seen.add(entry["id"])

    if not isinstance(entry["payload"], str) or not entry["payload"].strip():
        raise AssertionError(f"{CORPUS_PATH}: {case_id} has empty payload")

    try:
        SourceKind(str(entry["source_kind"]))
    except ValueError as exc:
        raise AssertionError(
            f"{CORPUS_PATH}: {case_id} has unknown source_kind {entry['source_kind']!r}"
        ) from exc

    if entry["expected_action"] not in _ACTIONS:
        raise AssertionError(
            f"{CORPUS_PATH}: {case_id} has invalid expected_action "
            f"{entry['expected_action']!r}"
        )
    if entry["expectation"] not in _EXPECTATIONS:
        raise AssertionError(
            f"{CORPUS_PATH}: {case_id} has invalid expectation {entry['expectation']!r}"
        )
    return entry


@lru_cache(maxsize=1)
def _load_validated_corpus() -> tuple[dict[str, Any], ...]:
    raw = _read_json(CORPUS_PATH)
    if not isinstance(raw, list):
        raise AssertionError(f"{CORPUS_PATH}: top-level value must be a list")

    seen: set[str] = set()
    return tuple(_validate_entry(entry, seen=seen) for entry in raw)


def load_corpus(
    *,
    source_kind: str | None = None,
    category: str | None = None,
    expectation: str | None = None,
) -> list[dict[str, Any]]:
    entries = list(_load_validated_corpus())
    if source_kind is not None:
        entries = [case for case in entries if case["source_kind"] == source_kind]
    if category is not None:
        entries = [case for case in entries if case["category"] == category]
    if expectation is not None:
        entries = [case for case in entries if case["expectation"] == expectation]
    return entries


def load_detection_floors() -> list[dict[str, Any]]:
    raw = _read_json(FLOORS_PATH)
    if not isinstance(raw, list):
        raise AssertionError(f"{FLOORS_PATH}: top-level value must be a list")
    for index, bucket in enumerate(raw):
        if not isinstance(bucket, dict):
            raise AssertionError(f"{FLOORS_PATH}: bucket {index} is not an object")
        for key in ("source_kind", "category", "floor"):
            if key not in bucket:
                raise AssertionError(f"{FLOORS_PATH}: bucket {index} missing {key}")
        SourceKind(str(bucket["source_kind"]))
        floor = bucket["floor"]
        if not isinstance(floor, int | float) or not 0 <= floor <= 1:
            raise AssertionError(f"{FLOORS_PATH}: bucket {index} has invalid floor")
    return raw
