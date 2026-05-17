#!/usr/bin/env python3
"""Validate or replace the vendored injection corpus from a local JSON file.

This script intentionally does not fetch remote sources. Upstream corpora must
be downloaded and reviewed separately, then passed as a local normalized JSON
file with ``--source-file``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "tests" / "fixtures" / "injection_payloads" / "corpus.json"
SOURCES_PATH = ROOT / "tests" / "fixtures" / "injection_payloads" / "SOURCES.md"


def _load(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise SystemExit(f"{path}: top-level value must be a list")
    if not all(isinstance(entry, dict) for entry in data):
        raise SystemExit(f"{path}: every corpus entry must be an object")
    return data


def _validate_candidate(candidate: list[dict]) -> None:
    from tests.helpers.injection_corpus import _validate_entry

    seen: set[str] = set()
    for entry in candidate:
        _validate_entry(entry, seen=seen)


def _verify_sha(path: Path, expected_sha: str) -> str:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual.lower() != expected_sha.lower():
        raise SystemExit(
            f"{path}: sha256 mismatch: expected {expected_sha}, got {actual}"
        )
    return actual


def _append_source_audit(source_file: Path, sha256: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    line = f"\n- {timestamp}: refreshed from `{source_file}` with sha256 `{sha256}`.\n"
    SOURCES_PATH.write_text(
        SOURCES_PATH.read_text(encoding="utf-8") + line,
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, help="local normalized corpus JSON")
    parser.add_argument("--source-sha", help="expected sha256 for --source-file")
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the vendored corpus with --source-file after validation",
    )
    args = parser.parse_args()

    current = _load(CORPUS_PATH)
    if args.source_file is None:
        print(f"Current corpus: {len(current)} entries")
        print("No source file supplied; dry run complete.")
        return 0

    candidate = _load(args.source_file)
    _validate_candidate(candidate)
    actual_sha = (
        _verify_sha(args.source_file, args.source_sha)
        if args.source_sha
        else hashlib.sha256(args.source_file.read_bytes()).hexdigest()
    )
    print(f"Current corpus: {len(current)} entries")
    print(f"Candidate corpus: {len(candidate)} entries")
    print(f"Candidate sha256: {actual_sha}")

    if not args.write:
        print("Dry run only. Re-run with --write after reviewing the candidate diff.")
        return 0

    shutil.copyfile(args.source_file, CORPUS_PATH)
    if args.source_sha:
        _append_source_audit(args.source_file, actual_sha)
    print(f"Replaced {CORPUS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
