#!/usr/bin/env python3
"""Validate the public Codex metadata surface without reading private state."""

from __future__ import annotations

import glob
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {"", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".sh", ".py"}
GENERATED_PREFIXES = (
    "results/",
    "runtime/",
    "outputs/",
    "reports/",
    "logs/",
    "scratch/",
    "evals/captures/",
    "evals/prompt_checks/",
)
MACHINE_LOCAL_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_])(?:~[/\\]|\$\{?HOME\}?|/Users/|/home/|/private/|"
    r"/Volumes/|/var/folders/|[A-Za-z]:\\)"
)
CROSS_TOOL_RE = re.compile(
    r"(?:CLAUDE[-.A-Za-z0-9]*\.md|GEMINI[-.A-Za-z0-9]*\.md|"
    r"(?<![A-Za-z0-9_.-])\.(?:claude|gemini)(?:/|\b))",
    re.IGNORECASE,
)
ENV_ASSIGNMENT_RE = re.compile(
    r"(?m)(?<![A-Za-z0-9_])(?:export\s+)?[A-Z][A-Z0-9_]{2,}\s*=\s*[^\s`]"
    r'|^\s*(?:["\'][A-Z][A-Z0-9_]{2,}["\']|'
    r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\s*:\s*[^\s`]"
)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
FENCED_CODE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
PATH_TOKEN_RE = re.compile(
    r"^(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.*{}<>/-]+(?::\d+)?$"
    r"|^[A-Za-z0-9_.-]+\.(?:md|json|toml|ya?ml|py|sh)$"
    r"|^Makefile$"
)
PUBLIC_ENV_TOKENS = {".env", ".env.example", ".env*"}


def _frontmatter(text: str) -> dict[str, Any]:
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    data = yaml.safe_load(text[4:end])
    return data if isinstance(data, dict) else {}


def _repo_relative(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def _is_public_surface(relative: str) -> bool:
    return relative == "AGENTS.md" or relative.startswith((".agents/", ".codex/"))


def _indexed_paths() -> frozenset[str]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        return frozenset()
    return frozenset(
        value.decode("utf-8", errors="surrogateescape")
        for value in completed.stdout.split(b"\0")
        if value
    )


def _target_is_indexed(target: Path, indexed_paths: frozenset[str]) -> bool:
    try:
        relative = target.relative_to(ROOT).as_posix()
    except ValueError:
        return False
    if target.is_dir():
        prefix = f"{relative.rstrip('/')}/"
        return any(path.startswith(prefix) for path in indexed_paths)
    return relative in indexed_paths


def _reference_tokens(text: str) -> list[str]:
    spans = [*INLINE_CODE_RE.findall(text), *FENCED_CODE_RE.findall(text)]
    return list(
        dict.fromkeys(
            raw.strip("'\"()[]{}<>,")
            for span in spans
            for raw in span.split()
            if raw.strip("'\"()[]{}<>,")
        )
    )


def _validate_reference(
    token: str,
    source: Path,
    *,
    indexed_paths: frozenset[str] | None = None,
) -> list[str]:
    token = token.rstrip(".,;:)")
    basename = token.rsplit("/", 1)[-1]
    if basename.startswith(".env") and basename not in PUBLIC_ENV_TOKENS:
        return [
            f"{source}: names a private environment-file variant; "
            "only .env and .env.example may be named"
        ]
    if not PATH_TOKEN_RE.fullmatch(token) or "<" in token or ">" in token:
        return []
    if token in PUBLIC_ENV_TOKENS or token.startswith(GENERATED_PREFIXES):
        return []
    token = re.sub(r":\d+$", "", token)
    matches = (
        glob.glob(str(ROOT / token), recursive=True)
        if any(ch in token for ch in "*?[")
        else []
    )
    target = ROOT / token
    candidates = [Path(value) for value in matches] or [target]
    if not target.exists() and not matches:
        return [f"{source}: references missing repository path: {token}"]
    if indexed_paths is not None and not any(
        _target_is_indexed(candidate, indexed_paths) for candidate in candidates
    ):
        return [f"{source}: references untracked repository path: {token}"]
    return []


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    if path.is_symlink():
        return [f"{path}: public Codex metadata must not be a symlink"]
    relative = _repo_relative(path)
    if relative is None or not _is_public_surface(relative):
        return [f"{path}: outside the public Codex metadata surface"]
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return errors

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path}: textual Codex metadata must be UTF-8"]

    if MACHINE_LOCAL_RE.search(text):
        errors.append(f"{path}: contains a machine-local or home-relative path")
    if ENV_ASSIGNMENT_RE.search(text):
        errors.append(f"{path}: contains an environment-style assignment")
    if relative.startswith((".agents/", ".codex/")) and CROSS_TOOL_RE.search(text):
        errors.append(
            f"{path}: repository Codex extension cites another tool's metadata"
        )
    if relative == "AGENTS.md" and re.search(
        r"CLAUDE[-.A-Za-z0-9]*\.md|GEMINI[-.A-Za-z0-9]*\.md", text, re.IGNORECASE
    ):
        errors.append(
            f"{path}: root guidance cites an ignored tool-specific instruction file"
        )

    if path.name == "SKILL.md":
        metadata = _frontmatter(text)
        expected_name = path.parent.name
        if metadata.get("name") != expected_name:
            errors.append(
                f"{path}: skill name must equal its directory name ({expected_name})"
            )
        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{path}: skill description is required")

    indexed_paths = _indexed_paths()
    reference_index = indexed_paths if relative in indexed_paths else None
    for token in _reference_tokens(text):
        errors.extend(_validate_reference(token, path, indexed_paths=reference_index))
    return errors


def main(argv: list[str]) -> int:
    paths = [
        Path(value)
        for value in argv[1:]
        if Path(value).is_file() or Path(value).is_symlink()
    ]
    errors = [error for path in paths for error in validate_file(path)]
    if paths:
        completed = subprocess.run(
            ["bash", str(ROOT / "scripts/check_tracked_deps.sh"), *map(str, paths)],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode:
            errors.append("Tracked-dependency validation failed for Codex metadata")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
