#!/usr/bin/env python3
"""Validate the public Claude metadata surface without reading private state."""

from __future__ import annotations

import glob
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
RULE_PREFIX = ".claude/rules/"
SKILL_PREFIX = ".claude/skills/"
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
    r"(?:AGENTS[-.A-Za-z0-9_]*\.md|GEMINI[-.A-Za-z0-9_]*\.md|"
    r"(?<![A-Za-z0-9_.-])\.(?:agents|codex|gemini|cursor|windsurf|aider|"
    r"cline|continue|roo|junie|amp)(?:/|\b)|\.cursorrules|\.clinerules|"
    r"\.windsurfrules|\.github/copilot-instructions\.md)",
    re.IGNORECASE,
)
ENV_ASSIGNMENT_RE = re.compile(
    r"(?m)(?<![A-Za-z0-9_])(?:export\s+)?[A-Z][A-Z0-9_]{2,}\s*=\s*[^\s`]"
    r'|^\s*(?:["\'][A-Z][A-Z0-9_]{2,}["\']|'
    r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\s*:\s*[^\s`]"
)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
FENCED_CODE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
ACTIVE_IMPORT_RE = re.compile(
    r"(?<![A-Za-z0-9_])@([~./A-Za-z0-9_-][^\s<>()\[\]{},\"'`]+)"
)
PATH_TOKEN_RE = re.compile(
    r"^(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.*{}<>/-]+(?::\d+)?$"
    r"|^[A-Za-z0-9_.-]+\.(?:md|json|toml|ya?ml|py|sh)$"
    r"|^Makefile$"
)
SYMBOLIC_SLASH_RE = re.compile(r"^[A-Z][A-Z0-9]*(?:/[A-Z][A-Z0-9]*)+$")
PUBLIC_ENV_TOKENS = {".env", ".env.example", ".env*"}


def _git_indexed_paths() -> frozenset[str]:
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


def _frontmatter(text: str) -> tuple[dict[str, Any], str | None]:
    if not text.startswith("---\n"):
        return {}, None
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, "unterminated YAML frontmatter"
    try:
        data = yaml.safe_load(text[4:end])
    except yaml.YAMLError as exc:
        return {}, f"invalid YAML frontmatter: {exc.problem or 'parse error'}"
    return (data if isinstance(data, dict) else {}), None


def _repo_relative(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return None


def _is_allowed_surface(relative: str) -> bool:
    if relative.startswith(RULE_PREFIX):
        return relative.endswith(".md")
    if not relative.startswith(SKILL_PREFIX):
        return False
    remainder = relative.removeprefix(SKILL_PREFIX)
    parts = remainder.split("/")
    return len(parts) == 2 and parts[0] != "synced" and parts[1] == "SKILL.md"


def validate_surface_inventory() -> list[str]:
    return [
        f"{relative}: unexpected tracked file in the public Claude metadata surface"
        for relative in sorted(_git_indexed_paths())
        if relative.startswith(".claude/") and not _is_allowed_surface(relative)
    ]


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


def _active_import_tokens(text: str) -> list[str]:
    without_fences = FENCED_CODE_RE.sub("", text)
    without_code = INLINE_CODE_RE.sub("", without_fences)
    return list(dict.fromkeys(ACTIVE_IMPORT_RE.findall(without_code)))


def _validate_reference(
    token: str,
    source: Path,
    *,
    indexed_paths: frozenset[str] | None,
) -> list[str]:
    token = token.rstrip(".,;:)")
    basename = token.rsplit("/", 1)[-1]
    if basename.startswith(".env") and basename not in PUBLIC_ENV_TOKENS:
        return [
            f"{source}: names a private environment-file variant; "
            "only .env and .env.example may be named"
        ]
    if (
        not PATH_TOKEN_RE.fullmatch(token)
        or SYMBOLIC_SLASH_RE.fullmatch(token)
        or "<" in token
        or ">" in token
    ):
        return []
    if token in PUBLIC_ENV_TOKENS or token.startswith(GENERATED_PREFIXES):
        return []
    token = re.sub(r":\d+$", "", token)
    matches = (
        glob.glob(str(ROOT / token), recursive=True)
        if any(character in token for character in "*?[")
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


def _validate_import(
    token: str,
    source: Path,
    indexed_paths: frozenset[str] | None,
) -> list[str]:
    token = token.rstrip(".,;:)")
    if token.startswith("~") or Path(token).is_absolute():
        return [f"{source}: imports a machine-local path: @{token}"]
    target = (source.parent / token).resolve()
    try:
        relative = target.relative_to(ROOT).as_posix()
    except ValueError:
        return [f"{source}: imports content outside the repository: @{token}"]
    if not target.is_file():
        return [f"{source}: imports a missing repository file: @{token}"]
    if indexed_paths is not None and relative not in indexed_paths:
        return [f"{source}: imports an untracked repository file: @{token}"]
    return []


def validate_file(path: Path) -> list[str]:
    if path.is_symlink():
        return [f"{path}: public Claude metadata must not be a symlink"]
    relative = _repo_relative(path)
    if relative is None or not _is_allowed_surface(relative):
        return [f"{path}: outside the public Claude metadata surface"]
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [f"{path}: textual Claude metadata must be UTF-8"]

    errors: list[str] = []
    if MACHINE_LOCAL_RE.search(text):
        errors.append(f"{path}: contains a machine-local or home-relative path")
    if CROSS_TOOL_RE.search(text):
        errors.append(f"{path}: cites another coding tool's metadata")
    if ENV_ASSIGNMENT_RE.search(text):
        errors.append(f"{path}: contains an environment-style assignment")

    metadata, frontmatter_error = _frontmatter(text)
    if frontmatter_error:
        errors.append(f"{path}: {frontmatter_error}")
    if relative.startswith(SKILL_PREFIX):
        expected_name = path.parent.name
        if metadata.get("name") != expected_name:
            errors.append(
                f"{path}: skill name must equal its directory name ({expected_name})"
            )
        description = metadata.get("description")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{path}: skill description is required")
    elif "paths" in metadata:
        scoped_paths = metadata["paths"]
        if (
            not isinstance(scoped_paths, list)
            or not scoped_paths
            or not all(
                isinstance(value, str) and value.strip() for value in scoped_paths
            )
        ):
            errors.append(f"{path}: rule paths must be a non-empty string list")

    indexed_paths = _git_indexed_paths()
    reference_index = indexed_paths if relative in indexed_paths else None
    for token in _reference_tokens(text):
        errors.extend(_validate_reference(token, path, indexed_paths=reference_index))
    for token in _active_import_tokens(text):
        errors.extend(_validate_import(token, path, reference_index))
    return errors


def main(argv: list[str]) -> int:
    paths = [
        Path(value)
        for value in argv[1:]
        if Path(value).is_file() or Path(value).is_symlink()
    ]
    public_paths = [
        path
        for path in paths
        if (relative := _repo_relative(path)) is not None
        and _is_allowed_surface(relative)
    ]
    errors = validate_surface_inventory()
    errors.extend(error for path in paths for error in validate_file(path))
    if public_paths:
        completed = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/check_tracked_deps.sh"),
                *map(str, public_paths),
            ],
            cwd=ROOT,
            check=False,
        )
        if completed.returncode:
            errors.append("Tracked-dependency validation failed for Claude metadata")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
