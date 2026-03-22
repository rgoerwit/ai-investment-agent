from __future__ import annotations

import subprocess
from pathlib import Path


def _git_output(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def get_git_metadata(cwd: Path | None = None) -> dict[str, str | bool | int | None]:
    """Best-effort git metadata for provenance."""
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd)
    commit = _git_output(["rev-parse", "HEAD"], cwd=cwd)
    status = _git_output(["status", "--short"], cwd=cwd)
    stash_output = _git_output(["stash", "list"], cwd=cwd)
    stash_count = 0 if stash_output is None else len(stash_output.splitlines())
    return {
        "git_branch": branch,
        "git_commit": commit,
        "dirty": bool(status),
        "has_stash": stash_count > 0,
        "stash_count": stash_count,
    }
