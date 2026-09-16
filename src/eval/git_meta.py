from __future__ import annotations

import subprocess
from pathlib import Path

GIT_COMMAND_TIMEOUT_SECONDS = 5.0


def _git_output(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_COMMAND_TIMEOUT_SECONDS,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def get_git_metadata(cwd: Path | None = None) -> dict[str, str | bool | int | None]:
    """Best-effort git metadata for provenance."""
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd) or None
    commit = _git_output(["rev-parse", "HEAD"], cwd=cwd) or None
    status = _git_output(["status", "--short"], cwd=cwd)
    stash_output = _git_output(["stash", "list"], cwd=cwd)
    stash_count = len(stash_output.splitlines()) if stash_output else 0
    return {
        "git_branch": branch,
        "git_commit": commit,
        # A missing status is unknown, never evidence that the worktree is clean.
        "dirty": status is None or bool(status),
        "has_stash": stash_count > 0,
        "stash_count": stash_count,
    }
