import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.eval.git_meta import GIT_COMMAND_TIMEOUT_SECONDS, get_git_metadata

_GIT = "/usr/bin/git"
_CLEAN = {
    ("rev-parse", "--abbrev-ref", "HEAD"): "main\n",
    ("rev-parse", "HEAD"): "abc123\n",
    ("status", "--short"): "",
    ("stash", "list"): "",
}


class _Completed:
    def __init__(self, stdout: str):
        self.stdout = stdout


class _FakeGit:
    """Answers git subcommands through the spawn helper, recording each call."""

    def __init__(self, responses: dict[tuple[str, ...], Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, tuple[str, ...], float]] = []

    def __call__(self, executable: str, *args: str, timeout: float) -> _Completed:
        self.calls.append((executable, args, timeout))
        command = args[2:] if args[:1] == ("-C",) else args
        answer = self.responses[command]
        if isinstance(answer, BaseException):
            raise answer
        return _Completed(answer)


@pytest.fixture
def fake_git(monkeypatch):
    def install(responses: dict[tuple[str, ...], Any]) -> _FakeGit:
        fake = _FakeGit(responses)
        monkeypatch.setattr("src.eval.git_meta.resolve_program", lambda name: _GIT)
        monkeypatch.setattr("src.eval.git_meta.run_program_sync", fake)
        return fake

    return install


def test_get_git_metadata_reports_clean_repo(fake_git):
    fake_git(_CLEAN)

    assert get_git_metadata() == {
        "git_branch": "main",
        "git_commit": "abc123",
        "dirty": False,
        "has_stash": False,
        "stash_count": 0,
    }


def test_get_git_metadata_reports_dirty_repo(fake_git):
    fake_git({**_CLEAN, ("status", "--short"): " M src/main.py\n"})

    metadata = get_git_metadata()

    assert metadata["dirty"] is True
    assert metadata["has_stash"] is False
    assert metadata["stash_count"] == 0


def test_get_git_metadata_handles_git_failing(fake_git):
    failure = subprocess.CalledProcessError(returncode=128, cmd="git")
    fake_git(dict.fromkeys(_CLEAN, failure))

    assert get_git_metadata() == {
        "git_branch": None,
        "git_commit": None,
        "dirty": True,
        "has_stash": False,
        "stash_count": 0,
    }


def test_an_absent_git_binary_spawns_nothing(monkeypatch):
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no git to run")

    monkeypatch.setattr("src.eval.git_meta.resolve_program", lambda name: None)
    monkeypatch.setattr("src.eval.git_meta.run_program_sync", forbidden)

    metadata = get_git_metadata()

    assert metadata["git_commit"] is None
    assert metadata["dirty"] is True


def test_get_git_metadata_records_stash_advisory(fake_git):
    fake_git(
        {
            **_CLEAN,
            ("stash", "list"): "stash@{0}: WIP on main\nstash@{1}: WIP on feat\n",
        }
    )

    metadata = get_git_metadata()

    assert metadata["dirty"] is False
    assert metadata["has_stash"] is True
    assert metadata["stash_count"] == 2


def test_every_git_command_is_bounded_and_runs_the_resolved_binary(fake_git):
    fake = fake_git(_CLEAN)

    get_git_metadata()

    assert [(executable, timeout) for executable, _, timeout in fake.calls] == [
        (_GIT, GIT_COMMAND_TIMEOUT_SECONDS)
    ] * 4


def test_a_directory_is_passed_as_git_dash_c_never_as_cwd(fake_git, tmp_path: Path):
    """A working directory would send CPython down the fork path."""
    fake = fake_git(_CLEAN)

    get_git_metadata(tmp_path)

    assert all(args[:2] == ("-C", str(tmp_path)) for _, args, _ in fake.calls)


def test_status_timeout_is_not_misreported_as_clean(fake_git):
    fake_git(
        {
            **_CLEAN,
            ("status", "--short"): subprocess.TimeoutExpired(
                "git", GIT_COMMAND_TIMEOUT_SECONDS
            ),
        }
    )

    metadata = get_git_metadata()

    assert metadata["git_commit"] == "abc123"
    assert metadata["dirty"] is True
