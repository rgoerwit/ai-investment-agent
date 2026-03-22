import subprocess

from src.eval.git_meta import get_git_metadata


class _Completed:
    def __init__(self, stdout: str):
        self.stdout = stdout


def test_get_git_metadata_reports_clean_repo(monkeypatch):
    responses = {
        ("rev-parse", "--abbrev-ref", "HEAD"): "main\n",
        ("rev-parse", "HEAD"): "abc123\n",
        ("status", "--short"): "",
        ("stash", "list"): "",
    }

    def fake_run(args, **kwargs):
        return _Completed(responses[tuple(args[1:])])

    monkeypatch.setattr("src.eval.git_meta.subprocess.run", fake_run)
    metadata = get_git_metadata()

    assert metadata == {
        "git_branch": "main",
        "git_commit": "abc123",
        "dirty": False,
        "has_stash": False,
        "stash_count": 0,
    }


def test_get_git_metadata_reports_dirty_repo(monkeypatch):
    responses = {
        ("rev-parse", "--abbrev-ref", "HEAD"): "main\n",
        ("rev-parse", "HEAD"): "abc123\n",
        ("status", "--short"): " M src/main.py\n",
        ("stash", "list"): "",
    }

    def fake_run(args, **kwargs):
        return _Completed(responses[tuple(args[1:])])

    monkeypatch.setattr("src.eval.git_meta.subprocess.run", fake_run)
    metadata = get_git_metadata()

    assert metadata["dirty"] is True
    assert metadata["has_stash"] is False
    assert metadata["stash_count"] == 0


def test_get_git_metadata_handles_git_unavailable(monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args)

    monkeypatch.setattr("src.eval.git_meta.subprocess.run", fake_run)
    metadata = get_git_metadata()

    assert metadata == {
        "git_branch": None,
        "git_commit": None,
        "dirty": False,
        "has_stash": False,
        "stash_count": 0,
    }


def test_get_git_metadata_records_stash_advisory(monkeypatch):
    responses = {
        ("rev-parse", "--abbrev-ref", "HEAD"): "main\n",
        ("rev-parse", "HEAD"): "abc123\n",
        ("status", "--short"): "",
        ("stash", "list"): "stash@{0}: WIP on main\nstash@{1}: WIP on feat\n",
    }

    def fake_run(args, **kwargs):
        return _Completed(responses[tuple(args[1:])])

    monkeypatch.setattr("src.eval.git_meta.subprocess.run", fake_run)
    metadata = get_git_metadata()

    assert metadata["dirty"] is False
    assert metadata["has_stash"] is True
    assert metadata["stash_count"] == 2
