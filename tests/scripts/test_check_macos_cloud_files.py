from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

from scripts import check_macos_cloud_files as cloud_files
from scripts.check_macos_cloud_files import (
    SF_DATALESS,
    find_dataless_files,
    format_pytest_error,
    generated_public_bytecode_files,
    materialize_files,
)


def test_candidate_enumeration_delegates_ignored_file_filtering_to_git(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(b"tracked.py\0new.py\0.env\0.env.local\0.env.example\0.envrc\0"),
        )

    monkeypatch.setattr(cloud_files.subprocess, "run", fake_run)

    assert cloud_files.repository_candidate_files(tmp_path) == [
        tmp_path / "tracked.py",
        tmp_path / "new.py",
        tmp_path / ".env.example",
    ]
    assert observed["command"] == [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ]
    assert observed["kwargs"] == {
        "cwd": tmp_path,
        "capture_output": True,
        "check": False,
    }


def test_bytecode_enumeration_includes_only_public_source_derivatives(
    tmp_path: Path,
) -> None:
    package = tmp_path / "package"
    cache = package / "__pycache__"
    cache.mkdir(parents=True)
    public_source = package / "public.py"
    unrelated_source = package / "private.py"
    public_bytecode = cache / "public.cpython-312.pyc"
    unrelated_bytecode = cache / "private.cpython-312.pyc"
    public_source.write_text("", encoding="utf-8")
    unrelated_source.write_text("", encoding="utf-8")
    public_bytecode.write_bytes(b"public")
    unrelated_bytecode.write_bytes(b"private")

    assert generated_public_bytecode_files(
        [public_source], cache_tag="cpython-312"
    ) == [public_bytecode]


def test_dataless_detection_uses_flags_without_reading_files(tmp_path: Path) -> None:
    local = tmp_path / "local.py"
    offloaded = tmp_path / "offloaded.py"
    local.write_text("local", encoding="utf-8")
    offloaded.write_text("offloaded", encoding="utf-8")
    stat_calls: list[Path] = []

    def fake_lstat(path: Path) -> SimpleNamespace:
        stat_calls.append(path)
        flags = SF_DATALESS if path == offloaded else 0
        return SimpleNamespace(st_flags=flags)

    assert find_dataless_files([local, offloaded], stat_function=fake_lstat) == [
        offloaded
    ]
    assert stat_calls == [local, offloaded]


def test_dataless_detection_is_portable_when_st_flags_is_absent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ordinary.py"
    path.write_text("ordinary", encoding="utf-8")

    assert (
        find_dataless_files([path], stat_function=lambda _path: SimpleNamespace()) == []
    )


def test_materialize_files_reads_complete_contents(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"a" * (1024 * 1024 + 1))
    second.write_bytes(b"second")

    materialize_files([first, second])

    assert first.read_bytes() == b"a" * (1024 * 1024 + 1)
    assert second.read_bytes() == b"second"


def test_pytest_error_is_bounded_and_repo_relative(tmp_path: Path) -> None:
    paths = [tmp_path / "tests" / f"test_{index}.py" for index in range(7)]

    message = format_pytest_error(paths, tmp_path)

    assert "macOS offloaded 7 repository file(s)" in message
    assert "tests/test_0.py" in message
    assert "tests/test_4.py" in message
    assert "test_5.py" not in message
    assert "... and 2 more" in message
    assert str(tmp_path) not in message
    assert "--hydrate" in message
