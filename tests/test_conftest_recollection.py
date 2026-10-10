"""Subdirectory conftest fixtures must survive pytest re-collecting a directory.

Under pytest 9.1, ``llm_runtime/a.py test_root.py llm_runtime/b.py`` re-collects
the ``tests`` package and builds a second ``Dir llm_runtime``; without
``tests/helpers/conftest_recollection.py`` the tests under that copy lose the
subdirectory conftest -- named fixtures error, autouse ones are skipped silently.
See that module's docstring for the pytest source path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"

_SUB_CONFTEST = """
import pytest


@pytest.fixture
def named_fixture():
    return 1


@pytest.fixture(autouse=True)
def autouse_fixture(request):
    request.node.autouse_ran = True
"""

_TEST_FIRST = "def test_first(named_fixture):\n    assert named_fixture == 1\n"
_TEST_LAST = (
    "def test_last_named(named_fixture):\n"
    "    assert named_fixture == 1\n\n\n"
    "def test_last_autouse(request):\n"
    "    assert getattr(request.node, 'autouse_ran', False)\n"
)
_HOOK_IMPORT = (
    "from tests.helpers.conftest_recollection import (  # noqa: F401\n"
    "    pytest_make_collect_report,\n"
    ")\n"
)


def _child_env() -> dict[str, str]:
    # PYTHONPATH, never cwd=, so the child is spawned with posix_spawn.
    env = {**os.environ}
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) + os.pathsep + existing if existing else str(REPO_ROOT)
    )
    return env


def _run_pytest(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args],
        capture_output=True,
        text=True,
        env=_child_env(),
        close_fds=False,
    )


def _sandwiched_tree(root: Path, *, with_hook: bool) -> list[str]:
    sub = root / "sub"
    sub.mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / "conftest.py").write_text(_HOOK_IMPORT if with_hook else "")
    (sub / "conftest.py").write_text(_SUB_CONFTEST)
    (sub / "test_first.py").write_text(_TEST_FIRST)
    (sub / "test_last.py").write_text(_TEST_LAST)
    (root / "test_root.py").write_text("def test_root():\n    pass\n")
    return [
        "-q",
        "-c",
        str(root / "pytest.ini"),
        f"--rootdir={root}",
        f"--confcutdir={root}",
        str(sub / "test_first.py"),
        str(root / "test_root.py"),
        str(sub / "test_last.py"),
    ]


def test_without_the_hook_pytest_drops_the_recollected_directory_conftest(tmp_path):
    """The planted defect: if this starts passing, pytest fixed it upstream and
    ``tests/helpers/conftest_recollection.py`` can be retired."""
    result = _run_pytest(_sandwiched_tree(tmp_path, with_hook=False))

    out = result.stdout
    assert result.returncode != 0, out
    assert "fixture 'named_fixture' not found" in out, out
    assert "test_last.py::test_last_autouse" in out, out
    assert "1 failed, 2 passed, 1 error" in out, out


def test_hook_keeps_named_and_autouse_fixtures_on_the_recollected_copy(tmp_path):
    result = _run_pytest(_sandwiched_tree(tmp_path, with_hook=True))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout, result.stdout


def test_real_sandwiched_llm_runtime_files_resolve_restore_runtime_config():
    """The exact ordering that produced 5 setup errors on this suite."""
    result = _run_pytest(
        [
            "--setup-plan",
            f"--rootdir={TESTS_DIR}",
            f"--confcutdir={TESTS_DIR}",
            str(TESTS_DIR / "llm_runtime" / "test_timeouts.py"),
            str(TESTS_DIR / "test_llms_flex.py"),
            str(TESTS_DIR / "llm_runtime" / "test_factory.py"),
        ]
    )

    out = result.stdout
    assert result.returncode == 0, out[-3000:] + result.stderr[-2000:]
    assert "not found" not in out, out[-3000:]
    assert "SETUP    F restore_runtime_config" in out, out[-3000:]
