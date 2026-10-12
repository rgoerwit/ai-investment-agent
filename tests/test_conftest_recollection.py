"""Subdirectory conftest fixtures must survive pytest re-collecting a directory.

Under pytest 9.1, ``llm_runtime/a.py test_root.py llm_runtime/b.py`` re-collects
the ``tests`` package and builds a second ``Dir llm_runtime``; without
``tests/helpers/conftest_recollection.py`` the tests under that copy lose the
subdirectory conftest -- named fixtures error, autouse ones are skipped silently.
See that module's docstring for the pytest source path.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import types
from pathlib import Path

from tests.helpers.conftest_recollection import _reparse_conftest

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


# --- applicability: the hook acts only where the 9.1 mechanism and signature exist ---


class _Manager:
    _pending_conftests: dict = {}

    def __init__(self):
        self.calls = []

    def parsefactories(self, node_or_obj=None, nodeid=None, *, holder=None, node=None):
        self.calls.append((holder, node))


def _collector(manager, tmp_path):
    plugin = object()
    pluginmanager = types.SimpleNamespace(
        list_name_plugin=lambda: [
            (str(tmp_path / "conftest.py"), plugin),
            (None, object()),
            (str(tmp_path / "other" / "conftest.py"), object()),
        ]
    )
    return types.SimpleNamespace(
        path=tmp_path,
        session=types.SimpleNamespace(_fixturemanager=manager),
        config=types.SimpleNamespace(pluginmanager=pluginmanager),
    ), plugin


def test_reparse_applies_when_mechanism_and_signature_exist(tmp_path):
    manager = _Manager()
    collector, plugin = _collector(manager, tmp_path)

    assert _reparse_conftest(collector) is True
    assert manager.calls == [(plugin, collector)]


def test_reparse_is_a_noop_without_the_deferral_mechanism(tmp_path):
    class Older:
        def __init__(self):
            self.calls = []

        def parsefactories(self, *, holder=None, node=None):
            self.calls.append((holder, node))

    manager = Older()
    collector, _ = _collector(manager, tmp_path)

    assert _reparse_conftest(collector) is False
    assert manager.calls == []


def test_reparse_is_a_noop_without_the_keyword_signature(tmp_path):
    class Legacy(_Manager):
        def parsefactories(self, node_or_obj, nodeid=None):
            raise AssertionError("must not be called")

    collector, _ = _collector(Legacy(), tmp_path)

    assert _reparse_conftest(collector) is False


def test_reparse_is_a_noop_when_the_signature_cannot_be_inspected(
    tmp_path, monkeypatch
):
    manager = _Manager()
    collector, _ = _collector(manager, tmp_path)

    def refuse(_obj):
        raise ValueError("no signature")

    monkeypatch.setattr(inspect, "signature", refuse)

    assert _reparse_conftest(collector) is False
    assert manager.calls == []


def test_installed_pytest_manager_matches_the_detected_mechanism(request, tmp_path):
    """On the installed pytest the real manager is judged as its source says."""
    manager = request.session._fixturemanager
    expected = hasattr(manager, "_pending_conftests") and {"holder", "node"} <= set(
        inspect.signature(manager.parsefactories).parameters
    )
    collector, _ = _collector(manager, tmp_path)
    collector.config.pluginmanager.list_name_plugin = lambda: []

    assert _reparse_conftest(collector) is expected
