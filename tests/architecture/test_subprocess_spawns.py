"""Every child process under ``src/`` starts in ``src/subprocesses.py``.

That module is the one place the ``posix_spawn`` configuration lives (an
absolute executable, ``close_fds=False``, no ``cwd``) along with the bounded
kill-and-reap cleanup. A spawn written anywhere else forks by default, and in
the threaded analysis process ``fork()`` runs Network.framework's atfork
handler, which can crash the child -- the pdftotext and git children both did
until October 2026. This scan keeps a new call site from bypassing the helper.

AST, not text: a comment explaining ``fork()`` must not trip it. Module aliases
(``import subprocess as sp``) and imported names (``from os import fork``) are
resolved, so a spelling cannot hide the call.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
OWNER = "subprocesses.py"

SPAWNING: dict[str, frozenset[str]] = {
    "subprocess": frozenset(
        {
            "run",
            "Popen",
            "call",
            "check_call",
            "check_output",
            "getoutput",
            "getstatusoutput",
        }
    ),
    "asyncio": frozenset({"create_subprocess_exec", "create_subprocess_shell"}),
    "os": frozenset(
        {
            "fork",
            "forkpty",
            "system",
            "popen",
            "posix_spawn",
            "posix_spawnp",
            *(
                f"spawn{suffix}"
                for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")
            ),
            *(
                f"exec{suffix}"
                for suffix in ("l", "le", "lp", "lpe", "v", "ve", "vp", "vpe")
            ),
        }
    ),
    "multiprocessing": frozenset({"Process", "Pool", "get_context"}),
    "concurrent.futures": frozenset({"ProcessPoolExecutor"}),
    "anyio": frozenset({"open_process", "run_process"}),
}
# Event-loop methods: distinctive enough to flag on any receiver.
LOOP_METHODS = frozenset({"subprocess_exec", "subprocess_shell"})


def _spawning_module(module: str) -> str | None:
    for owner in SPAWNING:
        if module == owner or module.startswith(owner + "."):
            return owner
    return None


def _dotted(node: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value, aliases)
        return None if base is None else f"{base}.{node.attr}"
    return None


def spawn_sites(source: str) -> list[tuple[str, int]]:
    """Every spawning API ``source`` names, with its line."""
    tree = ast.parse(source)
    modules: dict[str, str] = {}
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                modules[bound] = alias.name if alias.asname else bound
        elif isinstance(node, ast.ImportFrom) and node.module:
            owner = _spawning_module(node.module)
            for alias in node.names:
                bound = alias.asname or alias.name
                if owner is not None and alias.name in SPAWNING[owner]:
                    imported[bound] = f"{node.module}.{alias.name}"
                else:
                    modules[bound] = f"{node.module}.{alias.name}"

    sites: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in imported:
            sites.append((imported[node.id], node.lineno))
        elif isinstance(node, ast.Attribute):
            if node.attr in LOOP_METHODS:
                sites.append((node.attr, node.lineno))
                continue
            base = _dotted(node.value, modules)
            owner = None if base is None else _spawning_module(base)
            if owner is not None and node.attr in SPAWNING[owner]:
                sites.append((f"{base}.{node.attr}", node.lineno))
    return sites


def test_only_the_spawn_helper_starts_a_child_process():
    offenders = [
        f"{path.relative_to(SRC)}:{line}: {name}"
        for path in sorted(SRC.rglob("*.py"))
        if path.relative_to(SRC).as_posix() != OWNER
        for name, line in spawn_sites(path.read_text(encoding="utf-8"))
    ]
    assert offenders == [], (
        "Start child processes through src/subprocesses.py (posix_spawn, "
        "bounded cleanup):\n" + "\n".join(offenders)
    )


def test_the_owner_itself_is_seen_by_the_scan():
    """The exemption is not hiding a scan that finds nothing anywhere."""
    names = {name for name, _ in spawn_sites((SRC / OWNER).read_text())}
    assert names == {"asyncio.create_subprocess_exec", "subprocess.run"}


@pytest.mark.parametrize(
    "source",
    [
        "import subprocess\nsubprocess.run(['git'])",
        "import subprocess as sp\nsp.Popen(['git'])",
        "from subprocess import check_output\ncheck_output(['git'])",
        "from subprocess import run as go\ngo(['git'])",
        "import asyncio\nasyncio.create_subprocess_exec('/bin/ls')",
        "import asyncio\nasyncio.subprocess.create_subprocess_exec('/bin/ls')",
        "from asyncio import create_subprocess_shell\ncreate_subprocess_shell('ls')",
        "loop.subprocess_exec(factory, '/bin/ls')",
        "import os\nos.fork()",
        "import os\nos.system('ls')",
        "from os import posix_spawn\nposix_spawn('/bin/ls', [], {})",
        "import multiprocessing\nmultiprocessing.Process(target=f)",
        "from multiprocessing import Pool\nPool(2)",
        "import concurrent.futures\nconcurrent.futures.ProcessPoolExecutor()",
        "from concurrent.futures import ProcessPoolExecutor\nProcessPoolExecutor()",
        "import anyio\nanyio.open_process(['ls'])",
    ],
)
def test_a_planted_spawn_is_caught(source):
    assert spawn_sites(source) != []


@pytest.mark.parametrize(
    "source",
    [
        "# subprocess.run(['git']) and os.fork() are retired here\nx = 1",
        "import subprocess\nraise subprocess.CalledProcessError(1, 'git')",
        "import asyncio\npipe = asyncio.subprocess.PIPE",
        "import os\nos.path.join('a', 'b')",
        "runner.run(job)",
        "from src.subprocesses import run_program\nrun_program('/bin/ls', input=b'', timeout=1)",
        "from concurrent.futures import ThreadPoolExecutor\nThreadPoolExecutor()",
    ],
)
def test_harmless_spellings_are_not_flagged(source):
    assert spawn_sites(source) == []
