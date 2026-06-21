"""Chart extractor modules must be importable first, in a fresh interpreter.

Regression for the pm_block circular import: pm_block →
src.agents.pm_verdict_metadata → agents/__init__ → decision_nodes → pm_block.
The full test suite masks import-order bugs because conftest/other tests load
src.agents before tests/charts collects, so each case runs in a clean
subprocess interpreter.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "src.charts.extractors.pm_block",
        "src.charts.extractors.data_block",
        "src.charts.extractors.valuation",
        "src.charts.chart_node",
    ],
)
def test_chart_module_imports_cleanly_first(module: str):
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=60,
        # Keep fds open so CPython uses posix_spawn instead of fork()+exec(): a
        # fork() in this gRPC/Network.framework-loaded test process SIGSEGVs in
        # Apple's atfork handler on macOS. See CLAUDE.md (macOS-Specific Issues).
        close_fds=False,
    )
    assert (
        proc.returncode == 0
    ), f"importing {module} first failed (circular import?):\n{proc.stderr[-800:]}"
