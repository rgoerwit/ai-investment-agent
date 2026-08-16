"""Shared guards for the advanced test package."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _retrospective_memo_is_never_the_real_one(monkeypatch, tmp_path):
    """Point the evaluation memo at ``tmp_path`` for every test in this package.

    ``run_retrospective`` persists a memo of which snapshots it has already
    priced. Left at its production default that is ``runtime/`` inside the
    repository, so a test run would both dirty the working tree and — worse —
    leak state between tests: the second test to price a given snapshot identity
    would find it memoized and silently skip the work it was written to assert.
    """
    monkeypatch.setattr(
        "src.retrospective.DEFAULT_EVALUATION_MEMO_PATH",
        tmp_path / "retrospective_evaluations.json",
        raising=False,
    )
