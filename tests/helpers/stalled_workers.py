"""Release and join a stalled blocking worker before the test's patches are undone.

``run_blocking_call`` workers are daemon threads named ``blocking:<label>``. A test
that parks one on a ``threading.Event`` must release and join it in a ``finally``:
if an assertion fails first, a worker that outlives the test's monkeypatch teardown
runs against restored (real) modules. ``stalled_worker`` does that on every path.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

JOIN_TIMEOUT_SECONDS = 2.0


def join_workers(label: str, timeout: float | None = None) -> list[threading.Thread]:
    """Join every ``blocking:<label>`` worker; return those still alive."""
    workers = [
        t for t in threading.enumerate() if t.name.startswith(f"blocking:{label}")
    ]
    for worker in workers:
        worker.join(JOIN_TIMEOUT_SECONDS if timeout is None else timeout)
    return [t for t in workers if t.is_alive()]


@contextmanager
def stalled_worker(label: str) -> Iterator[threading.Event]:
    """Yield the event a fake blocks on; on exit release it and join the workers.

    Raises ``AssertionError`` when a worker survives the join and no other error is
    already propagating.
    """
    release = threading.Event()
    try:
        yield release
    finally:
        release.set()
        survivors = join_workers(label)
    assert survivors == [], f"stalled workers outlived their release: {survivors}"
