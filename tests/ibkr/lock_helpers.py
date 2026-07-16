from __future__ import annotations

import sys
import time
from pathlib import Path

from src.ibkr.analysis_index import _analysis_index_lock


def hold_analysis_index_lock_until_signal(
    results_dir: str, hold_seconds: float, ready_path: str
) -> None:
    """Acquire the index lock, signal readiness via ``ready_path``, then hold.

    Readiness is a marker file (not a ``multiprocessing.Event``) so this can run
    under a ``posix_spawn`` subprocess instead of a ``multiprocessing`` spawn
    Process. The latter calls ``fork()``, which SIGSEGVs in Apple's
    Network.framework atfork handler on macOS once the gRPC stack is loaded; see
    CLAUDE.md (macOS-Specific Issues). ``posix_spawn`` runs no atfork handlers.
    """
    with _analysis_index_lock(Path(results_dir)):
        Path(ready_path).write_text("ready")
        time.sleep(hold_seconds)


if __name__ == "__main__":
    # CLI entrypoint: python -m tests.ibkr.lock_helpers <results_dir> <secs> <ready>
    hold_analysis_index_lock_until_signal(sys.argv[1], float(sys.argv[2]), sys.argv[3])
