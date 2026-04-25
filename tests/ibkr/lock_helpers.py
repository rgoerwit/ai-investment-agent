from __future__ import annotations

import time
from pathlib import Path

from src.ibkr.analysis_index import _analysis_index_lock


def hold_analysis_index_lock(results_dir: str, hold_seconds: float, ready) -> None:
    """Helper process that holds the index lock long enough to test blocking."""
    with _analysis_index_lock(Path(results_dir)):
        ready.set()
        time.sleep(hold_seconds)
