from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def test_find_gems_pool_interrupt_exits_promptly_without_active_children(tmp_path):
    runner = tmp_path / "find_gems_interrupt_runner.py"
    runner.write_text(
        textwrap.dedent(
            f"""
            import multiprocessing as mp
            import os
            import signal
            import sys
            import threading
            import time

            sys.path.insert(0, {str(_SCRIPTS_DIR)!r})
            import find_gems

            def blocking_worker(task):
                row, *_ = task
                time.sleep(30)
                return {{"YF_Ticker": row["YF_Ticker"]}}

            def trigger_interrupt():
                time.sleep(0.5)
                os.kill(os.getpid(), signal.SIGINT)

            find_gems._passes_filters = lambda row, **kwargs: True
            threading.Thread(target=trigger_interrupt, daemon=True).start()

            passing, enriched = find_gems._collect_enrichment_results(
                [{{"YF_Ticker": "7203.T"}}, {{"YF_Ticker": "6758.T"}}],
                fx_rates={{"USD": 1.0}},
                criteria=find_gems.ScreenCriteria(),
                workers=2,
                worker_fn=blocking_worker,
            )
            print(f"PASSING={{len(passing)}}", flush=True)
            print(f"ENRICHED={{len(enriched)}}", flush=True)
            print(f"ACTIVE_CHILDREN={{len(mp.active_children())}}", flush=True)
            """
        )
    )

    completed = subprocess.run(
        [sys.executable, str(runner)],
        capture_output=True,
        text=True,
        close_fds=False,
        timeout=5,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0
    assert "Interrupted! Returning partial results..." in output
    assert "ACTIVE_CHILDREN=0" in output
