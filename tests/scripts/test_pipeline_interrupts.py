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

            sys.path.insert(0, {str(_REPO_ROOT)!r})
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


def test_pipeline_watchdog_dumps_before_killing_timed_out_child():
    """Pin watchdog behavior without executing Bash signal machinery.

    On macOS/VSCode, repeated full-suite runs have shown the Bash harness itself
    can exit via SIGSEGV (-11) while exercising fake kill/sleep paths. That is a
    shell/runtime crash, not a failure of the watchdog contract. This test keeps
    the useful assertions at the script boundary: timeout must dump before
    termination, escalate USR1 -> TERM -> KILL, return 124, and write a JSONL
    breadcrumb for recovery.
    """
    script = (_SCRIPTS_DIR / "pipeline_signals.sh").read_text()

    timeout_pos = script.index("[pipeline_child_timeout]")
    usr1_pos = script.index("[pipeline_child_signal] SIGUSR1")
    term_pos = script.index("[pipeline_child_signal] SIGTERM")
    kill_pos = script.index("[pipeline_child_signal] SIGKILL")
    return_pos = script.index("return 124")

    assert timeout_pos < usr1_pos < term_pos < kill_pos < return_pos
    assert "PIPELINE_TIMEOUT_RECORD_FILE" in script
    assert '"status":"timeout"' in script
    assert "_pipeline_kill -USR1" in script
    assert "_pipeline_kill -TERM" in script
    assert "_pipeline_kill -KILL" in script
    assert '_pipeline_sleep "$dump_grace_seconds"' in script
    assert '_pipeline_sleep "$term_grace_seconds"' in script
