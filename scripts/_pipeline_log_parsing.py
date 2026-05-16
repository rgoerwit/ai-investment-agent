"""Shared parser for pipeline ticker logs.

Reads a `scratch/<TICKER>-LOG-<DATE>_<mode>.txt` file produced by
`pipeline_signals.sh::run_tracked_child` and returns a structured dict
with counts of network / timeout / breaker events plus elapsed seconds.

This is the common backend for:
- `scripts/extract_run_health.py` (per-run badge string)
- `scripts/pipeline_batch_summary.py` (end-of-batch warning roll-up)

Design notes:
- Parses *line-oriented*. Each event lands on its own line and we count
  occurrences of specific structlog event names. This is cheaper than
  parsing full structlog records and resilient to truncation.
- Counting is *defensive*: a malformed or partially-written log returns
  whatever it counted; never raises.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LogStats:
    """Per-run signal counts. Zero-defaulted, JSON-serializable."""

    dns_failures: int = 0
    connect_errors: int = 0
    timeouts: int = 0  # hard_timeout_exceeded events
    network_breaker_opened: int = 0
    llm_circuit_opened: int = 0
    pending_task_dumps: int = 0
    pipeline_child_timeout: bool = False
    exit_status: int | None = None
    elapsed_seconds: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def is_clean(self) -> bool:
        """A run is 'clean' if nothing notable happened. Used by callers
        to decide whether to emit anything at all."""
        return (
            self.dns_failures == 0
            and self.connect_errors == 0
            and self.timeouts == 0
            and self.network_breaker_opened == 0
            and self.llm_circuit_opened == 0
            and self.pending_task_dumps == 0
            and not self.pipeline_child_timeout
            and (self.exit_status is None or self.exit_status == 0)
        )


# Regexes operate on substrings, not full JSON parses — robust to formatting.
_RE_DNS = re.compile(r"failure_kind='dns_resolution'")
_RE_CONNECT = re.compile(r"failure_kind='connect_error'")
_RE_HARD_TIMEOUT = re.compile(r"event='hard_timeout_exceeded'")
_RE_NETWORK_BREAKER_OPEN = re.compile(r"event='network_breaker_opened'")
_RE_LLM_CIRCUIT_OPEN = re.compile(r"event='llm_circuit_opened'")
_RE_PENDING_DUMP = re.compile(r"event='pending_tasks_dump'")
_RE_CHILD_TIMEOUT = re.compile(r"\[pipeline_child_timeout\]")
_RE_CHILD_EXIT = re.compile(
    r"\[pipeline_child_exit\] status=(?P<status>-?\d+) elapsed=(?P<elapsed>\d+)s"
)


def parse_log(path: Path | str) -> LogStats:
    """Parse one pipeline ticker log file. Returns LogStats; never raises."""
    p = Path(path)
    dns = connect = timeouts = nbo = lco = pdumps = 0
    child_timeout = False
    exit_status: int | None = None
    elapsed: float | None = None

    try:
        # Iterate line-by-line so a 50MB log doesn't materialize in memory.
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if _RE_DNS.search(line):
                    dns += 1
                if _RE_CONNECT.search(line):
                    connect += 1
                if _RE_HARD_TIMEOUT.search(line):
                    timeouts += 1
                if _RE_NETWORK_BREAKER_OPEN.search(line):
                    nbo += 1
                if _RE_LLM_CIRCUIT_OPEN.search(line):
                    lco += 1
                if _RE_PENDING_DUMP.search(line):
                    pdumps += 1
                if _RE_CHILD_TIMEOUT.search(line):
                    child_timeout = True
                m = _RE_CHILD_EXIT.search(line)
                if m:
                    try:
                        exit_status = int(m.group("status"))
                        elapsed = float(m.group("elapsed"))
                    except (ValueError, TypeError):
                        pass
    except FileNotFoundError:
        return LogStats()
    except OSError:
        return LogStats()

    return LogStats(
        dns_failures=dns,
        connect_errors=connect,
        timeouts=timeouts,
        network_breaker_opened=nbo,
        llm_circuit_opened=lco,
        pending_task_dumps=pdumps,
        pipeline_child_timeout=child_timeout,
        exit_status=exit_status,
        elapsed_seconds=elapsed,
    )


# ---- badge formatting -----------------------------------------------------

# Per-key thresholds — only surface counts above these. Single DNS failure
# during a 5-minute run isn't news; ten of them is. One hard-timeout is
# expected on the quick-mode slow tail; two is worth a glance.
_BADGE_THRESHOLDS = {
    "dns": 5,
    "connect": 3,
    "timeout": 2,
}


def format_badge(stats: LogStats) -> str:
    """Render a compact badge string for the pipeline `[OK]/[FAIL]` line.

    Returns the empty string when nothing is worth surfacing. Format:
        net: dns=14 timeout=2 breaker_trip=1 elapsed=358s
    """
    parts: list[str] = []
    if stats.dns_failures >= _BADGE_THRESHOLDS["dns"]:
        parts.append(f"dns={stats.dns_failures}")
    if stats.connect_errors >= _BADGE_THRESHOLDS["connect"]:
        parts.append(f"connect={stats.connect_errors}")
    if stats.timeouts >= _BADGE_THRESHOLDS["timeout"]:
        parts.append(f"timeout={stats.timeouts}")
    # Breaker trips are always notable — they're the "deeper trouble" signal.
    if stats.network_breaker_opened > 0:
        parts.append(f"breaker_trip={stats.network_breaker_opened}")
    if stats.llm_circuit_opened > 0:
        parts.append(f"circuit_open={stats.llm_circuit_opened}")
    if stats.pipeline_child_timeout:
        parts.append("child_timeout=1")

    if not parts:
        return ""

    # Append elapsed only if we surfaced anything else — it's context, not
    # a signal on its own.
    if stats.elapsed_seconds is not None:
        parts.append(f"elapsed={int(stats.elapsed_seconds)}s")

    return "net: " + " ".join(parts)
