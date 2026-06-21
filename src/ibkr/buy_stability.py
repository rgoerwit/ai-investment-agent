"""Deterministic BUY stability / hysteresis gate (advisory).

Addresses the verdict-noise defect: a fresh BUY whose tally hovers at the
zone threshold can flip BUY<->HOLD<->DNI run-to-run on the same data. This
module decides whether a BUY should be *withheld* (advised down to REVIEW)
because recent same-ticker analyses disagreed, or because it is marginal with
an unresolved peak/transient flag.

It is intentionally pure + side-effect free (except the file-scanning history
helper) so it can be unit-tested in isolation and wired as an advisory at the
action layer WITHOUT rewriting the Portfolio Manager's narrative verdict.

Lives under src.ibkr (its only consumer is the off-watchlist opportunity
finder) and depends only on the neutral src.pm_decision_parser + the
dependency-free validators flag set — so the enabled gate path never imports
src.agents (and its heavy LangGraph/LLM surface). A boundary test enforces this.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import structlog

from src.pm_decision_parser import canonicalize_pm_verdict, parse_final_decision_scores

# Canonical set lives in the dependency-free validators module so lightweight
# callers (e.g. IBKR indexing) need not import the agents package. Re-exported
# here for backward compatibility.
from src.validators.quality_flags import PEAK_OR_TRANSIENT_FLAGS

logger = structlog.get_logger(__name__)

_FILENAME_DATE_RE = re.compile(r"_(\d{8})_(\d{6})_analysis\.json$")

__all__ = [
    "PEAK_OR_TRANSIENT_FLAGS",
    "BuyStabilityConfig",
    "assess_buy_stability",
    "load_recent_same_ticker_verdicts",
]


@dataclass(frozen=True, slots=True)
class BuyStabilityConfig:
    """Run-scoped knobs for the BUY stability gate."""

    enabled: bool = False
    lookback_days: int = 7
    margin_tally: float = 0.5

    @classmethod
    def from_config(cls, config: Any) -> BuyStabilityConfig:
        """Build from the global Settings object (env/.env/default chain)."""
        return cls(
            enabled=bool(getattr(config, "buy_stability_enabled", False)),
            lookback_days=int(getattr(config, "buy_stability_lookback_days", 7)),
            margin_tally=float(getattr(config, "buy_stability_margin_tally", 0.5)),
        )


def _is_buy(verdict: str | None) -> bool:
    return bool(verdict) and verdict.strip().upper().replace(" ", "_") == "BUY"


def assess_buy_stability(
    verdict: str | None,
    prior_verdicts: list[str],
    *,
    risk_tally: float | None = None,
    active_flags: object = (),
    cfg: BuyStabilityConfig,
) -> str | None:
    """Return a withhold reason if a BUY is unstable/marginal, else None.

    - ``prior_verdicts``: recent same-ticker verdicts within the lookback window
      (most-recent ordering is not required).
    - ``risk_tally`` / ``active_flags``: optional; when supplied, a marginal BUY
      (tally >= margin) carrying an unresolved peak/transient flag is also
      withheld. When unavailable (e.g. the analysis index does not persist the
      tally) the reproducibility check still applies.

    Never mutates inputs. Returns an advisory string; the caller decides how to
    act (typically: do not emit the BUY recommendation; surface for REVIEW).
    """
    if not cfg.enabled or not _is_buy(verdict):
        return None

    # Reproducibility: a BUY contradicted by a recent same-ticker run is unstable.
    contradictions = [v for v in prior_verdicts if v and not _is_buy(v)]
    if contradictions:
        return (
            f"BUY contradicted by recent same-ticker verdict(s) "
            f"{sorted(set(contradictions))} within {cfg.lookback_days}d — withhold for REVIEW"
        )

    # Quality-margin: marginal BUY with an unresolved peak/transient flag should
    # not initiate on a single run.
    unresolved = bool(set(_as_flag_set(active_flags)) & PEAK_OR_TRANSIENT_FLAGS)
    marginal = risk_tally is not None and risk_tally >= cfg.margin_tally
    if unresolved and marginal:
        return (
            "BUY at marginal risk tally with an unresolved peak/transient flag — "
            "withhold pending stability"
        )

    return None


def _as_flag_set(active_flags: object) -> set[str]:
    """Coerce a flag container (set/list/tuple of str) to a set[str]."""
    if isinstance(active_flags, str):
        return {active_flags}
    try:
        return {str(f) for f in active_flags}  # type: ignore[union-attr]
    except TypeError:
        return set()


def load_recent_same_ticker_verdicts(
    ticker: str,
    *,
    lookback_days: int,
    results_dir: str,
    now: datetime | None = None,
    exclude_path: str | None = None,
) -> list[str]:
    """Return canonical verdicts for same-ticker analyses within the lookback.

    Scans ``{results_dir}/{ticker}_YYYYMMDD_HHMMSS_analysis.json``. Malformed or
    unreadable files are skipped (logged at debug), never raised. The current
    analysis (``exclude_path``) is excluded so a BUY is not compared to itself.

    Prior verdicts are parsed with the same neutral final-decision parser the
    IBKR analysis index uses to produce ``AnalysisRecord.verdict``, so the gate's
    history reading is coherent with the verdicts it gates against.
    """
    if not ticker or lookback_days <= 0 or not results_dir:
        return []
    now = now or datetime.now()
    cutoff = now - timedelta(days=lookback_days)
    exclude_abs = os.path.abspath(exclude_path) if exclude_path else None

    verdicts: list[str] = []
    pattern = os.path.join(results_dir, f"{ticker}_*_analysis.json")
    for path in sorted(glob.glob(pattern)):
        if exclude_abs and os.path.abspath(path) == exclude_abs:
            continue
        match = _FILENAME_DATE_RE.search(os.path.basename(path))
        if not match:
            continue
        try:
            dt = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            continue
        if dt < cutoff:
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
            decision = (data.get("final_decision") or {}).get("decision", "")
            verdict = canonicalize_pm_verdict(
                parse_final_decision_scores(str(decision)).get("verdict")
            )
        except (OSError, json.JSONDecodeError, ValueError, AttributeError) as exc:
            # debug-level raw error= is permitted by the logging standard; log
            # only the filename, not the full local path.
            logger.debug(
                "buy_stability_history_skip",
                file=os.path.basename(path),
                error=str(exc),
            )
            continue
        if verdict and verdict != "UNPARSEABLE":
            verdicts.append(verdict)
    return verdicts
