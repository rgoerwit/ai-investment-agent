"""Cross-shape readers for report rendering and retrospective scoring.

The memo, the source-confidence builder, and the quality judge all need to
read from two structurally different dicts:

- **Runtime AgentState** (during graph execution) — top-level keys like
  ``final_trade_decision``, ``investment_plan``, ``fundamentals_report``.
- **Saved analysis JSON** (``results/*_analysis.json``) — fields are nested
  under ``final_decision.decision``, ``investment_analysis.investment_plan``,
  ``reports.fundamentals_report``.

This module is the single seam where that dual-shape lookup lives. It mirrors
the pattern established by ``src.agents.support.get_bear_history`` (Tranche 1)
but is kept out of the agents package — ``support.py`` is already 612 lines,
and these helpers are report-layer concerns.

All helpers return ``""`` for missing fields rather than raising; the
downstream renderers degrade to "UNAVAILABLE" placeholders.
"""

from __future__ import annotations

from typing import Any

from src.claim_policy import RAW_FINANCIAL_METRICS_INPUT
from src.tooling.structured_ingress import render_structured_ingress_payload


def _safe(source: Any) -> dict:
    """Coerce non-dict inputs to an empty dict to keep callers branch-free."""
    return source if isinstance(source, dict) else {}


def get_pm_output(source: Any) -> str:
    """Return the Portfolio Manager's final verdict text from either shape.

    Resolution order:

    1. Runtime ``final_trade_decision`` — populated during graph execution.
    2. ``reports.portfolio_manager`` — speculative future shape if persistence
       ever moves PM output into ``reports``.
    3. ``final_decision.decision`` — the current saved-JSON shape written by
       ``src.persistence.save_results_to_file`` (line 371).
    """
    s = _safe(source)
    return (
        s.get("final_trade_decision")
        or (s.get("reports") or {}).get("portfolio_manager")
        or (s.get("final_decision") or {}).get("decision")
        or ""
    )


def get_investment_plan(source: Any) -> str:
    """Return Research Manager's investment plan from either shape.

    Resolution order: runtime ``investment_plan`` → saved JSON
    ``investment_analysis.investment_plan`` (``persistence.py:354``).
    """
    s = _safe(source)
    return (
        s.get("investment_plan")
        or (s.get("investment_analysis") or {}).get("investment_plan")
        or ""
    )


def get_fundamentals_report(source: Any) -> str:
    """Return the Fundamentals Analyst report from either shape."""
    s = _safe(source)
    return (
        s.get("fundamentals_report")
        or (s.get("reports") or {}).get("fundamentals_report")
        or ""
    )


def get_raw_fundamentals_data(source: Any) -> str:
    """Return code-owned raw metrics JSON, with legacy artifacts as fallback."""
    s = _safe(source)
    return (
        render_structured_ingress_payload(s, RAW_FINANCIAL_METRICS_INPUT)
        or s.get("raw_fundamentals_data")
        or (s.get("source_artifacts") or {}).get("raw_fundamentals_data")
        or ""
    )


def get_valuation_params(source: Any) -> str:
    """Return the Valuation Calculator's structured-block output from either shape."""
    s = _safe(source)
    return (
        s.get("valuation_params")
        or (s.get("reports") or {}).get("valuation_params")
        or ""
    )


def get_auditor_report(source: Any) -> str:
    """Return the Forensic Auditor output from either shape."""
    s = _safe(source)
    return (
        s.get("auditor_report") or (s.get("reports") or {}).get("auditor_report") or ""
    )


def get_consultant_review(source: Any) -> str:
    """Return the External Consultant review from either shape."""
    s = _safe(source)
    return (
        s.get("consultant_review")
        or (s.get("reports") or {}).get("consultant_review")
        or ""
    )


def get_apac_regional_report(source: Any) -> str:
    """Return the APAC Regional Specialist output from either shape."""
    s = _safe(source)
    return (
        s.get("apac_regional_report")
        or (s.get("reports") or {}).get("apac_regional_report")
        or ""
    )


def get_red_flags(source: Any) -> list[dict[str, Any]]:
    """Return persisted/runtime red flags, if available."""
    s = _safe(source)
    flags = s.get("red_flags")
    return list(flags) if isinstance(flags, list) else []


def get_effective_red_flags(source: Any) -> list[dict[str, Any]]:
    """Return red flags after deterministic report-stage reconciliation.

    No reconciliation is applied today. The OCF period-mismatch suppression that
    used to live here was retired: it could only ever *remove* a risk flag, its
    comparability contract was weaker than the flag-raising path it mirrored
    (bare floats, so no period/currency/scope guard ran), and it never once fired
    across the persisted artifact history. Kept as the named seam any future
    report-stage reconciliation should hang from, so the nine call sites do not
    have to move again.
    """
    return get_red_flags(_safe(source))
