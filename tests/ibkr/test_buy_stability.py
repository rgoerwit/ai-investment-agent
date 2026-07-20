"""Tests for the deterministic BUY stability / hysteresis gate."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.ibkr.analysis_index import _build_analysis_record_from_data
from src.ibkr.buy_stability import (
    BuyStabilityConfig,
    assess_buy_stability,
    load_recent_same_ticker_verdicts,
)
from tests.import_boundary import assert_no_offenders

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG = BuyStabilityConfig(enabled=True, lookback_days=7, margin_tally=0.5)


def _pm_block(verdict: str) -> str:
    return f"### --- START PM_BLOCK ---\nVERDICT: {verdict}\n### --- END PM_BLOCK ---"


class TestAssessBuyStability:
    def test_disabled_is_inert(self):
        cfg = BuyStabilityConfig(enabled=False)
        assert assess_buy_stability("BUY", ["HOLD"], cfg=cfg) is None

    def test_non_buy_verdict_ignored(self):
        assert assess_buy_stability("HOLD", ["DO_NOT_INITIATE"], cfg=_CFG) is None

    def test_contradicted_buy_withheld(self):
        note = assess_buy_stability("BUY", ["BUY", "HOLD"], cfg=_CFG)
        assert note is not None and "contradicted" in note.lower()

    def test_consistent_buy_history_passes(self):
        assert assess_buy_stability("BUY", ["BUY", "BUY"], cfg=_CFG) is None

    def test_no_history_does_not_withhold(self):
        # Error-handling: absent history is treated as no contradiction.
        assert assess_buy_stability("BUY", [], cfg=_CFG) is None

    def test_marginal_with_unresolved_flag_withheld(self):
        note = assess_buy_stability(
            "BUY",
            [],
            risk_tally=0.5,
            active_flags={"CYCLICAL_PEAK_WARNING"},
            cfg=_CFG,
        )
        assert note is not None and "marginal" in note.lower()

    def test_suppression_flag_counts_as_unresolved(self):
        note = assess_buy_stability(
            "BUY",
            [],
            risk_tally=0.6,
            active_flags=("MOAT_BONUS_SUPPRESSED_PEAK_TRANSIENT",),
            cfg=_CFG,
        )
        assert note is not None

    def test_marginal_without_flag_passes(self):
        assert (
            assess_buy_stability(
                "BUY", [], risk_tally=0.9, active_flags=set(), cfg=_CFG
            )
            is None
        )

    def test_flag_but_not_marginal_passes(self):
        assert (
            assess_buy_stability(
                "BUY",
                [],
                risk_tally=0.0,
                active_flags={"CYCLICAL_PEAK_WARNING"},
                cfg=_CFG,
            )
            is None
        )

    def test_malformed_active_flags_do_not_crash(self):
        assert (
            assess_buy_stability("BUY", [], risk_tally=1.0, active_flags=None, cfg=_CFG)
            is None
        )


class TestLoadRecentSameTickerVerdicts:
    _NOW = datetime(2026, 6, 19, 12, 0, 0)

    def _write(self, d, name, decision_text=None, *, raw=None, quick_mode=None):
        path = d / name
        if raw is not None:
            path.write_text(raw)
        else:
            payload = {"final_decision": {"decision": decision_text}}
            if quick_mode is not None:
                payload["run_summary"] = {"quick_mode": quick_mode}
            path.write_text(json.dumps(payload))
        return str(path)

    def test_history_records_carry_date_and_mode(self, tmp_path):
        """The SELL confirmation gate needs dated, mode-aware records —
        verdict strings alone cannot enforce spacing or exclude quick runs."""
        from src.ibkr.buy_stability import load_recent_same_ticker_history

        self._write(
            tmp_path,
            "TEST.T_20260617_100000_analysis.json",
            _pm_block("SELL"),
            quick_mode=True,
        )
        self._write(
            tmp_path,
            "TEST.T_20260618_100000_analysis.json",
            _pm_block("BUY"),
            quick_mode=False,
        )
        history = load_recent_same_ticker_history(
            "TEST.T",
            lookback_days=7,
            results_dir=str(tmp_path),
            now=self._NOW,
        )
        assert [(r.verdict, r.is_quick_mode) for r in history] == [
            ("SELL", True),
            ("BUY", False),
        ]
        # Ascending timestamp order, real datetimes
        assert history[0].analysis_dt < history[1].analysis_dt
        assert history[0].analysis_dt == datetime(2026, 6, 17, 10, 0, 0)

    def test_history_missing_run_summary_is_mode_unknown(self, tmp_path):
        """Absent/empty run_summary → mode unknown (None), never full-mode.

        Unknown mode carries no sell-confirmation authority; the legacy False
        default let pre-run_summary artifacts confirm executable sells."""
        from src.ibkr.buy_stability import load_recent_same_ticker_history

        self._write(tmp_path, "TEST.T_20260618_100000_analysis.json", _pm_block("SELL"))
        history = load_recent_same_ticker_history(
            "TEST.T", lookback_days=7, results_dir=str(tmp_path), now=self._NOW
        )
        assert len(history) == 1
        assert history[0].is_quick_mode is None

    def test_history_non_dict_run_summary_kept_as_mode_unknown(self, tmp_path):
        """A truthy non-dict run_summary must degrade to mode-unknown, not
        raise AttributeError and silently drop the verdict record."""
        from src.ibkr.buy_stability import load_recent_same_ticker_history

        self._write(
            tmp_path,
            "TEST.T_20260618_100000_analysis.json",
            raw=json.dumps(
                {
                    "final_decision": {"decision": _pm_block("SELL")},
                    "run_summary": "corrupted-string-not-a-dict",
                }
            ),
        )
        history = load_recent_same_ticker_history(
            "TEST.T", lookback_days=7, results_dir=str(tmp_path), now=self._NOW
        )
        assert len(history) == 1
        assert history[0].is_quick_mode is None
        assert history[0].verdict == "SELL"

    def test_verdict_wrapper_stays_compatible(self, tmp_path):
        """The BUY-gate view is a thin projection of the history records."""
        self._write(tmp_path, "TEST.T_20260618_100000_analysis.json", _pm_block("HOLD"))
        verdicts = load_recent_same_ticker_verdicts(
            "TEST.T", lookback_days=7, results_dir=str(tmp_path), now=self._NOW
        )
        assert verdicts == ["HOLD"]

    def test_returns_in_window_excludes_current_and_stale(self, tmp_path):
        self._write(tmp_path, "TEST.T_20260617_100000_analysis.json", _pm_block("BUY"))
        self._write(tmp_path, "TEST.T_20260618_100000_analysis.json", _pm_block("HOLD"))
        self._write(
            tmp_path, "TEST.T_20260601_100000_analysis.json", _pm_block("HOLD")
        )  # stale (>7d)
        current = self._write(
            tmp_path, "TEST.T_20260619_120000_analysis.json", _pm_block("BUY")
        )

        verdicts = load_recent_same_ticker_verdicts(
            "TEST.T",
            lookback_days=7,
            results_dir=str(tmp_path),
            now=self._NOW,
            exclude_path=current,
        )
        assert verdicts == ["BUY", "HOLD"]  # stale + current excluded

    def test_malformed_files_skipped_not_raised(self, tmp_path):
        self._write(tmp_path, "TEST.T_20260618_100000_analysis.json", _pm_block("BUY"))
        self._write(
            tmp_path, "TEST.T_20260617_100000_analysis.json", raw="{not valid json"
        )
        verdicts = load_recent_same_ticker_verdicts(
            "TEST.T", lookback_days=7, results_dir=str(tmp_path), now=self._NOW
        )
        assert verdicts == ["BUY"]

    def test_unparseable_verdict_omitted(self, tmp_path):
        self._write(tmp_path, "TEST.T_20260618_100000_analysis.json", "no verdict text")
        verdicts = load_recent_same_ticker_verdicts(
            "TEST.T", lookback_days=7, results_dir=str(tmp_path), now=self._NOW
        )
        assert verdicts == []

    def test_other_ticker_files_ignored(self, tmp_path):
        self._write(
            tmp_path, "OTHER.T_20260618_100000_analysis.json", _pm_block("HOLD")
        )
        verdicts = load_recent_same_ticker_verdicts(
            "TEST.T", lookback_days=7, results_dir=str(tmp_path), now=self._NOW
        )
        assert verdicts == []

    def test_missing_dir_returns_empty(self, tmp_path):
        verdicts = load_recent_same_ticker_verdicts(
            "TEST.T",
            lookback_days=7,
            results_dir=str(tmp_path / "does_not_exist"),
            now=self._NOW,
        )
        assert verdicts == []


class TestImportBoundary:
    def test_enabled_gate_module_imports_no_agents_package(self):
        """The default-on gate path must stay agents-free.

        Runs in a fresh interpreter (in-process sys.modules is polluted by other
        tests) and asserts importing the gate pulls in no src.agents module — and
        thus none of the heavy LangGraph/LLM surface that agents/__init__ loads.
        """
        assert_no_offenders(
            "import sys\n"
            "import src.ibkr.buy_stability  # noqa\n"
            "bad = sorted(m for m in sys.modules "
            "if m == 'src.agents' or m.startswith('src.agents.'))\n"
            "print('AGENTS:' + ','.join(bad))\n",
            sentinel="AGENTS",
            cwd=str(_REPO_ROOT),
            message="src.ibkr.buy_stability imported src.agents",
        )


class TestParserCoherenceWithAnalysisIndex:
    """The gate's prior-verdict reading must agree with the verdict the IBKR
    analysis index records for the same saved decision (both share the neutral
    parse_final_decision_scores; they differ only in final canonicalization,
    which agrees for the canonical PM verdicts)."""

    @pytest.mark.parametrize("verdict", ["BUY", "HOLD", "SELL", "DO_NOT_INITIATE"])
    def test_gate_history_matches_analysis_record_verdict(self, tmp_path, verdict):
        decision = (
            f"### --- START PM_BLOCK ---\nVERDICT: {verdict}\n### --- END PM_BLOCK ---"
        )
        data = {
            "prediction_snapshot": {"ticker": "TEST.T", "currency": "JPY"},
            "final_decision": {"decision": decision},
            "investment_analysis": {"trader_plan": ""},
        }
        record = _build_analysis_record_from_data(
            Path("TEST.T_20260601_000000_analysis.json"), data
        )
        path = tmp_path / "TEST.T_20260601_000000_analysis.json"
        path.write_text(json.dumps(data))
        gate_verdicts = load_recent_same_ticker_verdicts(
            "TEST.T",
            lookback_days=3650,
            results_dir=str(tmp_path),
            now=datetime(2026, 6, 2),
        )
        assert record is not None
        assert gate_verdicts == [record.verdict]
