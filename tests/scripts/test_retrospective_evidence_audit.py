"""The shadow report must not present eligibility as production.

`--dispositions` computes offline what the corpus *could* produce. The danger is
stage conflation: "eligible for pricing" is upstream of dedup, the memo, the
per-run budget, and the excess-return trigger, and a priced snapshot that does
not trigger produces nothing at all. An earlier version reported 6,995 eligible
snapshots as "~6,981 review-only records" — an overstatement of roughly 2.3x
against the measured 42.7% trigger rate, and exactly the number someone would
use to justify a sweep.

The denominator must also be auditable. Two runs reported 7,952 and 4,732
identities and both were correct; only one had archive directories configured.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import retrospective_evidence_audit as audit  # noqa: E402


def _artifact(tmp_path: Path, name: str, **snapshot_fields) -> Path:
    snapshot = {
        "ticker": "T.T",
        "analysis_date": "2026-01-01",
        "verdict": "BUY",
        "bear_risks_excerpt": "1. Cyclical exposure at a peak.",
    }
    snapshot.update(snapshot_fields)
    path = tmp_path / name
    path.write_text(json.dumps({"prediction_snapshot": snapshot}))
    return path


class TestTheReportDistinguishesItsStages:
    def test_eligibility_is_never_called_production(self, tmp_path, capsys):
        _artifact(tmp_path, "T.T_20260101_000000_analysis.json")
        assert audit._dispositions((tmp_path,), None) == 0
        out = capsys.readouterr().out
        assert "eligible for pricing" in out
        assert "would be priced" not in out, (
            "eligibility is upstream of dedup, memo, budget and the trigger"
        )
        assert "unless its excess return clears" in out

    def test_a_tender_record_is_conditional_on_triggering(self, tmp_path, capsys):
        _artifact(
            tmp_path,
            "T.T_20260101_000000_analysis.json",
            m_and_a_status="ACTIVE_TENDER",
        )
        audit._dispositions((tmp_path,), None)
        out = capsys.readouterr().out
        assert "*if*" in out and "they trigger" in out, (
            "a tender produces a record only after clearing the threshold"
        )
        assert "never injectable" in out

    def test_the_driver_table_is_labelled_hypothetical(self, tmp_path, capsys):
        """Corrected twice. It first read "if every priced outcome resolved to
        this driver" — presenting eligible candidates as priced. The second
        version said "among comparisons that TRIGGER", which was equally untrue:
        the counts are of eligible candidates, and which ones trigger cannot be
        known without pricing them."""
        _artifact(tmp_path, "T.T_20260101_000000_analysis.json")
        audit._dispositions((tmp_path,), None)
        out = capsys.readouterr().out
        assert "HYPOTHETICAL" in out
        assert "among comparisons that TRIGGER" not in out
        assert "not of triggered comparisons" in out


class TestTheDenominatorIsAuditable:
    def test_scope_and_counts_are_printed(self, tmp_path, capsys):
        _artifact(tmp_path, "T.T_20260101_000000_analysis.json")
        (tmp_path / "broken_analysis.json").write_text("{not json")
        (tmp_path / "X.T_20260101_000000_analysis.json").write_text(json.dumps({}))
        audit._dispositions((tmp_path, tmp_path / "absent"), None)
        out = capsys.readouterr().out
        assert "sources scanned" in out
        assert "CONFIGURED BUT MISSING" in out, (
            "a mistyped archive path silently halves the corpus otherwise"
        )
        assert "malformed 1" in out
        assert "no snapshot 1" in out

    def test_the_totals_reconcile(self, tmp_path, capsys):
        for index in range(3):
            _artifact(tmp_path, f"T{index}.T_2026010{index}_000000_analysis.json")
        _artifact(tmp_path, "N.T_20260104_000000_analysis.json", bear_risks_excerpt="")
        audit._dispositions((tmp_path,), None)
        out = capsys.readouterr().out
        total = int(
            out.split("snapshots (by identity, live tree preferred):")[1].split()[0]
        )
        buckets = sum(
            int(line.split("(")[0].strip())
            for line in out.splitlines()
            if line.startswith("  ") and "%)" in line and "carry a recorded" not in line
        )
        assert buckets == total, f"{buckets} bucketed against {total} scanned"


class TestTheCeilingIsExactNotEstimated:
    def test_a_snapshot_with_no_regime_can_never_be_injectable(self, tmp_path, capsys):
        _artifact(tmp_path, "T.T_20260101_000000_analysis.json")
        audit._dispositions((tmp_path,), None)
        out = capsys.readouterr().out
        assert "CEILING" in out
        ceiling_line = [
            ln for ln in out.splitlines() if "carry a recorded regime" in ln
        ]
        assert ceiling_line and ceiling_line[0].strip().startswith("0 "), (
            "without a regime the contextual path is unreachable at any price"
        )

    def test_a_stable_recorded_regime_raises_the_ceiling(self, tmp_path, capsys):
        """The only injectable-capable branch, and nothing exercised it.

        Every other test asserts the ceiling at zero — which a counter that never
        increments would also satisfy. This stubs the one input the corpus cannot
        supply (a comparable regime; no snapshot carries the macro fingerprint)
        and asserts the count moves.
        """
        from src.retrospective import CachedRegimeDelta

        _artifact(
            tmp_path,
            "T.T_20260101_000000_analysis.json",
            regime_at_decision={"risk_appetite": "RISK_ON", "shock_type": "NONE"},
        )
        import src.retrospective as retro

        original = retro.resolve_cached_regime_delta
        retro.resolve_cached_regime_delta = lambda *_a, **_k: CachedRegimeDelta(
            shifted=False, shift_reason="no change in risk appetite or shock"
        )
        try:
            audit._dispositions((tmp_path,), None)
        finally:
            retro.resolve_cached_regime_delta = original

        out = capsys.readouterr().out
        ceiling_line = next(ln for ln in out.splitlines() if "carry a recorded" in ln)
        assert ceiling_line.strip().startswith("1 "), ceiling_line
        assert "structural, not incidental" not in out, (
            "the zero-diagnosis block must not print when the ceiling is nonzero"
        )

    @pytest.mark.parametrize("mode", ["--dispositions"])
    def test_the_mode_is_read_only(self, tmp_path, mode):
        """No pricing, no writes: the artifact must be byte-identical after."""
        path = _artifact(tmp_path, "T.T_20260101_000000_analysis.json")
        before = path.read_bytes()
        audit._dispositions((tmp_path,), None)
        assert path.read_bytes() == before


class TestNoExtrapolatedStatisticsAreReported:
    """The report computes from disk or says it cannot.

    An earlier version multiplied the eligible count by a 42.7% trigger rate
    measured across five band-ordered probes. The rate was real, but it is not a
    corpus statistic, is not derivable from this report's inputs, and reads as
    computed because everything around it is. It also changed no decision — the
    injectable ceiling already answers whether a sweep is worth paying for.
    """

    def test_no_trigger_rate_is_published(self, tmp_path, capsys):
        _artifact(tmp_path, "T.T_20260101_000000_analysis.json")
        audit._dispositions((tmp_path,), None)
        out = capsys.readouterr().out
        assert "42.7" not in out
        assert "cannot be known without pricing them" in out

    def test_the_hypothetical_table_does_not_claim_to_be_triggered(
        self, tmp_path, capsys
    ):
        _artifact(tmp_path, "T.T_20260101_000000_analysis.json")
        audit._dispositions((tmp_path,), None)
        out = capsys.readouterr().out
        assert "HYPOTHETICAL" in out
        assert "counts of\neligible candidates, not of triggered comparisons" in out
        assert "Active tenders are excluded here" in out
