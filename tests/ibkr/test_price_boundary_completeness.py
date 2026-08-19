"""A refusal that leaves a second copy is not a refusal.

Round-three review found that the first cut of the price-coherence work nulled
the canonical ``AnalysisRecord`` levels and left the *same rejected numbers* on
the nested ``TradeBlockData`` — which the dashboard serializes verbatim. The
real GAMA.L record therefore still showed operators 900/750/1150 while the
record itself reported ``price_levels_coherent=False``. That is the competing-
store pattern this architecture exists to prevent, introduced by the fix for it.

These tests are deliberately behavioural and run against the **real artifact**
where one exists: a synthetic fixture is what let the gap through, since it
proved the canonical fields were cleared and never looked at the shadow copy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.ibkr.analysis_index import _build_analysis_record_from_file

REAL_GAMA = Path("results/GAMA.L_20260815_100957_analysis.json")

# `results/` is gitignored, so a skipif on the real artifact silently disables
# this regression in CI and in any clean checkout — the assertions that matter
# most would be the ones that never run. The committed fixture carries the same
# shape (levels reachable only via the parsed TRADE_BLOCK, which is exactly the
# path that leaked), so the boundary assertion is mandatory; the real artifact
# stays as an optional canary that the fixture still resembles production.
FIXTURE_GAMA = Path(__file__).resolve().parents[1] / (
    "fixtures/gama_scale_incoherent_analysis.json"
)

# Every price level the real artifact carried, all pence against a 9.76 GBP
# price. None of these may survive anywhere on the loaded record.
REJECTED_LEVELS = (900.0, 750.0, 1150.0, 1400.0)


def _write(tmp_path: Path, ticker: str, snapshot: dict) -> Path:
    path = tmp_path / f"{ticker}_20260815_100957_analysis.json"
    path.write_text(json.dumps({"prediction_snapshot": snapshot, "reports": {}}))
    return path


class TestRejectedLevelsLeaveNoShadowCopy:
    """Mandatory: runs on the committed fixture, not the gitignored corpus."""

    @pytest.mark.parametrize("artifact", [FIXTURE_GAMA], ids=["committed-fixture"])
    def test_no_rejected_level_survives_anywhere_on_the_record(self, artifact):
        record = _build_analysis_record_from_file(artifact)
        assert record is not None
        assert record.price_levels_coherent is False
        # Whole-payload scan, not named fields: checking named fields is exactly
        # what missed the nested TradeBlockData copy the first time.
        payload = json.dumps(record.model_dump(mode="json"))
        for level in REJECTED_LEVELS:
            assert str(level) not in payload, (
                f"rejected level {level} still reachable in the serialized record"
            )

    @pytest.mark.parametrize("artifact", [FIXTURE_GAMA], ids=["committed-fixture"])
    def test_the_dashboard_payload_is_clean(self, artifact):
        from src.web.ibkr_dashboard.serializers import _serialize_analysis

        record = _build_analysis_record_from_file(artifact)
        assert record is not None
        payload = json.dumps(_serialize_analysis(record), default=str)
        for level in REJECTED_LEVELS:
            assert str(level) not in payload
        assert record.ticker in payload  # not vacuously empty

    def test_the_fixture_still_matches_production(self):
        """The fixture must keep exercising the path that actually leaked.

        If the levels ever arrive via the snapshot instead of the parsed
        TRADE_BLOCK, this fixture would stop covering the shadow-copy case
        while still passing.
        """
        raw = json.loads(FIXTURE_GAMA.read_text())
        snapshot = raw["prediction_snapshot"]
        assert all(
            snapshot[key] is None
            for key in ("entry_price", "stop_price", "target_1_price")
        ), "levels must reach the record only through the parsed TRADE_BLOCK"
        assert "TRADE_BLOCK" in raw["investment_analysis"]["trader_plan"]

    @pytest.mark.skipif(not REAL_GAMA.exists(), reason="real artifact not present")
    def test_the_real_artifact_exposes_no_rejected_level_anywhere(self):
        """Optional canary — the corpus is gitignored, so this may skip."""
        record = _build_analysis_record_from_file(REAL_GAMA)
        assert record is not None
        assert record.price_levels_coherent is False

        # Serialize the whole record the way every consumer eventually does.
        # Scanning the payload rather than named fields is the point: a future
        # surface that adds another price path is caught without editing this.
        payload = json.dumps(record.model_dump(mode="json"))
        for level in REJECTED_LEVELS:
            assert str(level) not in payload, (
                f"rejected level {level} still reachable in the serialized record"
            )

    @pytest.mark.skipif(not REAL_GAMA.exists(), reason="real artifact not present")
    def test_non_price_trade_block_fields_survive(self):
        """Refuse the prices, not the whole block — action/conviction are fine."""
        record = _build_analysis_record_from_file(REAL_GAMA)
        assert record is not None
        assert record.trade_block.action
        assert record.trade_block.conviction

    @pytest.mark.skipif(not REAL_GAMA.exists(), reason="real artifact not present")
    def test_the_dashboard_payload_carries_no_rejected_level(self):
        """The surface that actually showed the operator the bad numbers.

        `_serialize_analysis` reads `analysis.trade_block.*` directly, which is
        why nulling only the canonical fields left the leak wide open.
        """
        from src.web.ibkr_dashboard.serializers import _serialize_analysis

        record = _build_analysis_record_from_file(REAL_GAMA)
        assert record is not None
        payload = json.dumps(_serialize_analysis(record), default=str)
        for level in REJECTED_LEVELS:
            assert str(level) not in payload
        # Sanity: the payload is real, not an empty dict passing vacuously.
        assert record.ticker in payload

    def test_a_coherent_record_keeps_its_trade_block_prices(self, tmp_path):
        """The non-regression: sanitizing must be conditional, not blanket."""
        path = _write(
            tmp_path,
            "OK.L",
            {
                "ticker": "OK.L",
                "analysis_date": "2026-08-15",
                "verdict": "BUY",
                "current_price": 100.0,
                "currency": "GBP",
                "entry_price": 95.0,
                "stop_price": 80.0,
                "target_1_price": 130.0,
            },
        )
        record = _build_analysis_record_from_file(path)
        assert record is not None
        assert record.price_levels_coherent is True
        assert record.entry_price == 95.0


class TestUnassessedIsNotCoherent:
    """`price_levels_coherent` is tri-state; None must not collapse to True.

    The first cut wrote ``not coherence.is_incoherent``, so a record with no
    usable reference price reported ``True`` — an asserted clean result derived
    from missing data, and a direct contradiction of the field's own comment.
    The helper's tri-state was tested; the consumer flattened it one layer down.
    """

    def test_no_reference_price_records_none_not_true(self, tmp_path):
        path = _write(
            tmp_path,
            "NOPRICE.L",
            {
                "ticker": "NOPRICE.L",
                "analysis_date": "2026-08-15",
                "verdict": "HOLD",
                "current_price": None,
                "currency": "GBP",
                "entry_price": 900.0,
            },
        )
        record = _build_analysis_record_from_file(path)
        assert record is not None
        assert record.price_levels_coherent is None
        # Unassessed is not a rejection: the levels are left alone.
        assert record.entry_price == 900.0

    def test_no_levels_at_all_records_none(self, tmp_path):
        path = _write(
            tmp_path,
            "NOLEVELS.T",
            {
                "ticker": "NOLEVELS.T",
                "analysis_date": "2026-08-15",
                "verdict": "HOLD",
                "current_price": 1000.0,
                "currency": "JPY",
            },
        )
        record = _build_analysis_record_from_file(path)
        assert record is not None
        assert record.price_levels_coherent is None

    def test_the_three_states_are_distinguishable(self, tmp_path):
        """True / False / None must not be conflated by any consumer."""
        coherent = _build_analysis_record_from_file(
            _write(
                tmp_path,
                "A.L",
                {
                    "ticker": "A.L",
                    "analysis_date": "2026-08-15",
                    "verdict": "BUY",
                    "current_price": 100.0,
                    "currency": "GBP",
                    "entry_price": 95.0,
                },
            )
        )
        incoherent = _build_analysis_record_from_file(
            _write(
                tmp_path,
                "B.L",
                {
                    "ticker": "B.L",
                    "analysis_date": "2026-08-15",
                    "verdict": "BUY",
                    "current_price": 9.76,
                    "currency": "GBP",
                    "entry_price": 900.0,
                },
            )
        )
        unassessed = _build_analysis_record_from_file(
            _write(
                tmp_path,
                "C.L",
                {
                    "ticker": "C.L",
                    "analysis_date": "2026-08-15",
                    "verdict": "BUY",
                    "current_price": None,
                    "currency": "GBP",
                    "entry_price": 900.0,
                },
            )
        )
        assert [
            coherent.price_levels_coherent,
            incoherent.price_levels_coherent,
            unassessed.price_levels_coherent,
        ] == [True, False, None]
