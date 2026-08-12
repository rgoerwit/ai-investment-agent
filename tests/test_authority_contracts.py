"""Cross-layer authority contracts: which source may each decision read?

Unit tests elsewhere check that each consumer behaves correctly on its own
inputs. These check the thing that unit tests structurally cannot: that a
consumer reads the *right source* when two layers disagree. An accidental
authority bypass — reparsing prose next to a canonical ledger, trusting an
unvalidated projection, reading a raw status string past its decoder — is
invisible to a per-consumer test because each layer is individually correct.

The rule under test:

    A deterministic check may consume projected text only when the field is
    guaranteed to be a valid canonical projection. Any check that changes a
    gate, score, BUY eligibility, or removes another flag must consume
    canonical or typed evidence directly.

The matrix:

    consumer                          authoritative source
    --------------------------------  ----------------------------------
    pre-screening red-flag engine     DecisionInputs (snapshot-backed)
    verdict floor                     DecisionInputs (snapshot-backed)
    DNI review-candidate tag          PM decision text (deliberate)
    portfolio flag index              persisted red_flags ledger
    report-stage red flags            persisted red_flags (passthrough)
    publication gate                  decoded snapshot/trace contract
"""

from __future__ import annotations

from pathlib import Path

import src.ibkr.analysis_index as analysis_index
from src.decision_inputs import DecisionInputs
from src.reporting.state_access import get_effective_red_flags
from src.runtime_diagnostics.artifact_status import _decode_snapshot_status
from src.validators.metric_extractor import extract_metrics

_DATA_BLOCK = (
    "### --- START DATA_BLOCK ---\n"
    "ADJUSTED_HEALTH_SCORE: 70.0% (based on 12 available points)\n"
    "ADJUSTED_GROWTH_SCORE: 68.0% (based on 6 available points)\n"
    "PE_RATIO_TTM: 14.0\n"
    "REVENUE_CAGR_3Y: 12.0%\n"
    "SECTOR: Industrials\n"
    "### --- END DATA_BLOCK ---"
)


def _snapshot(status: str = "VALID", **scores: float) -> dict:
    return {
        "contract_status": status,
        "scorecards": {
            kind: {"percentage": pct, "decision_eligible": True}
            for kind, pct in scores.items()
        },
    }


class TestScoreAuthority:
    """A VALID snapshot owns the decision scores; a rejected one owns nothing."""

    def test_snapshot_beats_a_diverging_data_block(self):
        inputs = DecisionInputs.from_metrics_and_snapshot(
            extract_metrics(_DATA_BLOCK),
            "Industrials",
            _snapshot(HEALTH=51.0, GROWTH=49.0),
            ticker="TEST",
        )

        assert inputs.health_decision_pct == 51.0
        assert inputs.growth_decision_pct == 49.0
        assert inputs.decision_metrics["adjusted_health_score"] == 51.0

    def test_rejected_contract_never_silently_promotes_the_data_block(self):
        """The failure mode this suite exists for: the canonical layer rejects the
        payload, and the unprojected narrative score quietly becomes the gate."""
        for status in ("INVALID", "DECODE_FAILED"):
            inputs = DecisionInputs.from_metrics_and_snapshot(
                extract_metrics(_DATA_BLOCK),
                "Industrials",
                {"contract_status": status},
                ticker="TEST",
            )
            assert inputs.snapshot_authoritative is False
            assert inputs.health_score_reliable is False
            assert inputs.growth_score_reliable is False

    def test_one_decoder_serves_every_snapshot_authority_reader(self):
        """DecisionInputs and the publication gate must not reach opposite
        verdicts on one payload. A future-schema snapshot claiming VALID is the
        case that separates a raw-string read from a decoded one."""
        payload = {"contract_status": "VALID", "schema_version": 10**6}

        inputs = DecisionInputs.from_metrics_and_snapshot(
            extract_metrics(_DATA_BLOCK), "Industrials", payload, ticker="TEST"
        )

        assert _decode_snapshot_status(payload) == "DECODE_FAILED"
        assert inputs.snapshot_status == _decode_snapshot_status(payload)
        assert inputs.snapshot_authoritative is False


class TestConsumerSourceMatrix:
    """One assertion per load-bearing consumer: which source did it read?"""

    def test_verdict_floor_reads_decision_inputs(self):
        import inspect

        from src.agents.verdict_policy import maybe_floor_verdict_to_hold

        params = inspect.signature(maybe_floor_verdict_to_hold).parameters
        assert "decision_inputs" in params
        assert "fundamentals_report" not in params

    def test_dni_tag_deliberately_reads_the_pm_text(self):
        """Not a bypass: the tag describes what the PM asserted about its own
        gates, so the PM document is the right source. Pinned so it is not
        "corrected" to DecisionInputs later."""
        import inspect

        from src.agents.verdict_policy import maybe_tag_dni_review_candidate

        params = inspect.signature(maybe_tag_dni_review_candidate).parameters
        assert "decision_inputs" not in params
        source = inspect.getsource(maybe_tag_dni_review_candidate)
        assert "parse_final_decision_scores" in source

    def test_portfolio_index_reads_the_ledger_not_the_prose(self, monkeypatch):
        def boom(*args, **kwargs):  # pragma: no cover - must never run
            raise AssertionError("the persisted ledger must win over saved prose")

        monkeypatch.setattr(analysis_index, "extract_metrics", boom)

        capital, quality, source = analysis_index._extract_flag_types(
            {
                "red_flags": [{"type": "CYCLICAL_PEAK_WARNING"}],
                "reports": {"fundamentals_report": _DATA_BLOCK},
            },
            "TEST.T",
        )

        assert source == "PERSISTED_CANONICAL"
        assert quality == ("CYCLICAL_PEAK_WARNING",)
        assert capital == ()

    def test_report_stage_is_a_passthrough(self):
        """Nothing between the persisted ledger and the renderer may drop a flag."""
        flags = [{"type": "OCF_SOURCE_DISCREPANCY", "risk_penalty": 0.5}]
        assert get_effective_red_flags({"red_flags": flags}) == flags


class TestDegradationIsHonest:
    """Absent evidence must never render as clean evidence."""

    def test_failed_rederivation_is_labelled_not_empty(self, monkeypatch):
        monkeypatch.setattr(
            analysis_index,
            "extract_metrics",
            lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")),
        )

        record = analysis_index._build_analysis_record_from_data(
            Path("TEST.T_20260601_000000_analysis.json"),
            {
                "prediction_snapshot": {"ticker": "TEST.T", "currency": "JPY"},
                "reports": {"fundamentals_report": _DATA_BLOCK},
                "investment_analysis": {"trader_plan": ""},
            },
        )

        assert record is not None
        assert record.quality_flag_types == ()
        assert record.quality_flags_available is False

    def test_buy_stability_withholds_when_flag_evidence_is_missing(self):
        from src.ibkr.buy_stability import BuyStabilityConfig, assess_buy_stability

        cfg = BuyStabilityConfig(enabled=True, lookback_days=7, margin_tally=0.5)
        assert (
            assess_buy_stability(
                "BUY",
                [],
                risk_tally=0.9,
                active_flags=(),
                flags_available=False,
                cfg=cfg,
            )
            is not None
        )


class TestAdvisoryFindingsStayAdvisory:
    """Observability signals must not become issuer-risk penalties."""

    def test_ocf_not_comparable_is_zero_penalty_and_note_only(self):
        from src.validators.financial_rules import (
            OcfObservation,
            detect_ocf_corroboration_flag,
        )

        flag = detect_ocf_corroboration_flag(
            OcfObservation(amount=1.0e9, period="FY2025", source="DATA_BLOCK"),
            OcfObservation(amount=5.0e8, period="TTM", source="FORENSIC_AUDITOR"),
            ticker="TEST",
        )

        assert flag is not None
        assert flag["type"] == "OCF_PERIOD_NOT_COMPARABLE"
        assert flag["risk_penalty"] == 0.0
        assert flag["action"] == "NOTE"
        assert flag.get("blocks_buy") is not True

    def test_score_unreliable_flags_are_data_quality_not_stock_risk(self):
        from src.validators.financial_rules import detect_red_flags

        inputs = DecisionInputs.from_metrics_and_snapshot(
            extract_metrics(_DATA_BLOCK),
            "Industrials",
            {"contract_status": "INVALID"},
            ticker="TEST",
        )
        flags, _pre_screen = detect_red_flags(inputs, "TEST")
        unreliable = [f for f in flags if f["type"].endswith("_SCORE_UNRELIABLE")]

        assert {f["type"] for f in unreliable} == {
            "HEALTH_SCORE_UNRELIABLE",
            "GROWTH_SCORE_UNRELIABLE",
        }
        # Blocks a BUY (the arithmetic is indeterminate) but never penalizes the
        # name for the model's own failure.
        assert all(f["risk_penalty"] == 0.0 for f in unreliable)
        assert all(f["blocks_buy"] is True for f in unreliable)
