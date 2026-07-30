"""Collected analysis-index tests extracted from reconciler cases."""

from pathlib import Path

import src.ibkr.analysis_index as analysis_index
from src.ibkr.analysis_index import (
    _build_analysis_record_from_data,
    _extract_flag_types,
    _extract_tool1_financial_metrics,
    _parse_scores_from_final_decision,
)
from tests.import_boundary import assert_no_offenders

_REPO_ROOT = Path(__file__).resolve().parents[2]
from tests.ibkr.reconciler_cases import (
    TestLoadLatestAnalyses,
    TestParseScoresFromFinalDecision,
)

_PEAK_FUNDAMENTALS = (
    "### --- START DATA_BLOCK ---\n"
    "ROA_PERCENT: 18.0%\nROA_5Y_AVG: 8.0%\nPROFITABILITY_TREND: UNSTABLE\n"
    "### --- END DATA_BLOCK ---"
)


def test_build_analysis_record_populates_risk_tally_and_quality_flags():
    record = _build_analysis_record_from_data(
        Path("TEST.T_20260601_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "TEST.T",
                "analysis_date": "2026-06-01",
                "verdict": "BUY",
                "currency": "JPY",
                "health_adj": 70.0,
                "risk_tally": 0.75,
            },
            "reports": {"fundamentals_report": _PEAK_FUNDAMENTALS},
            "investment_analysis": {"trader_plan": ""},
        },
    )
    assert record is not None
    assert record.risk_tally == 0.75
    assert "CYCLICAL_PEAK_WARNING" in record.quality_flag_types


class TestPortfolioEvidenceExtraction:
    """Evidence is loaded from persisted run_summary markers + root red_flags —
    never reparsed from prose. Legacy/malformed artifacts degrade to
    complete=False (conservative disposition), never crash indexing."""

    _BASE = {
        "prediction_snapshot": {
            "ticker": "TEST.T",
            "analysis_date": "2026-07-01",
            "verdict": "DO_NOT_INITIATE",
            "currency": "JPY",
            "health_adj": 70.0,
        },
        "investment_analysis": {"trader_plan": ""},
    }

    @staticmethod
    def _record(extra):
        return _build_analysis_record_from_data(
            Path("TEST.T_20260701_000000_analysis.json"),
            {**TestPortfolioEvidenceExtraction._BASE, **extra},
        )

    def test_marker_and_flags_populate_evidence(self):
        record = self._record(
            {
                "run_summary": {"verdict_dni_review_candidate": True},
                "red_flags": [
                    {"type": "GROWTH_SCORE_UNRELIABLE", "blocks_buy": True},
                    {"type": "PFIC_PROBABLE", "risk_penalty": 1.0},
                    {"type": "MOAT_DURABLE_ADVANTAGE", "risk_penalty": -1.0},
                ],
            }
        )
        assert record.evidence.complete is True
        assert record.evidence.dni_review_candidate is True
        assert record.evidence.buy_blocking_flag_types == ("GROWTH_SCORE_UNRELIABLE",)
        assert record.evidence.compliance_flag_types == ("PFIC_PROBABLE",)
        assert record.evidence.mandatory_exit_flag_types == ()

    def test_marker_false_still_counts_as_complete(self):
        """Key presence (not truthiness) distinguishes marker-aware artifacts."""
        record = self._record(
            {
                "run_summary": {"verdict_dni_review_candidate": False},
                "red_flags": [],
            }
        )
        assert record.evidence.complete is True
        assert record.evidence.dni_review_candidate is False

    def test_legacy_artifact_is_evidence_incomplete(self):
        record = self._record({"run_summary": {"quick_mode": False}})
        assert record.evidence.complete is False

    def test_malformed_red_flags_degrade_without_crash(self):
        record = self._record(
            {
                "run_summary": {"verdict_dni_review_candidate": True},
                "red_flags": "not-a-list",
            }
        )
        assert record.evidence.complete is False
        assert record.evidence.buy_blocking_flag_types == ()

    def test_non_dict_flag_entries_are_skipped(self):
        record = self._record(
            {
                "run_summary": {"verdict_dni_review_candidate": False},
                "red_flags": [None, "junk", {"blocks_buy": True}],  # typeless
            }
        )
        assert record.evidence.buy_blocking_flag_types == ()


def test_build_analysis_record_parses_risk_tally_from_decision_fallback():
    # Snapshot missing risk_tally + verdict → fallback parses from the PM decision.
    record = _build_analysis_record_from_data(
        Path("TEST.T_20260601_000000_analysis.json"),
        {
            "prediction_snapshot": {"ticker": "TEST.T", "currency": "JPY"},
            "final_decision": {
                "decision": (
                    "### --- START PM_BLOCK ---\nVERDICT: BUY\nRISK_TALLY: 1.25\n"
                    "### --- END PM_BLOCK ---"
                )
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )
    assert record is not None
    assert record.risk_tally == 1.25


def test_build_analysis_record_preserves_raw_reject_verdict():
    # Regression for the neutral-parser decouple: the shared
    # parse_final_decision_scores must return the RAW verdict so the
    # analysis-index call-site _normalize_verdict keeps "REJECT" verbatim
    # (canonicalize_pm_verdict would have collapsed it to DO_NOT_INITIATE).
    record = _build_analysis_record_from_data(
        Path("TEST.T_20260601_000000_analysis.json"),
        {
            "prediction_snapshot": {"ticker": "TEST.T", "currency": "JPY"},
            "final_decision": {"decision": "PORTFOLIO MANAGER VERDICT: REJECT"},
            "investment_analysis": {"trader_plan": ""},
        },
    )
    assert record is not None
    assert record.verdict == "REJECT"


def test_build_analysis_record_quality_flags_empty_without_fundamentals():
    record = _build_analysis_record_from_data(
        Path("TEST.T_20260601_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "TEST.T",
                "verdict": "BUY",
                "health_adj": 70.0,
                "currency": "JPY",
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )
    assert record is not None
    assert record.quality_flag_types == ()


def test_extract_flag_types_partitions_in_one_pass():
    """Single pass returns (capital_flag_types, quality_flag_types)."""
    capital, quality = _extract_flag_types(
        {"reports": {"fundamentals_report": _PEAK_FUNDAMENTALS}}, "TEST.T"
    )
    assert "CYCLICAL_PEAK_WARNING" in quality
    assert capital == ()  # peak-only report has no idle-cash flags


def test_extract_flag_types_empty_without_fundamentals():
    assert _extract_flag_types({}, "TEST.T") == ((), ())


def test_extract_flag_types_reuses_base_metrics(monkeypatch):
    """Index loading should parse the fundamentals DATA_BLOCK once per file."""
    report = "malformed but present"
    parsed_metrics = {"cycle_position": "PEAK"}
    extract_calls = []

    def fake_extract_metrics(
        fundamentals_report: str,
        *,
        ticker: str | None = None,
        source_file: str | None = None,
    ) -> dict:
        extract_calls.append((fundamentals_report, ticker, source_file))
        return parsed_metrics

    def fake_detect_red_flags(metrics: dict, *, ticker: str):
        assert metrics is parsed_metrics
        assert ticker == "TEST.T"
        return [{"type": "CYCLICAL_PEAK_WARNING"}], 0.0

    def fake_detect_moat_flags(
        fundamentals_report: str,
        ticker: str = "UNKNOWN",
        *,
        base_metrics: dict | None = None,
    ) -> list[dict]:
        assert fundamentals_report == report
        assert ticker == "TEST.T"
        assert base_metrics is parsed_metrics
        return []

    def fake_detect_capital_flags(
        fundamentals_report: str,
        ticker: str = "UNKNOWN",
        value_trap_report: str | None = None,
        sector=None,
        *,
        base_metrics: dict | None = None,
    ) -> list[dict]:
        assert fundamentals_report == report
        assert ticker == "TEST.T"
        assert value_trap_report is None
        assert sector is None
        assert base_metrics is parsed_metrics
        return [{"type": "CAPITAL_IDLE_CASH_RISK"}]

    monkeypatch.setattr(analysis_index, "extract_metrics", fake_extract_metrics)
    monkeypatch.setattr(analysis_index, "detect_red_flags", fake_detect_red_flags)
    monkeypatch.setattr(analysis_index, "detect_moat_flags", fake_detect_moat_flags)
    monkeypatch.setattr(
        analysis_index,
        "detect_capital_efficiency_flags",
        fake_detect_capital_flags,
    )

    capital, quality = analysis_index._extract_flag_types(
        {"reports": {"fundamentals_report": report}},
        "TEST.T",
        source_file="TEST.T_20260601_000000_analysis.json",
    )

    assert extract_calls == [(report, "TEST.T", "TEST.T_20260601_000000_analysis.json")]
    assert capital == ("CAPITAL_IDLE_CASH_RISK",)
    assert quality == ("CYCLICAL_PEAK_WARNING",)


def test_parse_scores_handles_signed_risk_tally():
    """Negative tallies (bonuses below zero) must parse with their sign intact."""
    assert _parse_scores_from_final_decision("RISK_TALLY: -0.5")["risk_tally"] == -0.5
    assert (
        _parse_scores_from_final_decision("TOTAL RISK COUNT: -1.0 (Effective 0.0)")[
            "risk_tally"
        ]
        == -1.0
    )


def test_parse_scores_positive_risk_tally_regression():
    assert _parse_scores_from_final_decision("RISK_TALLY: 2.25")["risk_tally"] == 2.25
    assert (
        _parse_scores_from_final_decision("**TOTAL RISK COUNT**: 3.5")["risk_tally"]
        == 3.5
    )


def test_parse_scores_malformed_risk_tally_safe():
    # No usable number → key absent, no crash.
    assert "risk_tally" not in _parse_scores_from_final_decision("RISK_TALLY: --")


def test_indexing_does_not_import_agents_package():
    """IBKR analysis indexing must not pull in the heavy src.agents stack.

    Runs in a fresh interpreter (sys.modules is polluted by other tests in-process)
    and asserts that importing analysis_index + building a record imports no
    src.agents module — guards the quality-flag constant ownership boundary.
    """
    assert_no_offenders(
        "import sys\n"
        "from pathlib import Path\n"
        "from src.ibkr.analysis_index import _build_analysis_record_from_data as build\n"
        "build(Path('T.T_20260601_000000_analysis.json'), {\n"
        "  'prediction_snapshot': {'ticker':'T.T','verdict':'BUY','health_adj':70.0,"
        "'currency':'JPY','risk_tally':0.5},\n"
        "  'reports': {'fundamentals_report':"
        "'### --- START DATA_BLOCK ---\\nROA_PERCENT: 18.0%\\nROA_5Y_AVG: 8.0%\\n"
        "PROFITABILITY_TREND: UNSTABLE\\n### --- END DATA_BLOCK ---'},\n"
        "  'investment_analysis': {'trader_plan':''}})\n"
        "bad = sorted(m for m in sys.modules if m.startswith('src.agents'))\n"
        "print('AGENTS:' + ','.join(bad))\n",
        sentinel="AGENTS",
        cwd=str(_REPO_ROOT),
        message="IBKR indexing imported src.agents",
    )


def test_build_analysis_record_normalizes_legacy_healthcare_sector():
    record = _build_analysis_record_from_data(
        Path("7203.T_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Healthcare",
                "currency": "JPY",
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.sector == "Health Care"


def test_build_analysis_record_normalizes_consumer_cyclical_sector():
    record = _build_analysis_record_from_data(
        Path("2767.T_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "2767.T",
                "analysis_date": "2026-04-25",
                "verdict": "HOLD",
                "sector": "Consumer Cyclical",
                "currency": "JPY",
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.sector == "Consumer Discretionary"


def test_build_analysis_record_loads_macro_regime_block():
    record = _build_analysis_record_from_data(
        Path("7203.T_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Consumer Cyclical",
                "currency": "JPY",
            },
            "macro_regime_block": {
                "present": True,
                "risk_appetite": "RISK_OFF",
                "shock_type": "ENERGY",
                "shock_phase": "ACUTE",
                "equity_transmission": "EARNINGS_PRESSURE",
                "dip_posture": "WAIT_FOR_CONFIRMATION",
                "confidence": "MEDIUM",
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.macro_regime["risk_appetite"] == "RISK_OFF"


def test_build_analysis_record_legacy_json_defaults_macro_regime_empty():
    record = _build_analysis_record_from_data(
        Path("7203.T_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "7203.T",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Consumer Cyclical",
                "currency": "JPY",
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.macro_regime == {}
    assert record.data_quality == {}


def test_build_analysis_record_loads_data_vacuum_metadata():
    raw_fundamentals = (
        "=== RAW FINANCIAL DATA FOR 1264.TW ===\n\n"
        "### TOOL 1: get_financial_metrics\n"
        '{"_coverage_pct":0.294,"_quality":{"basics_ok":false},'
        '"_sources_used":[],"_ibkr_identity_confidence":"UNVERIFIED",'
        '"_ibkr_probe_error_kind":"NO_MATCH",'
        '"_ticker_rescue_original":"1264.TW",'
        '"_ticker_rescue_resolved":"1264.TWO",'
        '"_ticker_rescue_reason":"sibling_suffix_data_vacuum",'
        '"_ticker_rescue_ibkr_identity_confidence":"VERIFIED",'
        '"trailingPE":13.09}'
    )
    record = _build_analysis_record_from_data(
        Path("1264.TW_20260605_211727_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "1264.TW",
                "analysis_date": "2026-06-05",
                "verdict": "DO_NOT_INITIATE",
                "sector": "Consumer Staples",
                "currency": "TWD",
                "current_price": None,
            },
            "source_artifacts": {"raw_fundamentals_data": raw_fundamentals},
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.data_quality["coverage_pct"] == 29.4
    assert record.data_quality["basics_ok"] is False
    assert record.data_quality["sources_used"] == []
    assert record.data_quality["ibkr_identity_confidence"] == "UNVERIFIED"
    assert record.data_quality["ibkr_probe_error_kind"] == "NO_MATCH"
    assert record.data_quality["ticker_rescue_original"] == "1264.TW"
    assert record.data_quality["ticker_rescue_resolved"] == "1264.TWO"
    assert record.data_quality["ticker_rescue_reason"] == "sibling_suffix_data_vacuum"
    assert record.data_quality["ticker_rescue_ibkr_identity_confidence"] == "VERIFIED"
    assert record.data_quality["data_vacuum"] is True


def test_extract_tool1_financial_metrics_handles_malformed_transcripts():
    assert _extract_tool1_financial_metrics({}) == {}
    assert (
        _extract_tool1_financial_metrics(
            {"source_artifacts": {"raw_fundamentals_data": "no tool data here"}}
        )
        == {}
    )
    assert (
        _extract_tool1_financial_metrics(
            {"source_artifacts": {"raw_fundamentals_data": "get_financial_metrics"}}
        )
        == {}
    )
    assert (
        _extract_tool1_financial_metrics(
            {
                "source_artifacts": {
                    "raw_fundamentals_data": (
                        "### TOOL 1: get_financial_metrics\n{not-json"
                    )
                }
            }
        )
        == {}
    )


def test_build_analysis_record_repairs_legacy_currency():
    record = _build_analysis_record_from_data(
        Path("PINFRA.MX_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "PINFRA.MX",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Industrials",
                "currency": "USD",  # Legacy bug: saved as USD
                "fx_rate_to_usd": 1.0,  # Legacy bug: saved as 1.0
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "MXN"
    assert record.fx_rate_to_usd == 0.0576
    assert record.currency_repaired is True
    assert record.currency_repair_reason == "legacy_snapshot_usd_default"


def test_build_analysis_record_repairs_apr_wa_legacy_currency():
    record = _build_analysis_record_from_data(
        Path("APR.WA_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "APR.WA",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Industrials",
                "currency": "USD",
                "fx_rate_to_usd": 1.0,
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "PLN"
    assert record.fx_rate_to_usd == 0.268
    assert record.currency_repaired is True


def test_build_analysis_record_repairs_apr_ol_legacy_currency():
    record = _build_analysis_record_from_data(
        Path("APR.OL_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "APR.OL",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Industrials",
                "currency": "USD",
                "fx_rate_to_usd": 1.0,
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "NOK"
    assert record.fx_rate_to_usd == 0.105


def test_build_analysis_record_repairs_deme_br_legacy_currency():
    record = _build_analysis_record_from_data(
        Path("DEME.BR_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "DEME.BR",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Industrials",
                "currency": "USD",
                "fx_rate_to_usd": 1.0,
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "EUR"
    assert record.fx_rate_to_usd == 1.153


def test_build_analysis_record_preserves_valid_usd():
    record = _build_analysis_record_from_data(
        Path("AAPL_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "AAPL",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Information Technology",
                "currency": "USD",
                "fx_rate_to_usd": 1.0,
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "USD"
    assert record.fx_rate_to_usd == 1.0
    assert record.currency_repaired is False


def test_build_analysis_record_does_not_repair_bare_apr():
    record = _build_analysis_record_from_data(
        Path("APR_20260425_000000_analysis.json"),
        {
            "prediction_snapshot": {
                "ticker": "APR",
                "analysis_date": "2026-04-25",
                "verdict": "BUY",
                "sector": "Industrials",
                "currency": "USD",
                "fx_rate_to_usd": 1.0,
            },
            "investment_analysis": {"trader_plan": ""},
        },
    )

    assert record is not None
    assert record.currency == "USD"
    assert record.currency_repaired is False


class TestTickerKeySanitizer:
    """Historical runs invoked with trailing colons saved malformed keys
    ('GUD.AX:') that self-collide with the clean ticker in the ambiguity guard."""

    def test_trailing_punctuation_stripped(self):
        from src.ibkr.analysis_index import _sanitize_ticker_key

        assert _sanitize_ticker_key("GUD.AX:") == "GUD.AX"
        assert _sanitize_ticker_key(" cta.jo; ") == "CTA.JO"
        assert _sanitize_ticker_key("7203.T") == "7203.T"

    def test_deserialized_record_ticker_sanitized(self):
        from src.ibkr.analysis_index import _deserialize_analysis_record

        record = _deserialize_analysis_record(
            {"ticker": "GUD.AX:", "analysis_date": "2026-01-10", "file_path": "x"}
        )
        assert record.ticker == "GUD.AX"


class TestKillCriteriaExtraction:
    """Thesis-break triggers load from the saved bear history (v8)."""

    _BEAR = (
        "The bear case rests on leverage.\n"
        "### --- START KILL_CRITERIA ---\n"
        "TRIGGER_1: D/E ratio exceeds 1.8 for two consecutive quarters\n"
        "TRIGGER_2: Two consecutive years of negative free cash flow\n"
        "### --- END KILL_CRITERIA ---\n"
    )

    @staticmethod
    def _record(investment_analysis):
        return _build_analysis_record_from_data(
            Path("TEST.T_20260701_000000_analysis.json"),
            {
                "prediction_snapshot": {
                    "ticker": "TEST.T",
                    "analysis_date": "2026-07-01",
                    "verdict": "BUY",
                    "currency": "JPY",
                    "health_adj": 70.0,
                },
                "investment_analysis": investment_analysis,
            },
        )

    def test_kill_criteria_loaded_from_bear_history(self):
        record = self._record(
            {
                "trader_plan": "",
                "investment_debate": {"bear_history": self._BEAR},
            }
        )
        assert record is not None
        assert record.kill_criteria == (
            "D/E ratio exceeds 1.8 for two consecutive quarters",
            "Two consecutive years of negative free cash flow",
        )

    def test_legacy_artifact_without_debate_yields_empty(self):
        record = self._record({"trader_plan": ""})
        assert record is not None
        assert record.kill_criteria == ()

    def test_malformed_debate_shape_yields_empty(self):
        record = self._record(
            {"trader_plan": "", "investment_debate": {"bear_history": None}}
        )
        assert record is not None
        assert record.kill_criteria == ()
