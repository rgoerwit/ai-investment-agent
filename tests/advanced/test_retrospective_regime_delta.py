"""Step 7: did the world move, or did we just misjudge it?

A condition *change* is a delta, and a delta needs two endpoints. The snapshot
recorded the regime at decision time (T0) and nothing recorded it at retrospective
time (T1), so "an exogenous shift broke a sound thesis" was inexpressible.

T1 comes from the macro cache that already exists on disk — zero LLM cost, zero
new fetches. What that buys is weak evidence, and the naming says so:
``CachedRegimeDelta``, not ``RegimeDelta``. A macro brief is an advisory LLM
classification, and the cache only refreshes when some analysis happens to run in
that region, so "now" may be days stale.

Every degraded path yields ``shifted=None``, never ``False``. The repository has
been bitten by an unknown masquerading as a negative before (the ``is_quick_mode``
tri-state), and here a false ``False`` would let a market-driven outcome be
written up as a company failure.

The fixtures parse a **real** ``MACRO_REGIME_BLOCK`` — mocking ``parse_macro_regime``
would test the mock, and would not guard the enum contract.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from src.retrospective import (
    REGIME_STALENESS_MAX_DAYS,
    CachedRegimeDelta,
    resolve_cached_regime_delta,
)


def _brief(
    *,
    risk_appetite: str = "RISK_OFF",
    shock_type: str = "RATES",
    confidence: str = "HIGH",
) -> str:
    """A macro report in the shape `parse_macro_regime` actually consumes."""
    return (
        "### RATES & LIQUIDITY\n"
        "- Signal: BEARISH\n\n"
        "MACRO_REGIME_BLOCK:\n"
        f"RISK_APPETITE: {risk_appetite}\n"
        f"SHOCK_TYPE: {shock_type}\n"
        "SHOCK_PHASE: STABILIZING\n"
        "EQUITY_TRANSMISSION: MULTIPLE_COMPRESSION\n"
        "DIP_POSTURE: SCALE_SLOWLY\n"
        f"CONFIDENCE: {confidence}\n"
    )


def _write_cache(
    cache_dir,
    region: str = "KOREA",
    *,
    report: str | None = None,
    age_days: int = 1,
    fingerprint: str = "fp-abc",
    payload: dict | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    trade_date = (datetime.now() - timedelta(days=age_days)).strftime("%Y-%m-%d")
    body = (
        payload
        if payload is not None
        else {
            "version": 1,
            "region": region,
            "trade_date": trade_date,
            "fingerprint": fingerprint,
            "generated_at": datetime.now().isoformat(),
            "status": "generated",
            "report": report if report is not None else _brief(),
        }
    )
    (cache_dir / f"{region}.json").write_text(json.dumps(body))


def _snapshot(
    *,
    risk_appetite: str = "RISK_ON",
    shock_type: str = "NONE",
    confidence: str = "HIGH",
    fingerprint: str | None = "fp-abc",
    ticker: str = "001060.KS",
) -> dict:
    return {
        "ticker": ticker,
        "regime_at_decision": {
            "risk_appetite": risk_appetite,
            "shock_type": shock_type,
            "shock_phase": "NONE",
            "equity_transmission": "FLOWS_SUPPORT",
            "dip_posture": "BUYABLE",
            "confidence": confidence,
            "present": True,
        },
        "regime_confidence": confidence,
        "macro_fingerprint": fingerprint,
    }


class TestARealShift:
    def test_a_changed_risk_appetite_is_a_shift(self, tmp_path):
        _write_cache(tmp_path, report=_brief(risk_appetite="RISK_OFF"))
        delta = resolve_cached_regime_delta(_snapshot(), tmp_path)
        assert delta.shifted is True
        assert "RISK_ON -> RISK_OFF" in delta.shift_reason

    def test_both_changed_legs_are_named(self, tmp_path):
        _write_cache(
            tmp_path, report=_brief(risk_appetite="RISK_OFF", shock_type="RATES")
        )
        delta = resolve_cached_regime_delta(_snapshot(), tmp_path)
        assert "risk appetite: RISK_ON -> RISK_OFF" in delta.shift_reason
        assert "shock type: NONE -> RATES" in delta.shift_reason

    def test_an_unchanged_regime_is_not_a_shift(self, tmp_path):
        _write_cache(
            tmp_path, report=_brief(risk_appetite="RISK_ON", shock_type="NONE")
        )
        delta = resolve_cached_regime_delta(_snapshot(), tmp_path)
        assert delta.shifted is False
        assert delta.regime_now is not None
        assert delta.regime_now["risk_appetite"] == "RISK_ON"

    def test_the_region_is_inferred_from_the_ticker(self, tmp_path):
        _write_cache(tmp_path, region="JAPAN", report=_brief())
        delta = resolve_cached_regime_delta(_snapshot(ticker="7203.T"), tmp_path)
        assert delta.region == "JAPAN"
        assert delta.shifted is True

    def test_downstream_regime_fields_do_not_trigger_a_shift(self, tmp_path):
        """Only risk appetite and shock type are compared.

        ``equity_transmission`` and ``dip_posture`` describe the same shift, so
        counting them would report one change several times.
        """
        report = _brief(risk_appetite="RISK_ON", shock_type="NONE").replace(
            "EQUITY_TRANSMISSION: MULTIPLE_COMPRESSION",
            "EQUITY_TRANSMISSION: EARNINGS_PRESSURE",
        )
        _write_cache(tmp_path, report=report)
        assert resolve_cached_regime_delta(_snapshot(), tmp_path).shifted is False


class TestUnknownIsNeverFalse:
    """Each of these must withhold judgment, not deny a shift."""

    def test_no_cache_file(self, tmp_path):
        delta = resolve_cached_regime_delta(_snapshot(), tmp_path)
        assert delta.shifted is None
        assert "no cached macro brief" in delta.shift_reason

    def test_corrupt_cache_json(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "KOREA.json").write_text("{not json")
        assert resolve_cached_regime_delta(_snapshot(), tmp_path).shifted is None

    def test_a_json_list_instead_of_an_object(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "KOREA.json").write_text("[1,2,3]")
        assert resolve_cached_regime_delta(_snapshot(), tmp_path).shifted is None

    def test_a_brief_with_no_regime_block(self, tmp_path):
        _write_cache(tmp_path, report="### RATES\n- Signal: BEARISH\n")
        delta = resolve_cached_regime_delta(_snapshot(), tmp_path)
        assert delta.shifted is None
        assert "no regime block" in delta.shift_reason

    def test_a_stale_cache(self, tmp_path):
        _write_cache(tmp_path, age_days=REGIME_STALENESS_MAX_DAYS + 1)
        delta = resolve_cached_regime_delta(_snapshot(), tmp_path)
        assert delta.shifted is None
        assert "not 'now'" in delta.shift_reason

    def test_a_cache_exactly_at_the_staleness_limit_is_usable(self, tmp_path):
        _write_cache(tmp_path, age_days=REGIME_STALENESS_MAX_DAYS)
        assert resolve_cached_regime_delta(_snapshot(), tmp_path).shifted is True

    def test_a_cache_with_no_date(self, tmp_path):
        _write_cache(
            tmp_path,
            payload={"region": "KOREA", "fingerprint": "fp-abc", "report": _brief()},
        )
        assert resolve_cached_regime_delta(_snapshot(), tmp_path).shifted is None

    def test_a_legacy_snapshot_with_no_regime(self, tmp_path):
        _write_cache(tmp_path)
        delta = resolve_cached_regime_delta({"ticker": "001060.KS"}, tmp_path)
        assert delta.shifted is None
        assert "no regime recorded" in delta.shift_reason

    def test_low_confidence_at_decision_time(self, tmp_path):
        _write_cache(tmp_path)
        delta = resolve_cached_regime_delta(_snapshot(confidence="LOW"), tmp_path)
        assert delta.shifted is None
        assert "decision-time regime confidence is LOW" in delta.shift_reason

    def test_low_confidence_now(self, tmp_path):
        _write_cache(tmp_path, report=_brief(confidence="LOW"))
        delta = resolve_cached_regime_delta(_snapshot(), tmp_path)
        assert delta.shifted is None
        assert "current regime confidence is LOW" in delta.shift_reason

    def test_a_snapshot_with_no_ticker(self, tmp_path):
        snapshot = _snapshot()
        snapshot.pop("ticker")
        assert resolve_cached_regime_delta(snapshot, tmp_path).shifted is None


class TestAChangedClassifierIsNotAChangedWorld:
    """The subtle guard: a differing label may mean the *prompt* moved."""

    def test_a_fingerprint_mismatch_is_unknown_not_a_shift(self, tmp_path):
        _write_cache(
            tmp_path, fingerprint="fp-NEW", report=_brief(risk_appetite="RISK_OFF")
        )
        delta = resolve_cached_regime_delta(_snapshot(fingerprint="fp-OLD"), tmp_path)
        assert delta.shifted is None
        assert "summarizer prompt changed" in delta.shift_reason

    def test_a_matching_fingerprint_permits_the_comparison(self, tmp_path):
        _write_cache(
            tmp_path, fingerprint="fp-same", report=_brief(risk_appetite="RISK_OFF")
        )
        delta = resolve_cached_regime_delta(_snapshot(fingerprint="fp-same"), tmp_path)
        assert delta.shifted is True

    def test_a_legacy_snapshot_without_a_fingerprint_is_unknown(self, tmp_path):
        """Fail closed on comparability.

        An earlier revision compared anyway when either fingerprint was absent,
        to keep the legacy corpus usable. That optimizes coverage over
        correctness: without both fingerprints there is no way to tell a changed
        *classifier* from a changed *world*, which is the one error this guard
        exists to prevent. Legacy snapshots are honestly unknown; snapshots
        written from now on carry the fingerprint and compare properly.
        """
        _write_cache(tmp_path, report=_brief(risk_appetite="RISK_OFF"))
        delta = resolve_cached_regime_delta(_snapshot(fingerprint=None), tmp_path)
        assert delta.shifted is None
        assert "no macro summarizer fingerprint" in delta.shift_reason

    def test_a_cache_without_a_fingerprint_is_unknown(self, tmp_path):
        _write_cache(
            tmp_path,
            payload={
                "region": "KOREA",
                "trade_date": datetime.now().strftime("%Y-%m-%d"),
                "report": _brief(risk_appetite="RISK_OFF"),
            },
        )
        delta = resolve_cached_regime_delta(_snapshot(), tmp_path)
        assert delta.shifted is None
        assert "carries no summarizer fingerprint" in delta.shift_reason


class TestTheDecisionTimeRegionIsAuthoritative:
    """Re-deriving the region would follow a later mapping change."""

    def test_the_persisted_region_is_preferred_over_the_ticker(self, tmp_path):
        _write_cache(tmp_path, region="JAPAN", report=_brief(risk_appetite="RISK_OFF"))
        snapshot = _snapshot(ticker="001060.KS")  # would infer KOREA
        snapshot["macro_region"] = "JAPAN"
        delta = resolve_cached_regime_delta(snapshot, tmp_path)
        assert delta.region == "JAPAN"
        assert delta.shifted is True

    def test_it_falls_back_to_the_ticker_for_legacy_snapshots(self, tmp_path):
        _write_cache(tmp_path, region="KOREA", report=_brief(risk_appetite="RISK_OFF"))
        delta = resolve_cached_regime_delta(_snapshot(), tmp_path)
        assert delta.region == "KOREA"


class TestShape:
    def test_it_serializes(self, tmp_path):
        _write_cache(tmp_path)
        payload = resolve_cached_regime_delta(_snapshot(), tmp_path).to_dict()
        assert set(payload) == {
            "shifted",
            "shift_reason",
            "regime_now",
            "staleness_days",
            "t1_generated_at",
            "t1_fingerprint",
            "region",
        }

    def test_it_is_immutable(self):
        delta = CachedRegimeDelta(shifted=None, shift_reason="x")
        with pytest.raises(AttributeError):
            delta.shifted = True  # type: ignore[misc]

    def test_provenance_survives_an_unknown_verdict(self, tmp_path):
        """Even when withholding judgment, say which brief was consulted."""
        _write_cache(tmp_path, age_days=REGIME_STALENESS_MAX_DAYS + 5)
        delta = resolve_cached_regime_delta(_snapshot(), tmp_path)
        assert delta.shifted is None
        assert delta.region == "KOREA"
        assert delta.t1_fingerprint == "fp-abc"
        assert delta.staleness_days == REGIME_STALENESS_MAX_DAYS + 5

    def test_it_never_raises_on_a_hostile_snapshot(self, tmp_path):
        _write_cache(tmp_path)
        for hostile in (
            {},
            {"ticker": "001060.KS", "regime_at_decision": "not a mapping"},
            {"ticker": "001060.KS", "regime_at_decision": {}},
            {"ticker": 12345, "regime_at_decision": {"risk_appetite": "RISK_ON"}},
        ):
            assert resolve_cached_regime_delta(hostile, tmp_path).shifted is None


class TestNoNetworkOrLlm:
    def test_resolution_reads_only_the_cache(self, tmp_path, monkeypatch):
        """Zero cost by construction: no fetch, no summarizer call."""

        def _explode(*_a, **_k):
            raise AssertionError("the regime delta must not call the macro pipeline")

        monkeypatch.setattr("src.macro_context.get_macro_context", _explode)
        _write_cache(tmp_path, report=_brief(risk_appetite="RISK_OFF"))
        assert resolve_cached_regime_delta(_snapshot(), tmp_path).shifted is True
