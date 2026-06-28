"""L2 deterministic replay: frozen LLM outputs → stable pure-consumer results.

The pipeline's LLM nodes are stochastic, but the parsers/validators that consume
their text are pure functions. This pins the structured output of those consumers
against committed real captured artifacts (harvested from a 2330.TW quick run
under ``evals/captures/schema_v4/``), so a prompt/parser change that silently
alters downstream extraction is caught at zero LLM cost.

Two flavours:
- **Golden values** — explicit asserts on the high-value extracted fields.
- **Structural invariants** — properties that must hold regardless of the verdict
  (e.g. extreme D/E ⇒ pre-screening ``REJECT``), which are noise-free.

Regenerate the metrics snapshot after an *intended* parser change with::

    UPDATE_REPLAY_SNAPSHOTS=1 poetry run pytest tests/eval/test_deterministic_replay.py

See ``scratch/general-prompt-checking.md`` (L2) for the design.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.charts.extractors.pm_block import extract_pm_block
from src.charts.extractors.valuation import _extract_params
from src.ibkr.order_builder import parse_trade_block
from src.validators.financial_rules import detect_red_flags
from src.validators.metric_extractor import extract_metrics
from src.validators.sector_classifier import detect_sector
from src.validators.supplemental_extractors import extract_value_trap_score

FROZEN = Path(__file__).resolve().parents[1] / "fixtures" / "frozen"
_SNAPSHOT = FROZEN / "2330_TW_metrics_snapshot.json"


@pytest.fixture(autouse=True)
def _replay_no_network(request):
    """Pure parsing only — disable sockets (reuses pytest-socket)."""
    request.getfixturevalue("socket_disabled")


def _read(name: str) -> str:
    return (FROZEN / name).read_text(encoding="utf-8")


# --- golden values --------------------------------------------------------------


def test_data_block_metrics_golden():
    metrics = extract_metrics(_read("2330_TW_data_block.txt"))
    assert metrics["debt_to_equity"] == 20.0
    assert metrics["pe_ratio"] == 27.47
    assert metrics["pb_ratio"] == 8.71
    assert metrics["sector"] == "technology"
    assert metrics["adjusted_health_score"] == 75.0


def test_data_block_red_flags_golden():
    block = _read("2330_TW_data_block.txt")
    flags, pre_screen = detect_red_flags(
        extract_metrics(block), sector=detect_sector(block)
    )
    assert pre_screen == "PASS"
    assert {f["type"] for f in flags} == {
        "LOCAL_COVERAGE_HIGH",
        "OCF_SOURCE_DISCREPANCY",
    }


def test_pm_block_golden():
    pm = extract_pm_block(_read("2330_TW_pm_block.txt"))
    assert pm.verdict == "DO_NOT_INITIATE"
    assert pm.zone == "HIGH"
    assert pm.risk_tally == 1.58
    assert (pm.health_adj, pm.growth_adj) == (75, 100)


def test_value_trap_golden():
    vt = extract_value_trap_score(_read("2330_TW_value_trap_block.txt"))
    assert vt["score"] == 95
    assert vt["verdict"] == "ALIGNED"
    assert vt["trap_risk"] == "LOW"


def test_valuation_params_golden():
    vp = _extract_params(_read("2330_TW_valuation_params.txt"))
    assert vp.method == "P/E_NORMALIZATION"
    assert vp.current_pe == 27.47
    assert vp.peg_ratio == 1.02
    assert vp.confidence == "HIGH"


def test_trade_block_golden():
    tb = parse_trade_block(_read("2330_TW_trade_block.txt"))
    assert tb is not None
    assert tb.action == "HOLD"


# --- structural invariants (verdict-independent) --------------------------------


def test_extreme_leverage_forces_reject():
    """Invariant: D/E above the standard-sector reject threshold ⇒ REJECT.

    The body's ``- D/E: 0.20`` line is what the parser reads first; swapping it
    for an extreme value must flip pre-screening deterministically.
    """
    block = _read("2330_TW_data_block.txt").replace("D/E: 0.20", "D/E: 9.00")
    flags, pre_screen = detect_red_flags(
        extract_metrics(block), sector=detect_sector(block)
    )
    assert pre_screen == "REJECT"
    assert any(f["type"] == "EXTREME_LEVERAGE" for f in flags)


# --- golden snapshot (catches any unexpected field drift) -----------------------


def _metrics_snapshot(block: str) -> dict:
    metrics = extract_metrics(block)
    metrics.pop("_raw_report", None)  # huge + circular; not part of the contract
    return metrics


def test_metrics_snapshot_stable():
    current = _metrics_snapshot(_read("2330_TW_data_block.txt"))
    if os.environ.get("UPDATE_REPLAY_SNAPSHOTS") == "1":
        _SNAPSHOT.write_text(
            json.dumps(current, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        pytest.skip("snapshot regenerated")
    assert _SNAPSHOT.exists(), (
        "metrics snapshot missing — regenerate with "
        "UPDATE_REPLAY_SNAPSHOTS=1 pytest tests/eval/test_deterministic_replay.py"
    )
    expected = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    # Compare via JSON normalization so tuple/list and number formatting agree.
    assert json.loads(json.dumps(current, sort_keys=True, default=str)) == expected, (
        "extract_metrics output drifted from the committed snapshot — if intended, "
        "regenerate with UPDATE_REPLAY_SNAPSHOTS=1"
    )


# --- error handling -------------------------------------------------------------


@pytest.mark.parametrize(
    "bad", ["", "no structured block here", "### --- START DATA_BLOCK ---"]
)
def test_consumers_degrade_on_malformed_input(bad: str):
    """Malformed/empty input returns typed empties — never raises."""
    metrics = extract_metrics(bad)
    assert metrics["debt_to_equity"] is None
    pm = extract_pm_block(bad)
    assert pm.verdict is None
    assert parse_trade_block(bad) is None
    assert extract_value_trap_score(bad)["score"] is None
