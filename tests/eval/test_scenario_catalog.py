from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.scenario_catalog import (
    DEFAULT_SUITE_NAME,
    PromptCheckScenario,
    load_prompt_check_suite,
    load_prompt_check_suite_from_path,
)


def test_load_prompt_check_suite_none_returns_smoke():
    suite = load_prompt_check_suite(None)
    assert suite.name == DEFAULT_SUITE_NAME


def test_smoke_suite_contains_expected_global_basket():
    suite = load_prompt_check_suite("smoke")
    assert [scenario.ticker for scenario in suite.scenarios] == [
        "AAPL",
        "ASML.AS",
        "SAP.DE",
        "7203.T",
        "2330.TW",
        "0005.HK",
    ]


def test_strict_suite_uses_deep_strict_modes():
    suite = load_prompt_check_suite("strict")
    assert len(suite.scenarios) == 6
    assert all(scenario.quick is False for scenario in suite.scenarios)
    assert all(scenario.strict is True for scenario in suite.scenarios)


def test_duplicate_ticker_mode_tuple_rejected(tmp_path: Path):
    path = tmp_path / "dup.json"
    path.write_text(
        json.dumps(
            {
                "suite": "dup",
                "scenarios": [
                    {"ticker": "AAPL", "quick": True},
                    {"ticker": "AAPL", "quick": True},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate scenario"):
        load_prompt_check_suite_from_path(path)


def test_empty_scenarios_rejected(tmp_path: Path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"suite": "empty", "scenarios": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="has no scenarios"):
        load_prompt_check_suite_from_path(path)


def test_unknown_suite_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt_check_suite("not-a-real-suite")


def test_scenarios_ignore_optional_metadata_fields(tmp_path: Path):
    path = tmp_path / "meta.json"
    path.write_text(
        json.dumps(
            {
                "suite": "meta",
                "scenarios": [
                    {
                        "ticker": "AAPL",
                        "label": "Apple",
                        "region": "US",
                        "market": "NASDAQ",
                        "tags": ["us", "tech"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    suite = load_prompt_check_suite_from_path(path)
    assert suite.scenarios == (
        PromptCheckScenario(
            ticker="AAPL",
            quick=True,
            strict=False,
        ),
    )
