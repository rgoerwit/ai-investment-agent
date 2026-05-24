"""Tests for the APAC verdict rubric in the APAC Regional Specialist prompt
(Tranche 1, Step 3)."""

from __future__ import annotations

import json
import pathlib


def _load() -> dict:
    path = pathlib.Path("prompts/apac_regional_specialist.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_apac_prompt_version_bumped() -> None:
    data = _load()
    assert data["version"] == "1.2"


def test_apac_prompt_metadata_updated() -> None:
    md = _load()["metadata"]
    assert md["last_updated"] == "2026-05-24"
    assert "v1.2" in md["changes"]


def test_apac_prompt_verdict_rubric_present() -> None:
    msg = _load()["system_message"]
    assert "VERDICT CRITERIA" in msg
    # All three verdict labels still anchored as choices.
    assert "SUPPORT" in msg and "CAUTION" in msg and "OVERRIDE" in msg


def test_apac_prompt_default_is_support() -> None:
    """The rubric must explicitly instruct the model to default to SUPPORT,
    not CAUTION, when no concrete concern is named. This is the load-bearing
    fix for the observed 93% CAUTION collapse."""
    msg = _load()["system_message"]
    assert "SUPPORT, not CAUTION" in msg
    # And the silence sentinel is still preserved.
    assert "NO_MATERIAL_APAC_CONNECTION" in msg


def test_apac_prompt_native_constructs_preserved() -> None:
    """Pre-existing regional-construct content must survive the edit."""
    msg = _load()["system_message"]
    for marker in ("系列", "재벌", "VIE", "ECFA", "Bumiputera", "PBR/ROE"):
        assert marker in msg, f"missing native construct: {marker}"
