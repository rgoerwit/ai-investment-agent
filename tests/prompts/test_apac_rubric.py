"""Tests for the APAC verdict rubric in the APAC Regional Specialist prompt
(Tranche 1, Step 3)."""

from __future__ import annotations

import json
import pathlib
import re


def _load() -> dict:
    path = pathlib.Path("prompts/apac_regional_specialist.json")
    return json.loads(path.read_text(encoding="utf-8"))


def test_apac_prompt_version_bumped() -> None:
    version = _load()["version"]
    assert re.match(r"^\d+\.\d+$", version), version
    major, minor = (int(part) for part in version.split("."))
    assert (major, minor) >= (1, 3)


def test_apac_prompt_metadata_updated() -> None:
    md = _load()["metadata"]
    assert md["last_updated"] == "2026-06-29"
    assert "v1.4" in md["changes"]


def test_apac_value_up_credit_is_narrowly_scoped() -> None:
    """The Value-Up execution credit must stay scoped to the narrow Korean / APAC
    family-control slice and must NOT displace the core transmission-channel audit
    for non-Korean names (including non-APAC names with APAC supply chains)."""
    msg = _load()["system_message"]
    assert "VALUE_UP_EXECUTION_CREDIT" in msg
    assert "must never displace" in msg
    assert "non-APAC names with APAC supply chains" in msg


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


def test_apac_prompt_exposure_mode_not_silence_by_default() -> None:
    """v1.4: non-APAC-listed issuers must get a transmission-channel audit
    (APAC as a global supply-chain / market lens), not a default silence. The
    NO_MATERIAL_APAC_CONNECTION sentinel survives but is narrowed to a rare
    fallback."""
    msg = _load()["system_message"]
    assert "APAC-EXPOSURE mode" in msg
    assert "DOMESTIC-APAC mode" in msg
    # Silence is explicitly demoted from the default path.
    assert "SILENCE PROTOCOL (rare)" in msg
    assert "Do NOT silence merely because the issuer is not APAC-listed" in msg
    # Sentinel token stays byte-identical for the node/parsers.
    assert "NO_MATERIAL_APAC_CONNECTION" in msg


def test_apac_prompt_native_constructs_preserved() -> None:
    """Pre-existing regional-construct content must survive the edit."""
    msg = _load()["system_message"]
    for marker in ("系列", "재벌", "VIE", "ECFA", "Bumiputera", "PBR/ROE"):
        assert marker in msg, f"missing native construct: {marker}"
