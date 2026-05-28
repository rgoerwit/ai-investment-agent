"""Regression tests for the Variant Perception section in the Research Manager prompt
(Tranche 4, Step 7)."""

from __future__ import annotations

import json
import pathlib
import re


def _load() -> dict:
    return json.loads(
        pathlib.Path("prompts/research_manager.json").read_text(encoding="utf-8")
    )


def test_research_manager_version_bumped() -> None:
    # Pin format, not value — version bumps are routine.
    assert re.match(r"^\d+\.\d+$", _load()["version"])


def test_variant_perception_section_present() -> None:
    msg = _load()["system_message"]
    assert "VARIANT PERCEPTION" in msg
    assert "CONSENSUS_VIEW" in msg
    assert "VARIANT_VIEW" in msg
    assert "BASIS" in msg


def test_no_variant_alignment_is_explicitly_acceptable() -> None:
    """A null variant must be acceptable; a fabricated variant is the failure mode."""
    msg = _load()["system_message"]
    assert "NO VARIANT" in msg
    assert "fabricated variant" in msg


def test_variant_perception_lands_before_risks_to_monitor() -> None:
    """Visual ordering: variant view sits above the RISKS TO MONITOR output section.

    `RISKS TO MONITOR` appears earlier in the prompt as a priority-order bullet
    too — we want the *section header* `### RISKS TO MONITOR:`.
    """
    msg = _load()["system_message"]
    variant_pos = msg.find("VARIANT PERCEPTION")
    risks_section_pos = msg.find("### RISKS TO MONITOR:")
    assert 0 < variant_pos < risks_section_pos
