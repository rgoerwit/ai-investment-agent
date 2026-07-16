"""Structural guards for the OCF/NI filing-authority discipline (KTY.WA 2026-06-27).

These assert the prompt-level guardrails survive future edits: filing OCF must be
an exact extraction (FLA), the FILING AUTHORITY switch is gated on that (Senior),
and the growth narrative reconciles trailing vs recent recovery.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _load(name: str) -> dict:
    return json.loads(Path("prompts", name).read_text(encoding="utf-8"))


class TestForeignLanguagePrompt:
    def setup_method(self):
        self.p = _load("foreign_language_analyst.json")
        self.sm = self.p["system_message"]

    def test_version_well_formed(self):
        assert re.match(r"^\d+\.\d+$", self.p["version"])

    def test_requires_exact_extraction(self):
        assert "Exact extraction only" in self.sm

    def test_forbids_approximate_filing_ocf(self):
        # Must instruct N/A rather than a ~/approximate filing OCF.
        assert (
            "never an approximate" in self.sm.lower()
            or "Never report an approximate" in self.sm
        )


class TestSeniorFundamentalsPrompt:
    def setup_method(self):
        self.p = _load("fundamentals_analyst.json")
        self.sm = self.p["system_message"]

    def test_version_well_formed(self):
        assert re.match(r"^\d+\.\d+$", self.p["version"])

    def test_filing_authority_gated_on_exact_extraction(self):
        assert "confidently-extracted statement figure" in self.sm

    def test_net_income_filing_discipline(self):
        assert "Apply this same filing-authority discipline to NET INCOME" in self.sm

    def test_no_elite_cash_overclaim_rule(self):
        assert "Never assert 'elite' or 'superior' cash" in self.sm

    def test_growth_trailing_recovery_reconciliation(self):
        assert "Trailing-vs-recovery reconciliation" in self.sm

    def test_valid_json_roundtrip(self):
        # Re-serialize to confirm nothing exotic crept in.
        assert json.loads(json.dumps(self.p))["version"] == self.p["version"]
