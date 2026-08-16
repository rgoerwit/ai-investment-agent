"""Step 6: what the machine was when a verdict was formed.

The retrospective compares a past decision to a realized outcome, which is only
meaningful if the two runs are comparable — and nothing in a persisted artifact
recorded whether the *tooling* had changed. No git SHA, no prompt digest, no
binding digest, no thesis digest; ``app_release`` is a hand-edited string.

The trap this fixes, and the reason the digest is taken from the prompts a run
*used* rather than from the files on disk: ``compute_prompt_set_digest`` reads a
``digest`` key that normal runs never wrote, so it hashed ``{}`` — a **constant**
for every run in the corpus. A fingerprint that always compares equal is worse
than none, because it licenses a comparison that was never checked.

A Langfuse-resolved prompt keeps the local ``version`` and replaces the
``system_message`` (see ``PromptRegistry._resolve_langfuse_prompt``), so a
version-keyed or file-keyed digest is blind to exactly the override that changed
what the model was told.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from src.eval.prompt_digest import (
    agent_prompt_digest,
    agent_prompt_payload,
    prompt_digest,
    stable_digest,
)
from src.run_fingerprint import (
    CONTEXT_CHANGED,
    CONTEXT_SAME,
    CONTEXT_UNKNOWN,
    RunFingerprint,
    _prompt_set_digest,
    _thesis_digest,
    compute_run_fingerprint,
    reset_fingerprint_caches,
)


def _prompt(system_message: str, *, version: str = "1.0") -> SimpleNamespace:
    return SimpleNamespace(
        agent_key="fundamentals_analyst",
        agent_name="Fundamentals Analyst",
        version=version,
        system_message=system_message,
        category="analysis",
        requires_tools=True,
    )


def _used(prompt) -> dict:
    return {
        "fundamentals_report": {
            "agent_name": prompt.agent_name,
            "version": prompt.version,
            "digest": agent_prompt_digest(prompt),
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# The digest primitive
# ══════════════════════════════════════════════════════════════════════════════


class TestStableDigest:
    def test_it_is_order_independent(self):
        assert stable_digest({"a": 1, "b": 2}) == stable_digest({"b": 2, "a": 1})

    def test_it_separates_different_payloads(self):
        assert stable_digest({"a": 1}) != stable_digest({"a": 2})

    def test_the_legacy_name_is_the_same_function(self):
        assert prompt_digest is stable_digest

    def test_the_prompt_payload_includes_the_system_message(self):
        payload = agent_prompt_payload(_prompt("hello"))
        assert payload["system_message"] == "hello"


class TestPromptDigestSeesWhatTheModelWasTold:
    def test_an_edit_without_a_version_bump_changes_the_digest(self):
        """A version-keyed digest would report these two runs identical."""
        before = _prompt("Score health out of 12.", version="9.31")
        after = _prompt("Score health out of 14.", version="9.31")
        assert before.version == after.version
        assert agent_prompt_digest(before) != agent_prompt_digest(after)

    def test_a_langfuse_override_changes_the_digest(self):
        """The real override path: same local version, replaced system_message.

        A digest of the on-disk prompt file cannot see this, which is why the
        fingerprint reads the prompts a run actually used.
        """
        on_disk = _prompt("Original instructions.", version="9.31")
        from_langfuse = _prompt("Overridden instructions.", version="9.31")
        assert agent_prompt_digest(on_disk) != agent_prompt_digest(from_langfuse)

    def test_an_unchanged_prompt_is_stable_across_calls(self):
        prompt = _prompt("Stable text.")
        assert agent_prompt_digest(prompt) == agent_prompt_digest(prompt)

    def test_an_unserializable_prompt_degrades_instead_of_raising(self):
        """Provenance must never cost a run its work.

        A prompt object carrying a non-JSON field (a Mock in tests, an exotic
        metadata value in production) raised out of `_prompt_metadata` and broke
        macro summarization outright. An empty digest is already how consumers
        spell "not recorded".
        """
        from unittest.mock import MagicMock

        assert agent_prompt_digest(MagicMock()) == ""
        assert agent_prompt_digest(None) == ""

    def test_an_empty_digest_is_treated_as_absent_by_the_set_digest(self):
        assert _prompt_set_digest({"a": {"digest": ""}}) is None


class TestPromptSetDigestRefusesToHashNothing:
    def test_entries_without_a_digest_yield_none(self):
        """The regression: this used to hash `{}` — a constant for every run."""
        legacy = {"fundamentals_report": {"agent_name": "F", "version": "9.31"}}
        assert _prompt_set_digest(legacy) is None

    def test_an_empty_mapping_yields_none(self):
        assert _prompt_set_digest({}) is None

    def test_a_non_mapping_yields_none(self):
        assert _prompt_set_digest("not a dict") is None

    def test_two_different_prompt_sets_do_not_collide(self):
        first = _prompt_set_digest(_used(_prompt("A")))
        second = _prompt_set_digest(_used(_prompt("B")))
        assert first is not None and second is not None
        assert first != second

    def test_a_missing_digest_never_compares_same(self):
        """Two runs with unknown prompts must be UNKNOWN, not SAME."""
        blank = RunFingerprint(code_commit="abc", binding_digest="b", thesis_digest="t")
        assert blank.compare(blank) == CONTEXT_UNKNOWN


# ══════════════════════════════════════════════════════════════════════════════
# Thesis constants
# ══════════════════════════════════════════════════════════════════════════════


class TestThesisDigest:
    def test_it_resolves_in_the_real_repository(self):
        assert _thesis_digest() is not None

    def test_mutating_a_threshold_changes_it(self, monkeypatch):
        from src import thesis_constants

        before = _thesis_digest()
        monkeypatch.setattr(thesis_constants, "PE_MAX", 999.0, raising=False)
        assert _thesis_digest() != before

    def test_adding_a_new_threshold_changes_it(self, monkeypatch):
        """Proves introspection, not a hardcoded field list.

        A fixed list is how a table rots: a threshold added later would be
        silently uncovered by the fingerprint that exists to detect it.
        """
        from src import thesis_constants

        before = _thesis_digest()
        monkeypatch.setattr(thesis_constants, "SOME_NEW_GATE_PCT", 42.0, raising=False)
        assert _thesis_digest() != before

    def test_a_private_name_does_not_perturb_it(self, monkeypatch):
        from src import thesis_constants

        before = _thesis_digest()
        monkeypatch.setattr(thesis_constants, "_scratch", 1, raising=False)
        assert _thesis_digest() == before


# ══════════════════════════════════════════════════════════════════════════════
# Bindings
# ══════════════════════════════════════════════════════════════════════════════


class TestBindingDigest:
    def test_it_resolves_against_the_real_settings(self):
        from src.config import config

        fingerprint = compute_run_fingerprint({}, config)
        assert fingerprint.binding_digest is not None

    def test_a_different_model_changes_it(self, monkeypatch):
        """The most direct answer to 'which models changed' — no source edit."""
        from src.config import config
        from src.run_fingerprint import _binding_digest

        before = _binding_digest(config)
        monkeypatch.setattr(config, "quick_think_llm", "gemini-3.9-flash-preview")
        after = _binding_digest(config)
        assert before is not None
        assert after != before

    def test_an_unresolvable_plan_degrades_to_none(self, monkeypatch):
        from src.run_fingerprint import _binding_digest

        monkeypatch.setattr(
            "src.llm_runtime.bindings.resolve_binding_plan",
            lambda _s: (_ for _ in ()).throw(RuntimeError("bad config")),
        )
        assert _binding_digest(SimpleNamespace()) is None


# ══════════════════════════════════════════════════════════════════════════════
# Code identity and comparison
# ══════════════════════════════════════════════════════════════════════════════


class TestCodeIdentity:
    def test_the_commit_matches_the_repository(self):
        import subprocess

        reset_fingerprint_caches()
        fingerprint = compute_run_fingerprint({}, None)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        assert fingerprint.code_commit == head

    def test_a_non_repository_degrades_to_none(self, monkeypatch, tmp_path):
        from src import run_fingerprint as module

        reset_fingerprint_caches()
        monkeypatch.chdir(tmp_path)
        try:
            commit, dirty = module._code_metadata()
            assert commit is None
            assert dirty is False
        finally:
            reset_fingerprint_caches()

    def test_an_absent_git_binary_degrades_to_none(self, monkeypatch, tmp_path):
        from src import run_fingerprint as module

        reset_fingerprint_caches()
        monkeypatch.setenv("PATH", str(tmp_path))
        try:
            commit, _dirty = module._code_metadata()
            assert commit is None
        finally:
            monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
            reset_fingerprint_caches()

    def test_computing_a_fingerprint_never_raises(self, monkeypatch):
        monkeypatch.setattr(
            "src.eval.git_meta.get_git_metadata",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("git exploded")),
        )
        reset_fingerprint_caches()
        try:
            assert isinstance(compute_run_fingerprint({}, None), RunFingerprint)
        finally:
            reset_fingerprint_caches()


class TestComparison:
    def _full(self, **overrides) -> RunFingerprint:
        base = {
            "code_commit": "abc123",
            "code_dirty": False,
            "prompt_set_digest": "sha256:p",
            "binding_digest": "sha256:b",
            "thesis_digest": "sha256:t",
        }
        base.update(overrides)
        return RunFingerprint(**base)

    def test_identical_fingerprints_are_same(self):
        assert self._full().compare(self._full()) == CONTEXT_SAME

    @pytest.mark.parametrize(
        "axis",
        ["code_commit", "prompt_set_digest", "binding_digest", "thesis_digest"],
    )
    def test_any_axis_moving_is_changed(self, axis):
        assert self._full().compare(self._full(**{axis: "different"})) == (
            CONTEXT_CHANGED
        )

    def test_a_dirty_tree_on_either_side_is_unknown(self):
        """Two dirty trees at one commit are not the same machine."""
        clean, dirty = self._full(), self._full(code_dirty=True)
        assert clean.compare(dirty) == CONTEXT_UNKNOWN
        assert dirty.compare(clean) == CONTEXT_UNKNOWN
        assert dirty.compare(dirty) == CONTEXT_UNKNOWN

    def test_two_dirty_trees_are_never_same(self):
        dirty = self._full(code_dirty=True)
        assert dirty.compare(dirty) != CONTEXT_SAME

    def test_a_missing_counterpart_is_unknown(self):
        assert self._full().compare(None) == CONTEXT_UNKNOWN

    @pytest.mark.parametrize(
        "axis",
        ["code_commit", "prompt_set_digest", "binding_digest", "thesis_digest"],
    )
    def test_a_missing_axis_is_unknown_not_same(self, axis):
        """Absence of evidence about comparability is not evidence of it."""
        assert self._full().compare(self._full(**{axis: None})) == CONTEXT_UNKNOWN

    def test_a_legacy_artifact_compares_unknown(self):
        assert self._full().compare(RunFingerprint()) == CONTEXT_UNKNOWN


class TestSerialization:
    def test_round_trip(self):
        original = RunFingerprint(
            code_commit="abc",
            code_dirty=True,
            prompt_set_digest="sha256:p",
            binding_digest="sha256:b",
            thesis_digest="sha256:t",
        )
        assert RunFingerprint.from_dict(original.to_dict()) == original

    def test_from_dict_rejects_a_non_mapping(self):
        assert RunFingerprint.from_dict("nonsense") is None
        assert RunFingerprint.from_dict(None) is None

    def test_a_partial_payload_loads_with_none_axes(self):
        loaded = RunFingerprint.from_dict({"code_commit": "abc"})
        assert loaded is not None
        assert loaded.code_commit == "abc"
        assert loaded.prompt_set_digest is None

    def test_it_is_immutable(self):
        with pytest.raises(AttributeError):
            RunFingerprint().code_commit = "x"  # type: ignore[misc]


class TestPersistedShape:
    def test_the_snapshot_carries_the_fingerprint(self):
        """compare_to_reality loads only the snapshot — a sibling key is invisible."""
        from src.retrospective import extract_snapshot

        payload = RunFingerprint(code_commit="abc").to_dict()
        snapshot = extract_snapshot(
            {"final_trade_decision": "", "fundamentals_report": ""},
            "2767.T",
            run_fingerprint=payload,
        )
        assert snapshot["run_fingerprint"] == payload

    def test_a_legacy_snapshot_has_no_fingerprint(self):
        from src.retrospective import extract_snapshot

        snapshot = extract_snapshot(
            {"final_trade_decision": "", "fundamentals_report": ""}, "2767.T"
        )
        assert snapshot["run_fingerprint"] is None

    def test_the_macro_prompt_records_a_digest(self):
        """It reached `prompts_used` but with no digest, so the fingerprint
        silently excluded the one pre-graph seat that *is* recorded."""
        from src.macro_context import _prompt_metadata

        metadata = _prompt_metadata()
        assert metadata is not None
        assert metadata.get("digest", "").startswith("sha256:")

    def test_the_macro_digest_tracks_the_resolved_prompt(self):
        from src.macro_context import _prompt_metadata
        from src.prompts import get_prompt

        prompt = get_prompt("macro_context_analyst")
        assert _prompt_metadata()["digest"] == agent_prompt_digest(prompt)

    def test_every_recorded_prompt_entry_carries_a_digest(self):
        """A `prompts_used` entry without one is invisible to the fingerprint.

        `compute_prompt_set_digest` keys on `digest`, so a seat recorded without
        it is silently dropped — the failure mode is absence, not an error.
        """
        import inspect

        from src import macro_context
        from src.agents import analyst_nodes, apac_specialist_node

        for module in (analyst_nodes, apac_specialist_node, macro_context):
            source = inspect.getsource(module)
            assert "digest" in source, (
                f"{module.__name__} records a prompt without a content digest"
            )

    def test_both_write_sites_record_a_prompt_digest(self):
        """Without this key the set digest silently hashes an empty payload."""
        import inspect

        from src.agents import analyst_nodes, apac_specialist_node

        for module in (analyst_nodes, apac_specialist_node):
            source = inspect.getsource(module)
            assert "agent_prompt_digest(agent_prompt)" in source, (
                f"{module.__name__} writes prompts_used without a content digest"
            )
