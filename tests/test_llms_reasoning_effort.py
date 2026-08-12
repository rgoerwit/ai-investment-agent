"""Tests for provider-model reasoning capability selection.

The resolver is keyed by model *family*, not by vendor: every seat on the
OpenAI plane can be pointed at an OpenAI-compatible endpoint, so a family
registered in the table must behave identically whichever base URL serves it.
"""

import pytest

import src.llms as llms_mod
from src.llms import (
    _DEEP_REASONING_EFFORTS,
    _EFFORT_PREFERENCE_DEEPEST,
    _EFFORT_PREFERENCE_FULL,
    _EFFORT_PREFERENCE_PROSE,
    _EFFORT_PREFERENCE_QUICK,
    _OPENAI_REASONING_EFFORTS,
    _effort_preference_for_mode,
    _openai_reasoning_effort,
    _reserve_class_for_effort,
)

GPT5_MODELS = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.4-mini", "gpt-5"]


def _effort(model: str, *, quick_mode: bool) -> str | None:
    return _openai_reasoning_effort(
        model, preference=_effort_preference_for_mode(quick_mode)
    )


class TestGpt5BehaviorUnchanged:
    """The generalization must not move any GPT-5 seat."""

    @pytest.mark.parametrize("model", GPT5_MODELS)
    def test_quick_mode_uses_low_effort(self, model):
        assert _effort(model, quick_mode=True) == "low"

    @pytest.mark.parametrize("model", GPT5_MODELS)
    def test_normal_mode_uses_medium_effort(self, model):
        assert _effort(model, quick_mode=False) == "medium"

    @pytest.mark.parametrize("model", ["gpt-5.6-sol-pro", "gpt-5.4-pro"])
    def test_pro_variants_get_no_effort(self, model):
        assert _effort(model, quick_mode=True) is None
        assert _effort(model, quick_mode=False) is None

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "o3"])
    def test_non_reasoning_models_get_no_effort(self, model):
        assert _effort(model, quick_mode=True) is None
        assert _effort(model, quick_mode=False) is None

    @pytest.mark.parametrize("model", GPT5_MODELS)
    def test_gpt5_keeps_the_default_reserve(self, model):
        """GPT-5 never resolves deep, so its reserve class is untouched."""
        for quick_mode in (True, False):
            effort = _effort(model, quick_mode=quick_mode)
            assert _reserve_class_for_effort(effort) == "default"


class TestKimiFamily:
    """kimi-k3 documents low|high|max and defaults to max when unset."""

    def test_quick_mode_uses_low_effort(self):
        assert _effort("kimi-k3", quick_mode=True) == "low"

    def test_normal_mode_falls_through_undocumented_medium_to_high(self):
        assert "medium" == _EFFORT_PREFERENCE_FULL[0]
        assert _effort("kimi-k3", quick_mode=False) == "high"

    def test_prose_preference_uses_low_effort(self):
        assert (
            _openai_reasoning_effort("kimi-k3", preference=_EFFORT_PREFERENCE_PROSE)
            == "low"
        )

    def test_deepest_preference_uses_max(self):
        assert (
            _openai_reasoning_effort("kimi-k3", preference=_EFFORT_PREFERENCE_DEEPEST)
            == "max"
        )

    def test_deep_effort_earns_the_deep_reserve(self):
        """The reserve must scale with the reasoning actually requested.

        This is the 1088.HK fix: a high-effort model needs more completion-cap
        headroom than the small default reserve provides.
        """
        assert _reserve_class_for_effort(_effort("kimi-k3", quick_mode=False)) == "deep"
        assert (
            _reserve_class_for_effort(_effort("kimi-k3", quick_mode=True)) == "default"
        )

    @pytest.mark.parametrize(
        "model", ["kimi-k3", "KIMI-K3", "moonshot/kimi-k3", "kimi-k3-turbo-preview"]
    )
    def test_family_matching_is_case_and_vendor_prefix_tolerant(self, model):
        assert _effort(model, quick_mode=True) == "low"


class TestUnregisteredFamilies:
    """An unregistered family must get no reasoning parameter at all."""

    @pytest.mark.parametrize("model", ["glm-5.2", "deepseek-v4", "kimi-k2", "llama-4"])
    def test_no_effort_is_guessed(self, model):
        for preference in (
            _EFFORT_PREFERENCE_QUICK,
            _EFFORT_PREFERENCE_FULL,
            _EFFORT_PREFERENCE_DEEPEST,
        ):
            assert _openai_reasoning_effort(model, preference=preference) is None


class TestTableInvariants:
    def test_longest_prefix_wins_regardless_of_table_order(self, monkeypatch):
        """A shorter prefix must never shadow a longer one.

        The table is hand-maintained, so ordering must not be load-bearing:
        the resolution is asserted against the real table *and* against a
        reversed one, which puts the generic ``gpt-5`` entry ahead of the
        specific ``gpt-5.6`` entry.
        """
        # gpt-5.6 documents "max"; the legacy gpt-5 entry stops at "high".
        expected = {"gpt-5.6-sol": "max", "gpt-5": "high", "kimi-k3": "max"}

        for table in (
            _OPENAI_REASONING_EFFORTS,
            tuple(reversed(_OPENAI_REASONING_EFFORTS)),
            tuple(sorted(_OPENAI_REASONING_EFFORTS, key=lambda entry: len(entry[0]))),
        ):
            monkeypatch.setattr(llms_mod, "_OPENAI_REASONING_EFFORTS", table)
            for model, effort in expected.items():
                assert (
                    _openai_reasoning_effort(
                        model, preference=_EFFORT_PREFERENCE_DEEPEST
                    )
                    == effort
                ), f"{model} resolved wrongly under table order {[p for p, _ in table]}"

    def test_every_preference_resolves_for_every_registered_family(self):
        """A registered family must never fall through to no parameter."""
        for prefix, _efforts in _OPENAI_REASONING_EFFORTS:
            for preference in (
                _EFFORT_PREFERENCE_QUICK,
                _EFFORT_PREFERENCE_FULL,
                _EFFORT_PREFERENCE_PROSE,
                _EFFORT_PREFERENCE_DEEPEST,
            ):
                assert (
                    _openai_reasoning_effort(prefix, preference=preference) is not None
                ), f"{prefix} has no resolution for {preference}"

    def test_registered_efforts_are_documented_values_only(self):
        """Guard against re-adding a value a vendor merely tolerates."""
        known = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
        for prefix, efforts in _OPENAI_REASONING_EFFORTS:
            assert efforts <= known, f"{prefix} declares an unknown effort"

    def test_deep_efforts_are_the_expensive_tail(self):
        assert _DEEP_REASONING_EFFORTS == {"high", "xhigh", "max"}
        assert _reserve_class_for_effort(None) == "default"
        assert _reserve_class_for_effort("minimal") == "default"
