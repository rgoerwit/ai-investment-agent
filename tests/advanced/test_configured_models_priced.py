"""A2: no *configured* model may silently hit default pricing.

The blind spot that let ``kimi-k3`` report fabricated cost was that every
pricing assertion used a hand-typed model list — so a model set on a ``Settings``
field but absent from ``MODEL_PRICING_PER_1M`` passed CI green while its cost was
invented at the Flash-class default rate.

This guard derives the model-name fields by **introspecting** ``Settings`` (any
field ending ``_model`` or ``_llm``), so a newly-added model setting is covered
automatically — there is no list to forget to update.

Limit (deliberate): this catches drift in the **code defaults**. The operator's
``.env`` override (e.g. ``APAC_SPECIALIST_MODEL=kimi-k3``) is not a code default,
so it is NOT caught here — that path is covered at runtime by the
``unpriced_models`` field surfaced in the persisted artifact (A3). The two are
complementary.
"""

from src.config import Settings
from src.token_tracker import DEFAULT_PRICING_PER_1M, _lookup_model_pricing

MODEL_FIELDS = sorted(
    f for f in Settings.model_fields if f.endswith(("_model", "_llm"))
)


def test_model_fields_discovered() -> None:
    # If a rename empties this list the parity check below becomes vacuous.
    assert MODEL_FIELDS, "no *_model/*_llm fields found on Settings"
    # The gate-critical seats must be in scope (they end in _llm, not _model —
    # the easy-to-miss case).
    assert {"quick_think_llm", "deep_think_llm"} <= set(MODEL_FIELDS)


def test_no_configured_default_model_hits_default_pricing() -> None:
    cfg = Settings()
    for field in MODEL_FIELDS:
        model = getattr(cfg, field)
        if not model:  # empty/None = unset; falls back to another field
            continue
        assert _lookup_model_pricing(model) is not DEFAULT_PRICING_PER_1M, (
            f"{field}={model!r} is not in MODEL_PRICING_PER_1M — its cost would be "
            "fabricated at the default rate; add it to src/token_tracker.py"
        )


def test_negative_case_junk_model_is_caught(monkeypatch) -> None:
    # Prove the assertion actually fires when a field points at an unpriced model.
    cfg = Settings()
    monkeypatch.setattr(
        cfg, "consultant_model", "totally-unknown-model-9", raising=False
    )
    assert _lookup_model_pricing(cfg.consultant_model) is DEFAULT_PRICING_PER_1M
