"""An explicit seat policy override reaches the request; tiers stay with their vendor.

Before round 4 (October 2026) a client built with an explicit
``client_timeout_seconds`` (``seat_policy_with``) carried it only as its
construction timeout. Every dispatch re-sized the request from the scope's
policy, resolved from the *registered* seat by display name, or from the
class policy for a direct caller, so a writer built for 111 s sent about
298 s. And a compatible vendor, refused a service tier at construction,
forwarded one passed at invocation or through bound kwargs.

The precedence is one chain, resolved at dispatch by ``policy_for_provider``:
the model's explicit override, then the canonical seat policy (the scope),
then the receiving vendor's ceiling -- each bounded by the inherited deadline.
What is asserted is what the vendor library's request layer receives.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.llm_runtime.adapters import transports
from src.llm_runtime.bindings import resolve_binding_plan
from src.llm_runtime.construction import (
    build_model_for_seat,
    seat_policy_with,
    writer_seat_fallback_chain,
)
from src.llm_runtime.failover import failover_attempt
from src.llm_runtime.seats import SeatId
from src.llm_runtime.timeouts import BudgetExhausted, enclosing_deadline
from tests.llm_runtime import test_transport_deadlines as harness
from tests.llm_runtime.test_transport_deadlines import (
    _MESSAGES,
    MARGIN,
    MIN_USEFUL_FRACTION,
    OPENAI_CAP,
    RequestLayer,
    _CapacityError,
    _scope_for,
    _send,
    _send_sync,
    _settings,
)

# The harness's fake clock and pinned budget settings (autouse).
clock = harness.clock
budget_config = harness.budget_config

OVERRIDE = 111.0


def _with_override(seat_id: SeatId, settings: Settings, seconds: float = OVERRIDE):
    return build_model_for_seat(
        seat_id,
        settings=settings,
        plan=resolve_binding_plan(settings),
        execution_policy=seat_policy_with(seat_id, client_timeout_seconds=seconds),
    )


def _legacy_settings() -> Settings:
    # No group selectors: the legacy binding schema.
    return Settings(
        _env_file=None,
        google_api_key="g",
        claude_api_key="a",
        writer_model="claude-haiku-4-5",
    )


def _writer(path: str, seconds: float = OVERRIDE):
    settings = _settings() if path == "canonical" else _legacy_settings()
    llm = _with_override(SeatId.ARTICLE_WRITER, settings, seconds)
    assert isinstance(llm, transports.deadline_aware_anthropic_cls())
    return llm


class TestExplicitOverrideReachesTheRequest:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ["canonical", "legacy"])
    async def test_scoped_async(self, clock, path):
        llm = _writer(path)
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.ARTICLE_WRITER, "anthropic", 300.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(OVERRIDE)

    @pytest.mark.parametrize("path", ["canonical", "legacy"])
    def test_scoped_sync(self, clock, path):
        llm = _writer(path)
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.ARTICLE_WRITER, "anthropic", 300.0, clock),
        ):
            _send_sync(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(OVERRIDE)

    @pytest.mark.asyncio
    async def test_direct_caller_under_an_enclosing_deadline(self, clock):
        llm = _writer("canonical")
        with RequestLayer().patched() as layer, enclosing_deadline(500.0):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(OVERRIDE)

    @pytest.mark.asyncio
    async def test_direct_caller_with_no_deadline_keeps_the_constructed_ceiling(
        self, clock
    ):
        llm = _writer("canonical")
        with RequestLayer().patched() as layer:
            await _send(llm)
        [call] = layer.calls
        assert not isinstance(call["timeout"], int | float)
        assert call["client_timeout"] == pytest.approx(OVERRIDE)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("scoped", [True, False])
    async def test_a_shorter_deadline_still_wins(self, clock, scoped):
        llm = _writer("canonical")
        bound = (
            _scope_for(SeatId.ARTICLE_WRITER, "anthropic", 50.0, clock)
            if scoped
            else enclosing_deadline(50.0)
        )
        with RequestLayer().patched() as layer, bound:
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(50.0 - MARGIN)

    @pytest.mark.asyncio
    async def test_an_override_above_the_vendor_ceiling_is_honoured(self, clock):
        # The rule a registered seat's client timeout already follows: the
        # override replaces the ceiling; only the deadline bounds it.
        llm = _writer("canonical", 500.0)
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.ARTICLE_WRITER, "anthropic", 600.0, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_the_registered_seat_timeout_reaches_a_direct_caller(self, clock):
        # The health probe's registered 10 s, outside the runtime loop.
        llm = build_model_for_seat(
            SeatId.HEALTH_CHECK,
            settings=_settings(llm_operational_provider="openai"),
            plan=resolve_binding_plan(_settings(llm_operational_provider="openai")),
        )
        assert llm.request_cap_override == pytest.approx(10.0)
        with RequestLayer().patched() as layer, enclosing_deadline(500.0):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(10.0)


class TestMinimumUsefulAttempt:
    """The minimum is a share of the override, as of any request ceiling."""

    SECONDS = 100.0

    @pytest.mark.asyncio
    async def test_exactly_the_minimum_is_admitted(self, clock):
        llm = _writer("canonical", self.SECONDS)
        minimum = self.SECONDS * MIN_USEFUL_FRACTION
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.ARTICLE_WRITER, "anthropic", MARGIN + minimum, clock),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(minimum)

    @pytest.mark.asyncio
    async def test_just_below_the_minimum_nothing_is_sent(self, clock):
        llm = _writer("canonical", self.SECONDS)
        minimum = self.SECONDS * MIN_USEFUL_FRACTION
        with (
            RequestLayer().patched() as layer,
            _scope_for(
                SeatId.ARTICLE_WRITER, "anthropic", MARGIN + minimum - 0.001, clock
            ),
            pytest.raises(BudgetExhausted),
        ):
            await llm._agenerate(_MESSAGES)
        assert layer.calls == []


class TestOverrideAcrossTiersAndDelegates:
    @pytest.mark.asyncio
    async def test_flex_is_sized_by_its_ceiling_and_the_reissue_by_the_override(
        self, clock
    ):
        llm = _with_override(
            SeatId.CONSULTANT, _settings(openai_service_tier="flex"), 20.0
        )
        assert llm.service_tier == "flex"
        with (
            RequestLayer(
                script=[_CapacityError("503 flex capacity")]
            ).patched() as layer,
            _scope_for(SeatId.CONSULTANT, "openai", 33.0, clock),
        ):
            await _send(llm)
        flex, standard = layer.calls
        assert (flex["service_tier"], flex["timeout"]) == ("flex", pytest.approx(31.0))
        assert (standard["service_tier"], standard["timeout"]) == (
            "auto",
            pytest.approx(20.0),
        )

    @pytest.mark.asyncio
    async def test_the_failover_copy_keeps_the_override(self, clock):
        settings = _settings(
            openai_llm_reasoning_model="gpt-6.1-sol",
            llm_model_failovers={"gpt-6.1-sol": "gpt-5.4"},
        )
        llm = _with_override(SeatId.CONSULTANT, settings, 20.0)
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "openai", 33.0, clock),
            failover_attempt(),
        ):
            await _send(llm)
        [call] = layer.calls
        assert call["model"] == "gpt-5.4"
        assert call["timeout"] == pytest.approx(20.0)

    def test_model_copy_keeps_the_override(self):
        llm = _writer("canonical")
        assert llm.model_copy().request_cap_override == pytest.approx(OVERRIDE)

    @pytest.mark.asyncio
    async def test_every_writer_fallback_tier_carries_the_override(self, clock):
        settings = _settings()
        tiers = writer_seat_fallback_chain(
            settings=settings,
            plan=resolve_binding_plan(settings),
            policy_overrides={"client_timeout_seconds": 42.0},
        )
        assert [tier.label for tier in tiers] == ["review_group", "base_group"]
        review, base = (tier.build() for tier in tiers)
        assert review.request_cap_override == pytest.approx(42.0)
        assert base.request_cap_override == pytest.approx(42.0)
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.ARTICLE_WRITER_REVIEW_FALLBACK, "openai", 300.0, clock),
        ):
            await _send(review)
        [call] = layer.calls
        assert call["timeout"] == pytest.approx(42.0)

    def test_the_legacy_chain_refuses_an_override_it_cannot_keep(self):
        settings = _legacy_settings()
        with pytest.raises(ValueError, match="client_timeout_seconds"):
            writer_seat_fallback_chain(
                settings=settings,
                plan=resolve_binding_plan(settings),
                policy_overrides={"client_timeout_seconds": 42.0},
            )

    def test_the_legacy_chain_keeps_the_temperature(self):
        settings = _legacy_settings()
        tiers = writer_seat_fallback_chain(
            settings=settings,
            plan=resolve_binding_plan(settings),
            policy_overrides={
                "sampling_temperature": 0.3,
                "client_timeout_seconds": None,
            },
        )
        assert tiers


class TestStamping:
    def test_no_override_leaves_any_model_untouched(self):
        from src.llm_runtime.dispatch import attach_request_cap_override

        model = MagicMock()
        assert attach_request_cap_override(model, None) is model

    def test_an_override_a_model_cannot_carry_fails_loudly(self):
        from src.llm_runtime.dispatch import attach_request_cap_override

        with pytest.raises(TypeError, match="request_cap_override"):
            attach_request_cap_override(object(), 5.0)

    @pytest.mark.parametrize("bad", [0, -1.0])
    def test_a_non_positive_override_is_refused(self, bad):
        with pytest.raises(ValidationError):
            transports.new_openai_chat(
                model="gpt-5.4",
                api_key="k",
                dispatch_provider="openai",
                request_cap_override=bad,
            )

    def test_no_override_stamps_nothing(self):
        llm = build_model_for_seat(
            SeatId.ARTICLE_WRITER,
            settings=_settings(),
            plan=resolve_binding_plan(_settings()),
        )
        assert llm.request_cap_override is None


def _apac():
    settings = _settings()
    llm = build_model_for_seat(
        SeatId.APAC, settings=settings, plan=resolve_binding_plan(settings)
    )
    assert llm.dispatch_provider == "deepseek"
    return llm


class TestCompatibleVendorsRejectTiers:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tier", ["auto", "flex", "default"])
    async def test_an_invocation_tier_is_refused_before_send(self, clock, tier):
        llm = _apac()
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.APAC, "deepseek", 200.0, clock),
            pytest.raises(ValueError, match="service_tier"),
        ):
            await llm._agenerate(_MESSAGES, service_tier=tier)
        assert layer.calls == []

    def test_a_sync_invocation_tier_is_refused_before_send(self, clock):
        llm = _apac()
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.APAC, "deepseek", 200.0, clock),
            pytest.raises(ValueError, match="service_tier"),
        ):
            llm._generate(_MESSAGES, service_tier="auto")
        assert layer.calls == []

    @pytest.mark.asyncio
    async def test_a_bound_tier_is_refused_before_send(self, clock):
        llm = _apac()
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.APAC, "deepseek", 200.0, clock),
            pytest.raises(ValueError, match="service_tier"),
        ):
            await llm.bind(service_tier="flex").ainvoke("hi")
        assert layer.calls == []

    @pytest.mark.asyncio
    async def test_a_failover_copy_refuses_an_invocation_tier(self, clock):
        llm = transports.new_openai_chat(
            model="kimi-k3",
            api_key="k",
            base_url="https://api.moonshot.ai/v1",
            dispatch_provider="moonshot",
            failover_model="kimi-k2.5",
        )
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "moonshot", 200.0, clock),
            failover_attempt(),
            pytest.raises(ValueError, match="service_tier"),
        ):
            await llm._agenerate(_MESSAGES, service_tier="auto")
        assert layer.calls == []

    def test_a_tier_in_model_kwargs_is_refused_at_construction(self):
        # langchain-openai lifts the tier out of model_kwargs into the field.
        with (
            pytest.warns(UserWarning, match="model_kwargs"),
            pytest.raises(ValidationError, match="service_tier"),
        ):
            transports.new_openai_chat(
                model="deepseek-v4-pro",
                api_key="k",
                base_url="https://api.deepseek.com",
                dispatch_provider="deepseek",
                model_kwargs={"service_tier": "auto"},
            )

    @pytest.mark.asyncio
    async def test_an_explicit_none_tier_is_not_a_tier(self, clock):
        llm = _apac()
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.APAC, "deepseek", 200.0, clock),
        ):
            try:
                await llm._agenerate(_MESSAGES, service_tier=None)
            except Exception as exc:  # noqa: BLE001 - the fake layer ends the call
                assert type(exc).__name__ == "Captured"
        [call] = layer.calls
        assert call["service_tier"] == "<absent>"

    @pytest.mark.asyncio
    async def test_the_native_endpoint_still_takes_an_invocation_tier(self, clock):
        llm = build_model_for_seat(
            SeatId.CONSULTANT,
            settings=_settings(),
            plan=resolve_binding_plan(_settings()),
        )
        with (
            RequestLayer().patched() as layer,
            _scope_for(SeatId.CONSULTANT, "openai", 33.0, clock),
        ):
            try:
                await llm._agenerate(_MESSAGES, service_tier="auto")
            except Exception as exc:  # noqa: BLE001 - the fake layer ends the call
                assert type(exc).__name__ == "Captured"
        [call] = layer.calls
        assert call["service_tier"] == "auto"
        assert call["timeout"] == pytest.approx(min(OPENAI_CAP, 31.0))
