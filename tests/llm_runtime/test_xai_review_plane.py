"""End-to-end contract for the xAI Grok review plane.

The per-file suites each check one layer correctly; what they structurally cannot
catch is a disagreement *between* layers. The chain that matters here is
ladder -> resolved effort -> reasoning reserve -> transport kwargs -> billing,
because xAI documents that ``reasoning_effort`` defaults to ``high`` and that
reasoning cannot be disabled. A break anywhere along it is silent: the call
succeeds, spends its whole completion budget thinking, and persists a fragment.
"""

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.llm_runtime.bindings import BindingConfigurationError, resolve_binding_plan
from src.llm_runtime.budgets import resolve_generation_budget
from src.llm_runtime.construction import build_model_for_seat, reasoning_value_for_seat
from src.llm_runtime.profiles import resolve_profile
from src.llm_runtime.provider_policy import (
    _reset_cache_affinity_for_tests,
    cache_affinity_id,
    is_provider_qualified,
    provider_default_headers,
)
from src.llm_runtime.rate_limits import reset_fallback_limiters_for_tests
from src.llm_runtime.seats import SEATS, BindingGroup, SeatId
from src.runtime_diagnostics.failure_classification import infer_provider
from src.service_tiers import provider_flex_active
from src.token_tracker import TokenUsage, _lookup_model_pricing, _provider_for_model

_REVIEW_SEATS = (
    SeatId.CONSULTANT,
    SeatId.AUDITOR,
    SeatId.AUDITOR_ESCALATION,
    SeatId.EDITOR,
    SeatId.ARTICLE_WRITER_REVIEW_FALLBACK,
)


@pytest.fixture(autouse=True)
def _reset_process_state():
    _reset_cache_affinity_for_tests()
    reset_fallback_limiters_for_tests()
    yield
    _reset_cache_affinity_for_tests()
    reset_fallback_limiters_for_tests()


def _xai_settings(**overrides) -> Settings:
    values = {
        "llm_base_provider": "google",
        "llm_review_provider": "xai",
        "llm_regional_provider": "deepseek",
        "llm_writer_provider": "anthropic",
        "llm_operational_provider": "google",
        "llm_judge_provider": "google",
        "google_api_key": "google-key",
        "claude_api_key": "anthropic-key",
        "deepseek_api_key": "deepseek-key",
        "xai_api_key": "xai-key",
        "llm_consultant_mode": "required",
        "llm_auditor_mode": "required",
        "llm_editor_mode": "required",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class TestProfile:
    def test_grok_4_6_carries_the_documented_ladder_and_identity(self) -> None:
        profile = resolve_profile("grok-4.6")
        assert profile.identity.vendor_id == "xai"
        assert profile.identity.model_lineage == "grok"
        assert profile.identity.adapter_kind == "openai_compatible"
        # Vendor-documented set: no ``minimal``/``none``/``max``.
        assert profile.reasoning_ladder == ("low", "medium", "high", "xhigh")

    def test_variant_inherits_by_prefix(self) -> None:
        assert resolve_profile("grok-4.6-fast").identity.vendor_id == "xai"

    def test_grok_4_5_fails_closed_rather_than_inheriting_xhigh(self) -> None:
        # 4.5 documents only low|medium|high. Inheriting 4.6's row would promise
        # a value the vendor rejects, so an unregistered Grok must stay unknown.
        profile = resolve_profile("grok-4.5")
        assert profile.identity.vendor_id == "unknown"
        assert profile.reasoning_ladder == ()

    def test_unregistered_grok_cannot_bind_a_tool_calling_seat(self) -> None:
        with pytest.raises(BindingConfigurationError) as exc_info:
            resolve_binding_plan(_xai_settings(xai_llm_reasoning_model="grok-4.5"))
        assert "grok-4.5" in str(exc_info.value)


class TestQualification:
    def test_review_only(self) -> None:
        assert is_provider_qualified("xai", BindingGroup.REVIEW)
        for group in BindingGroup:
            if group is not BindingGroup.REVIEW:
                assert not is_provider_qualified("xai", group)

    def test_base_binding_is_rejected_by_name(self) -> None:
        with pytest.raises(BindingConfigurationError) as exc_info:
            resolve_binding_plan(_xai_settings(llm_base_provider="xai"))
        message = str(exc_info.value)
        assert "xai" in message and "base" in message


class TestBinding:
    def test_every_review_seat_resolves_to_the_endpoint_scoped_binding(self) -> None:
        plan = resolve_binding_plan(_xai_settings())
        for seat_id in _REVIEW_SEATS:
            binding = plan.bindings[seat_id]
            assert binding.provider == "xai", seat_id
            assert binding.identity.vendor_id == "xai", seat_id
            assert binding.endpoint_host == "api.x.ai", seat_id
            assert binding.model == "grok-4.6", seat_id

    def test_base_plane_is_untouched(self) -> None:
        plan = resolve_binding_plan(_xai_settings())
        assert plan.bindings[SeatId.BULL].provider == "google"
        assert plan.bindings[SeatId.APAC].provider == "deepseek"

    def test_review_independence_holds_against_a_google_base(self) -> None:
        plan = resolve_binding_plan(_xai_settings(llm_require_review_independence=True))
        base = plan.bindings[SeatId.BULL].identity
        review = plan.bindings[SeatId.CONSULTANT].identity
        assert base.vendor_id != review.vendor_id
        assert base.model_lineage != review.model_lineage


class TestEffortAndReserve:
    """The load-bearing chain: an unset effort means no reserve, and xAI still
    reasons at ``high``. Every review seat must therefore resolve an effort."""

    @pytest.mark.parametrize("seat_id", _REVIEW_SEATS)
    def test_every_review_seat_resolves_an_effort(self, seat_id: SeatId) -> None:
        plan = resolve_binding_plan(_xai_settings())
        spec = SEATS[seat_id]
        effort = reasoning_value_for_seat(
            plan.bindings[seat_id].profile, spec.normal_intent, adjust=False
        )
        assert effort is not None, seat_id
        assert effort in ("low", "medium", "high", "xhigh"), seat_id

    def test_escalation_earns_the_deep_reserve(self) -> None:
        plan = resolve_binding_plan(_xai_settings())
        spec = SEATS[SeatId.AUDITOR_ESCALATION]
        effort = reasoning_value_for_seat(
            plan.bindings[SeatId.AUDITOR_ESCALATION].profile,
            spec.normal_intent,
            adjust=False,
        )
        assert effort == "xhigh"
        budget = resolve_generation_budget(
            _xai_settings(), intent_tokens=4096, reasoning_value=effort
        )
        assert budget.reserve_tokens == 8192

    def test_an_absent_effort_would_disable_the_reserve(self) -> None:
        # Documents *why* the ladder is mandatory rather than decorative.
        budget = resolve_generation_budget(
            _xai_settings(), intent_tokens=4096, reasoning_value=None
        )
        assert budget.reserve_tokens == 0


class TestTransport:
    def test_consultant_builds_against_the_xai_endpoint(self) -> None:
        model = build_model_for_seat(
            SeatId.CONSULTANT, settings=_xai_settings(), quick_mode=False
        )
        assert str(model.openai_api_base) == "https://api.x.ai/v1"
        assert model.model_name == "grok-4.6"
        assert model.reasoning_effort == "medium"
        assert model.openai_api_key.get_secret_value() == "xai-key"

    def test_seat_resolved_effort_reaches_the_client(self) -> None:
        # Regression: the compat adapter previously took the seat-resolved
        # effort only for Moonshot, leaving every other compatible review
        # provider to send none.
        model = build_model_for_seat(
            SeatId.AUDITOR_ESCALATION, settings=_xai_settings(), quick_mode=False
        )
        assert model.reasoning_effort == "xhigh"

    def test_cache_affinity_header_is_sent(self) -> None:
        model = build_model_for_seat(
            SeatId.CONSULTANT, settings=_xai_settings(), quick_mode=False
        )
        assert model.default_headers["x-grok-conv-id"] == cache_affinity_id()

    def test_vendor_identity_is_stamped_for_breaker_keying(self) -> None:
        model = build_model_for_seat(
            SeatId.CONSULTANT, settings=_xai_settings(), quick_mode=False
        )
        assert model._llm_vendor_id == "xai"


class TestCacheAffinity:
    def test_stable_within_a_process(self) -> None:
        assert cache_affinity_id() == cache_affinity_id()

    def test_only_xai_receives_headers(self) -> None:
        assert "x-grok-conv-id" in provider_default_headers("xai")
        for provider in ("google", "openai", "anthropic", "moonshot", "zai"):
            assert provider_default_headers(provider) == {}

    def test_header_carries_no_credential(self) -> None:
        value = provider_default_headers("xai")["x-grok-conv-id"]
        assert "xai-key" not in value
        assert value.isalnum()


class TestAccounting:
    def test_grok_is_priced_and_not_left_to_the_default_cache_multiplier(
        self,
    ) -> None:
        pricing = _lookup_model_pricing("grok-4.6")
        assert pricing == {"prompt": 2.00, "cached_prompt": 0.50, "completion": 6.00}
        # 10% of input would be 0.20; xAI's real cached rate is 2.5x that.
        assert pricing["cached_prompt"] > pricing["prompt"] * 0.10

    def test_vendor_prefixed_id_resolves(self) -> None:
        assert _lookup_model_pricing("xai/grok-4.6") == _lookup_model_pricing(
            "grok-4.6"
        )

    def test_billing_vendor(self) -> None:
        assert _provider_for_model("grok-4.6") == "xai"
        assert _provider_for_model("xai/grok-4.6") == "xai"


class TestFailureAttribution:
    def test_grok_over_chatopenai_is_not_misread_as_openai(self) -> None:
        # The haystack joins model *and* class name, and Grok arrives as
        # ChatOpenAI -- so a branch ordered after the OpenAI one would match
        # "openai" in the class name and key the circuit breaker to the wrong
        # vendor, letting an xAI outage fast-fail OpenAI seats.
        assert infer_provider("grok-4.6", "ChatOpenAI") == "xai"

    def test_siblings_are_unaffected(self) -> None:
        assert infer_provider("gpt-5.4", "ChatOpenAI") == "openai"
        assert infer_provider("glm-5.2", "ChatOpenAI") == "zai"
        assert infer_provider("kimi-k3", "ChatOpenAI") == "moonshot"


class TestRateLimit:
    def test_seats_on_one_vendor_share_a_limiter(self) -> None:
        settings = _xai_settings()
        first = build_model_for_seat(
            SeatId.CONSULTANT, settings=settings, quick_mode=False
        )
        second = build_model_for_seat(
            SeatId.EDITOR, settings=settings, quick_mode=False
        )
        assert first.rate_limiter is not None
        assert first.rate_limiter is second.rate_limiter

    def test_absent_ceiling_yields_no_limiter_not_a_zero_rate_one(self) -> None:
        model = build_model_for_seat(
            SeatId.CONSULTANT,
            settings=_xai_settings(xai_rpm_limit=None),
            quick_mode=False,
        )
        assert model.rate_limiter is None

    def test_a_nonpositive_ceiling_is_rejected_at_config_time(self) -> None:
        with pytest.raises(ValidationError):
            _xai_settings(xai_rpm_limit=0)


class TestMissingCredential:
    """A credential gap must degrade or fail by declared seat mode, never crash
    and never construct a client that would authenticate as nobody."""

    @staticmethod
    def _keyless_auto() -> Settings:
        return _xai_settings(
            xai_api_key="",
            llm_consultant_mode="auto",
            llm_auditor_mode="auto",
            llm_editor_mode="auto",
        )

    def test_auto_mode_disables_every_review_seat_honestly(self) -> None:
        plan = resolve_binding_plan(self._keyless_auto())
        for seat_id in (SeatId.CONSULTANT, SeatId.AUDITOR, SeatId.EDITOR):
            status = plan.statuses[seat_id]
            assert status.enabled is False, seat_id
            assert "credential" in (status.reason or ""), seat_id

    def test_required_mode_fails_startup_naming_the_seat(self) -> None:
        with pytest.raises(BindingConfigurationError) as exc_info:
            resolve_binding_plan(
                _xai_settings(xai_api_key="", llm_consultant_mode="required")
            )
        message = str(exc_info.value)
        assert "consultant" in message
        assert "xai" in message

    def test_a_missing_xai_key_leaves_the_base_plane_working(self) -> None:
        # The review plane is optional by construction; losing it must not take
        # the analysis fleet down with it.
        plan = resolve_binding_plan(self._keyless_auto())
        assert plan.statuses[SeatId.BULL].enabled is True
        assert plan.bindings[SeatId.BULL].provider == "google"
        assert plan.statuses[SeatId.PORTFOLIO_MANAGER].enabled is True

    def test_a_whitespace_only_key_is_treated_as_missing(self) -> None:
        plan = resolve_binding_plan(
            _xai_settings(
                xai_api_key="   ",
                llm_consultant_mode="auto",
                llm_auditor_mode="auto",
                llm_editor_mode="auto",
            )
        )
        assert plan.statuses[SeatId.CONSULTANT].enabled is False


class TestEndpointMisconfiguration:
    """The base URL is the only thing standing between an xAI key and another
    vendor's endpoint, so every way of getting it wrong must be loud."""

    def test_blank_base_url_is_rejected_rather_than_defaulting_to_openai(self) -> None:
        # ChatOpenAI turns a blank base_url into None, and the OpenAI SDK fills
        # that with api.openai.com -- which would send the xAI key to OpenAI.
        with pytest.raises(BindingConfigurationError) as exc_info:
            resolve_binding_plan(_xai_settings(xai_api_base=""))
        assert "XAI_API_BASE" in str(exc_info.value)

    def test_whitespace_only_base_url_is_rejected_too(self) -> None:
        with pytest.raises(BindingConfigurationError):
            resolve_binding_plan(_xai_settings(xai_api_base="   "))

    @pytest.mark.parametrize(
        "bad_url", ["not-a-url", "ftp://api.x.ai/v1", "api.x.ai/v1", "://x"]
    )
    def test_malformed_base_url_names_the_setting(self, bad_url: str) -> None:
        with pytest.raises(BindingConfigurationError) as exc_info:
            resolve_binding_plan(_xai_settings(xai_api_base=bad_url))
        assert "XAI_API_BASE" in str(exc_info.value)

    def test_another_vendors_host_is_rejected(self) -> None:
        with pytest.raises(BindingConfigurationError) as exc_info:
            resolve_binding_plan(
                _xai_settings(xai_api_base="https://api.moonshot.ai/v1")
            )
        message = str(exc_info.value)
        assert "api.moonshot.ai" in message and "xai" in message

    def test_credentials_in_the_url_never_reach_the_endpoint_host(self) -> None:
        plan = resolve_binding_plan(
            _xai_settings(xai_api_base="https://user:secret@api.x.ai/v1?k=v")
        )
        host = plan.bindings[SeatId.CONSULTANT].endpoint_host
        assert host == "api.x.ai"
        assert "secret" not in str(host) and "user" not in str(host)

    def test_the_same_guard_covers_the_other_compatible_vendors(self) -> None:
        # The hazard is a property of the compatible transport, not of xAI.
        with pytest.raises(BindingConfigurationError):
            resolve_binding_plan(
                Settings(
                    _env_file=None,
                    llm_base_provider="google",
                    llm_review_provider="moonshot",
                    google_api_key="g",
                    moonshot_api_key="m",
                    moonshot_api_base="",
                )
            )


class TestSeatOverrides:
    def test_a_valid_reasoning_override_is_accepted(self) -> None:
        plan = resolve_binding_plan(
            _xai_settings(llm_seat_reasoning_overrides={"consultant": "xhigh"})
        )
        assert plan.bindings[SeatId.CONSULTANT].reasoning_value_override == "xhigh"

    def test_an_effort_outside_the_documented_ladder_is_rejected(self) -> None:
        # "max" is a Kimi/OpenAI value; xAI documents no such level, and sending
        # it would be a hard 400 on a gate-adjacent seat.
        with pytest.raises(BindingConfigurationError) as exc_info:
            resolve_binding_plan(
                _xai_settings(llm_seat_reasoning_overrides={"consultant": "max"})
            )
        message = str(exc_info.value)
        assert "max" in message
        assert "xhigh" in message  # the error names what *is* supported

    def test_a_cross_vendor_model_override_is_rejected(self) -> None:
        with pytest.raises(BindingConfigurationError) as exc_info:
            resolve_binding_plan(
                _xai_settings(llm_seat_model_overrides={"consultant": "gpt-5.4"})
            )
        message = str(exc_info.value)
        assert "gpt-5.4" in message and "openai" in message


class TestQuickMode:
    def test_review_plane_resolves_in_quick_mode(self) -> None:
        plan = resolve_binding_plan(_xai_settings())
        binding = plan.for_seat(SeatId.CONSULTANT, quick_mode=True)
        assert binding.provider == "xai"
        assert binding.model == "grok-4.6"

    @pytest.mark.parametrize("seat_id", _REVIEW_SEATS)
    def test_quick_seats_still_resolve_an_effort(self, seat_id: SeatId) -> None:
        plan = resolve_binding_plan(_xai_settings())
        spec = SEATS[seat_id]
        effort = reasoning_value_for_seat(
            plan.for_seat(seat_id, quick_mode=True).profile,
            spec.quick_intent,
            adjust=False,
        )
        assert effort is not None, seat_id

    def test_transport_builds_in_quick_mode(self) -> None:
        model = build_model_for_seat(
            SeatId.CONSULTANT, settings=_xai_settings(), quick_mode=True
        )
        assert str(model.openai_api_base) == "https://api.x.ai/v1"
        assert model.reasoning_effort in ("low", "medium", "high", "xhigh")


class TestServiceTierIsolation:
    def test_xai_sells_no_flex_tier_and_cannot_inherit_openais(self) -> None:
        # OPENAI_SERVICE_TIER is an OpenAI knob; a compatible vendor selling no
        # tier must not inherit its 900s timeout floor or its cost multiplier.
        assert provider_flex_active("xai") is False

    def test_compatible_client_timeout_applies(self) -> None:
        model = build_model_for_seat(
            SeatId.CONSULTANT,
            settings=_xai_settings(openai_compatible_client_timeout_seconds=300),
            quick_mode=False,
        )
        assert model.request_timeout == 300


class TestLegacySchemaUnaffected:
    def test_xai_fields_do_not_activate_the_new_schema(self) -> None:
        # The xAI fields carry defaults; merely adding them must not flip a
        # legacy operator onto the provider-scoped schema.
        plan = resolve_binding_plan(Settings(_env_file=None, google_api_key="g"))
        assert plan.schema == "legacy"
        assert plan.bindings[SeatId.BULL].provider == "google"


class TestCostAccounting:
    @staticmethod
    def _usage(**overrides) -> TokenUsage:
        values = {
            "timestamp": "2026-08-14T00:00:00Z",
            "agent_name": "Consultant",
            "model_name": "grok-4.6",
            "prompt_tokens": 1_000_000,
            "completion_tokens": 0,
            "total_tokens": 1_000_000,
        }
        values.update(overrides)
        return TokenUsage(**values)

    def test_a_grok_call_prices_from_the_xai_row(self) -> None:
        usage = self._usage(completion_tokens=1_000_000, total_tokens=2_000_000)
        assert usage.estimated_cost_usd == pytest.approx(8.00)  # 2.00 in + 6.00 out

    def test_cached_tokens_bill_at_the_vendor_rate_not_the_default_multiplier(
        self,
    ) -> None:
        usage = self._usage(cached_prompt_tokens=1_000_000)
        # All-cached input at xAI's $0.50/1M, not the 10% default ($0.20).
        assert usage.estimated_cost_usd == pytest.approx(0.50)

    def test_an_xai_call_is_not_billed_at_a_flex_discount(self) -> None:
        # xAI sells no discounted tier, so a stamped tier must not halve cost.
        usage = self._usage(service_tier="standard")
        assert usage.estimated_cost_usd == pytest.approx(2.00)
