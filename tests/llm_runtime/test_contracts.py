from types import SimpleNamespace

from src.llm_runtime.contracts import capture_construction_contract


class _FakeTransport(SimpleNamespace):
    pass


def test_capture_construction_contract_keeps_repo_owned_semantics() -> None:
    llm = _FakeTransport(
        model_name="example-reasoning-model",
        _configured_max_completion_tokens=4096,
        _configured_api_completion_tokens=6144,
        _configured_reasoning_reserve_tokens=2048,
        use_responses_api=True,
        service_tier="flex",
        extra_body={"thinking": {"type": "enabled"}},
        request_timeout=120,
        max_retries=3,
    )

    contract = capture_construction_contract(
        llm,
        seat_id="consultant",
        callback_agent="Consultant",
        reasoning_intent="medium",
        limiter_key=("moonshot", "api.moonshot.cn"),
    )

    assert contract.model == "example-reasoning-model"
    assert contract.intent_output_cap_tokens == 4096
    assert contract.api_output_cap_tokens == 6144
    assert contract.configured_reasoning_reserve_tokens == 2048
    assert contract.use_responses_api is True
    assert contract.extra_body == {"thinking": {"type": "enabled"}}
    assert contract.timeout_seconds == 120.0
    assert contract.max_retries == 3
    assert contract.callback_agent == "Consultant"
