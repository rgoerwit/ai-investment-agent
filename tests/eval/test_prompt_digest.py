from src.eval.prompt_digest import prompt_digest


def test_prompt_digest_is_stable_for_identical_payloads():
    payload = {
        "agent_key": "research_manager",
        "agent_name": "Research Manager",
        "version": "v1",
        "system_message": "Hello",
        "category": "research",
        "requires_tools": False,
    }

    assert prompt_digest(payload) == prompt_digest(dict(payload))


def test_prompt_digest_changes_when_payload_changes():
    base_payload = {
        "agent_key": "research_manager",
        "agent_name": "Research Manager",
        "version": "v1",
        "system_message": "Hello",
        "category": "research",
        "requires_tools": False,
    }
    changed_payload = {**base_payload, "system_message": "Hello again"}

    assert prompt_digest(base_payload) != prompt_digest(changed_payload)
