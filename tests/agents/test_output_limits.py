from src.agents.output_limits import cap_state_value


def test_cap_state_value_passes_short_content():
    text = "short"
    assert cap_state_value(text, "field") == text


def test_cap_state_value_truncates_oversized_content():
    capped = cap_state_value("x" * 210_000, "field", max_chars=200_000)
    assert capped.endswith("[...truncated]")
    assert len(capped) < 210_000
