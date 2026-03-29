from langchain_core.messages import AIMessage

from src.eval import serialization
from src.eval.serialization import contains_serialization_fallback, normalize_for_json


class OpaqueValue:
    __slots__ = ()


class _Recorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def warning(self, event: str, **kwargs) -> None:
        self.calls.append((event, kwargs))


def test_normalize_for_json_handles_langchain_messages():
    message = AIMessage(
        content="hello",
        additional_kwargs={"foo": "bar"},
        response_metadata={"model": "test"},
    )

    normalized = normalize_for_json(message)

    assert normalized["type"] == "AIMessage"
    assert normalized["content"] == "hello"
    assert normalized["additional_kwargs"] == {"foo": "bar"}
    assert normalized["response_metadata"] == {"model": "test"}


def test_normalize_for_json_tags_serialization_fallback_and_logs_once(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(serialization, "logger", recorder)
    serialization._LOGGED_SERIALIZATION_FALLBACK_TYPES.clear()

    first = normalize_for_json(OpaqueValue())
    second = normalize_for_json(OpaqueValue())

    assert first["__serialization_fallback__"] is True
    assert first["type"] == "OpaqueValue"
    assert second["__serialization_fallback__"] is True
    assert recorder.calls == [("serialization_fallback", {"value_type": "OpaqueValue"})]


def test_contains_serialization_fallback_detects_nested_marker():
    payload = {
        "items": [
            {"ok": True},
            {"nested": {"__serialization_fallback__": True, "type": "OpaqueValue"}},
        ]
    }

    assert contains_serialization_fallback(payload) is True
