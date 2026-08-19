"""Provider-neutral factory selecting an adapter from a resolved binding."""

from langchain_core.language_models import BaseChatModel

from src.llm_runtime.adapters.anthropic import AnthropicAdapter
from src.llm_runtime.adapters.base import ChatModelAdapter, SeatModelRequest
from src.llm_runtime.adapters.compat import CompatibleAdapter
from src.llm_runtime.adapters.google import GoogleAdapter
from src.llm_runtime.adapters.openai import OpenAIAdapter


class SeatModelFactory:
    def __init__(self) -> None:
        self._adapters: dict[str, ChatModelAdapter] = {
            "google_native": GoogleAdapter(),
            "openai_native": OpenAIAdapter(),
            "anthropic_native": AnthropicAdapter(),
            "openai_compatible": CompatibleAdapter(),
        }

    def adapter_for(self, request: SeatModelRequest) -> ChatModelAdapter:
        try:
            return self._adapters[request.binding.identity.adapter_kind]
        except KeyError as exc:
            raise ValueError(
                f"no adapter for {request.binding.identity.adapter_kind!r}"
            ) from exc

    def build(self, request: SeatModelRequest) -> BaseChatModel | None:
        adapter = self.adapter_for(request)
        llm = adapter.build(request)
        if llm is not None:
            llm._llm_adapter_kind = adapter.kind  # type: ignore[attr-defined]
            llm._llm_runtime_provider = request.binding.provider  # type: ignore[attr-defined]
            llm._llm_vendor_id = request.binding.identity.vendor_id  # type: ignore[attr-defined]
            llm._llm_model_lineage = request.binding.identity.model_lineage  # type: ignore[attr-defined]
        return llm

    def build_openai_transport_fallback(self, model_name: str) -> BaseChatModel:
        adapter = self._adapters["openai_native"]
        if not isinstance(adapter, OpenAIAdapter):
            raise TypeError("OpenAI adapter registration is invalid")
        return adapter.build_transport_fallback(model_name)
