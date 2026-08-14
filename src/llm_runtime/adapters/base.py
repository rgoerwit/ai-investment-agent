"""Shared adapter request and protocol."""

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

from src.llm_runtime.bindings import ResolvedBinding
from src.llm_runtime.seats import SeatSpec


@dataclass(frozen=True)
class SeatModelRequest:
    binding: ResolvedBinding
    seat: SeatSpec
    quick_mode: bool
    callbacks: tuple[BaseCallbackHandler, ...] = ()
    output_tokens: int | None = None
    reasoning_value: str | None = None
    service_tier: str | None = None
    settings: Any | None = None


class ChatModelAdapter(Protocol):
    kind: str

    def build(self, request: SeatModelRequest) -> BaseChatModel | None: ...
    def prepare_messages(self, messages: list[Any], *, seat: SeatSpec) -> list[Any]: ...
