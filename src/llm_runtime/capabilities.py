"""Capability vocabulary shared by seat requirements and model profiles."""

from enum import StrEnum


class Capability(StrEnum):
    """Behavior that must be verified before assigning a model to a seat."""

    TEXT_GENERATION = "text_generation"
    TOOL_CALLING = "tool_calling"
    STRUCTURED_OUTPUT = "structured_output"
    REASONING_CONTROL = "reasoning_control"
    RESPONSES_API = "responses_api"
