"""Evaluation support modules."""

from .baseline_capture import (
    BaselineCaptureConfig,
    BaselineCaptureManager,
    BaselinePreflightResult,
    get_active_capture_manager,
    get_active_capture_node,
    reset_active_capture_manager,
    set_active_capture_manager,
    set_active_capture_node,
)
from .capture_contract import (
    NODE_CAPTURE_SPECS,
    NodeCaptureSpec,
    get_node_capture_spec,
    iter_baseline_eligible_specs,
    iter_stage3_judge_specs,
)
from .capture_validation import (
    AgentValidationReport,
    CaptureValidationReport,
    validate_agent_bundle,
    validate_capture_bundle,
)
from .constants import CURRENT_CAPTURE_SCHEMA_VERSION

__all__ = [
    "AgentValidationReport",
    "BaselineCaptureConfig",
    "BaselineCaptureManager",
    "BaselinePreflightResult",
    "CaptureValidationReport",
    "CURRENT_CAPTURE_SCHEMA_VERSION",
    "NODE_CAPTURE_SPECS",
    "NodeCaptureSpec",
    "get_active_capture_node",
    "get_active_capture_manager",
    "get_node_capture_spec",
    "iter_baseline_eligible_specs",
    "iter_stage3_judge_specs",
    "reset_active_capture_manager",
    "set_active_capture_node",
    "set_active_capture_manager",
    "validate_agent_bundle",
    "validate_capture_bundle",
]
