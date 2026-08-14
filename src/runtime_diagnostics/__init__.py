"""Runtime diagnostics: provider/failure classification + artifact publishability.

Split into two cohesive submodules along their one bridge edge
(`failure_artifact` → `classify_failure`):

- ``failure_classification`` — provider inference + exception→``FailureDetails``.
- ``artifact_status`` — artifact status records, the provenance publication
  contract, and ``build_analysis_validity`` / ``is_publishable_analysis``.

The full historical public surface is re-exported here so every
``from src.runtime_diagnostics import ...`` site keeps working unchanged.
"""

from __future__ import annotations

from src.runtime_diagnostics.artifact_status import (
    FUNDAMENTALS_SYNC_FIELDS,
    OPTIONAL_PUBLISHABLE_ARTIFACTS,
    PROVENANCE_CONTRACT_VERSION,
    QUICK_OPTIONAL_PUBLISHABLE_ARTIFACTS,
    QUICK_REQUIRED_PUBLISHABLE_ARTIFACTS,
    REQUIRED_PUBLISHABLE_ARTIFACTS,
    SYNC_CHECK_FIELDS,
    ArtifactStatus,
    build_analysis_validity,
    failure_artifact,
    get_artifact_status,
    get_optional_publishable_artifacts,
    get_required_publishable_artifacts,
    get_valid_artifact_content,
    has_provenance_contract,
    is_artifact_complete,
    is_artifact_valid,
    is_publishable_analysis,
    stamp_provenance_contract,
    success_artifact,
)
from src.runtime_diagnostics.failure_classification import (
    ArtifactErrorKind,
    FailureDetails,
    FailureKind,
    ProviderName,
    classify_failure,
    get_base_url,
    get_class_name,
    get_endpoint_host,
    get_model_name,
    get_runtime_provider,
    infer_provider,
    is_provider_content_block,
)

__all__ = [
    # failure_classification
    "ProviderName",
    "FailureKind",
    "ArtifactErrorKind",
    "FailureDetails",
    "infer_provider",
    "is_provider_content_block",
    "get_model_name",
    "get_runtime_provider",
    "get_class_name",
    "get_base_url",
    "get_endpoint_host",
    "classify_failure",
    # artifact_status — constants
    "FUNDAMENTALS_SYNC_FIELDS",
    "SYNC_CHECK_FIELDS",
    "REQUIRED_PUBLISHABLE_ARTIFACTS",
    "QUICK_REQUIRED_PUBLISHABLE_ARTIFACTS",
    "OPTIONAL_PUBLISHABLE_ARTIFACTS",
    "QUICK_OPTIONAL_PUBLISHABLE_ARTIFACTS",
    "PROVENANCE_CONTRACT_VERSION",
    # artifact_status — provenance contract
    "stamp_provenance_contract",
    "has_provenance_contract",
    # artifact_status — status records + publishability
    "ArtifactStatus",
    "success_artifact",
    "failure_artifact",
    "get_artifact_status",
    "is_artifact_complete",
    "is_artifact_valid",
    "get_valid_artifact_content",
    "get_required_publishable_artifacts",
    "get_optional_publishable_artifacts",
    "build_analysis_validity",
    "is_publishable_analysis",
]
