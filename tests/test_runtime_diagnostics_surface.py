"""Guards the public import surface of the split ``runtime_diagnostics`` package.

Stage 6 split the 569-line module into ``failure_classification`` +
``artifact_status`` behind a re-exporting ``__init__``. These tests fail if a
re-export is dropped (breaking one of the ~25 ``from src.runtime_diagnostics
import ...`` sites) or if a private implementation name leaks into the package
API.
"""

from __future__ import annotations

import importlib

import pytest

import src.runtime_diagnostics as rd

# The complete historical public surface — every name that was importable from
# the pre-split module and must remain importable from the package.
HISTORICAL_PUBLIC_NAMES = frozenset(
    {
        # failure classification
        "ProviderName",
        "FailureKind",
        "ArtifactErrorKind",
        "FailureDetails",
        "infer_provider",
        "get_model_name",
        "get_class_name",
        "classify_failure",
        # constants
        "FUNDAMENTALS_SYNC_FIELDS",
        "SYNC_CHECK_FIELDS",
        "REQUIRED_PUBLISHABLE_ARTIFACTS",
        "QUICK_REQUIRED_PUBLISHABLE_ARTIFACTS",
        "OPTIONAL_PUBLISHABLE_ARTIFACTS",
        "QUICK_OPTIONAL_PUBLISHABLE_ARTIFACTS",
        "PROVENANCE_CONTRACT_VERSION",
        # provenance contract
        "stamp_provenance_contract",
        "has_provenance_contract",
        # artifact status + publishability
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
    }
)

# Private implementation helpers that must NOT be part of the package API.
PRIVATE_NAMES = (
    "_root_cause",
    "_extract_host",
    "_missing_provenance_failure",
    "_is_quick_mode_result",
    "_HOST_PATTERN",
)


@pytest.mark.parametrize("name", sorted(HISTORICAL_PUBLIC_NAMES))
def test_historical_name_importable(name: str) -> None:
    module = importlib.import_module("src.runtime_diagnostics")
    assert hasattr(
        module, name
    ), f"{name} no longer importable from runtime_diagnostics"


def test_all_matches_historical_surface() -> None:
    assert set(rd.__all__) == HISTORICAL_PUBLIC_NAMES


def test_all_entries_resolve() -> None:
    unresolved = [name for name in rd.__all__ if not hasattr(rd, name)]
    assert not unresolved, f"__all__ names not resolvable: {unresolved}"


def test_private_names_not_in_public_api() -> None:
    leaked = [name for name in PRIVATE_NAMES if name in rd.__all__]
    assert not leaked, f"private names leaked into __all__: {leaked}"


def test_seam_placement() -> None:
    """The two clusters live in their own submodules (the intended seam)."""
    from src.runtime_diagnostics import artifact_status, failure_classification

    assert hasattr(failure_classification, "classify_failure")
    assert hasattr(failure_classification, "_root_cause")
    assert hasattr(artifact_status, "build_analysis_validity")
    assert hasattr(artifact_status, "_missing_provenance_failure")
    # The bridge edge: artifact_status imports classify_failure, not vice versa.
    assert not hasattr(failure_classification, "build_analysis_validity")
