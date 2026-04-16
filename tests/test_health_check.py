import sys

import pytest

from src.health_check import check_python_version, get_package_version


@pytest.mark.parametrize(
    ("version_info", "expected_ok", "expected_message"),
    [
        ((3, 10, 14), False, "Requires Python >=3.12,<3.13"),
        ((3, 11, 11), False, "Requires Python >=3.12,<3.13"),
        ((3, 12, 11), True, None),
        ((3, 13, 0), False, "Requires Python >=3.12,<3.13"),
    ],
)
def test_check_python_version_matches_pyproject(
    monkeypatch, version_info, expected_ok, expected_message
):
    monkeypatch.setattr(sys, "version_info", version_info)

    ok, issues = check_python_version()

    assert ok is expected_ok
    if expected_message is None:
        assert issues == []
    else:
        assert issues == [
            f"Python {version_info[0]}.{version_info[1]} detected. {expected_message}"
        ]


# ---------------------------------------------------------------------------
# get_package_version — exercises importlib.metadata.version() path
# A future importlib-metadata major bump that changes the version() contract
# (signature, return type, or exception type) will break these.
# ---------------------------------------------------------------------------


def test_get_package_version_returns_string_for_known_package():
    """version() should return a PEP 440 version string, not None or 'unknown'."""
    result = get_package_version("rich")
    assert isinstance(result, str)
    assert result != "unknown"
    # Coarse sanity: looks like a version (digits and dots)
    assert any(c.isdigit() for c in result)


def test_get_package_version_uses_package_name_override():
    """package_name kwarg routes to importlib.metadata.version(package_name)."""
    result = get_package_version("google.generativeai", "google-generativeai")
    assert isinstance(result, str)


def test_get_package_version_returns_unknown_for_missing_package():
    """Non-existent package must not raise — returns 'unknown' gracefully."""
    result = get_package_version("_no_such_package_xyz_")
    assert result == "unknown"
