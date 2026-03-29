import sys

import pytest

from src.health_check import check_python_version


@pytest.mark.parametrize(
    ("version_info", "expected_ok", "expected_message"),
    [
        ((3, 10, 14), False, "Requires Python >=3.12,<3.13"),
        ((3, 11, 11), True, None),
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
