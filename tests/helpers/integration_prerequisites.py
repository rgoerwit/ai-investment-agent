"""Tell a real provider credential from the dummy ``pytest-env`` installs.

``[tool.pytest_env]`` in ``pyproject.toml`` overwrites ``GOOGLE_API_KEY``,
``TAVILY_API_KEY`` and ``FINNHUB_API_KEY`` with public dummies before any
conftest is imported, so the ambient variable of that name can never be an
operator's key. An integration test therefore reads the operator's key from
``INTEGRATION_<NAME>`` (untouched by pytest-env), falls back to ``<NAME>`` for a
run with pytest-env disabled, and rejects any value equal to the dummy
configured for ``<NAME>``. The dummies are read from ``pyproject.toml``, the
single place they are written.

The skip message names the variables to set, never a value.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import tomllib

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
INTEGRATION_PREFIX = "INTEGRATION_"


def pytest_env_dummies(pyproject: Path = _PYPROJECT) -> dict[str, str]:
    """The ``[tool.pytest_env]`` values, keyed by variable name."""
    table = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    section = table.get("tool", {}).get("pytest_env", {})
    return {name: str(value) for name, value in section.items()}


def real_credential(
    name: str,
    environ: Mapping[str, str] | None = None,
    dummies: Mapping[str, str] | None = None,
) -> str | None:
    """The operator's key for ``name``, or ``None`` when only a dummy or nothing is set."""
    env = os.environ if environ is None else environ
    dummy = (pytest_env_dummies() if dummies is None else dummies).get(name)
    for variable in (INTEGRATION_PREFIX + name, name):
        value = env.get(variable, "").strip()
        if value and value != dummy:
            return value
    return None


def missing_credential_reason(name: str) -> str:
    """Skip message naming the prerequisite, without echoing any value."""
    return (
        f"integration prerequisite missing: set {INTEGRATION_PREFIX}{name} to a real "
        f"key (pytest-env overwrites {name} with a dummy)"
    )
