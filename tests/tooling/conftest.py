"""Safety-net fixtures for tests/tooling/ — reset singletons after each test."""

import pytest

from src.tooling.inspection_service import INSPECTION_SERVICE
from src.tooling.inspector import NullInspector
from src.tooling.runtime import TOOL_SERVICE


@pytest.fixture(autouse=True)
def _reset_inspection_singletons():
    """Reset INSPECTION_SERVICE and TOOL_SERVICE after each test in tests/tooling/."""
    yield
    INSPECTION_SERVICE.configure(NullInspector())
    TOOL_SERVICE.clear_hooks()
