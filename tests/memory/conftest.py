from __future__ import annotations

import pytest

from src.tooling.inspection_service import INSPECTION_SERVICE
from src.tooling.inspector import NullInspector
from src.tooling.runtime import TOOL_SERVICE


@pytest.fixture(autouse=True)
def _reset_tooling_singletons_after_memory_tests():
    yield
    INSPECTION_SERVICE.configure(NullInspector())
    TOOL_SERVICE.clear_hooks()
