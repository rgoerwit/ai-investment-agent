"""Network isolation for the prompt-drift harness (L0/L1).

These tests are pure — they read the committed prompt JSON and exercise the
deterministic parsers, never an LLM or API. Disable sockets so an accidental
network call fails fast rather than silently slowing CI. Reuses pytest-socket's
``socket_disabled`` fixture, mirroring the ``security``-marked guard in the root
``tests/conftest.py`` (no new mechanism, per the harness design).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _prompt_harness_no_network(request):
    request.getfixturevalue("socket_disabled")
