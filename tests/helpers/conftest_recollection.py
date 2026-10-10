"""Keep subdirectory conftest fixtures visible when pytest re-collects a directory.

pytest 9.1 defers a conftest's fixtures until the first ``Directory`` collector
for the conftest's directory is collected (``FixtureManager._pending_conftests``
is popped once, in its ``pytest_make_collect_report`` wrapper) and then matches
them by node *identity* (``FixtureManager._matchfactories``).

``Session.collect`` re-collects a parent collector without its duplicate cache
when the remaining argument is a single file directly inside it
(``handle_dupes=False`` "for backward compat"). So a command line such as::

    tests/llm_runtime/a.py tests/test_x.py tests/llm_runtime/b.py

collects the ``tests`` package twice; the second pass builds a new
``Dir llm_runtime`` with the same nodeid, and ``b.py``'s items hang under it.
Its conftest was already parsed for the first instance, so ``b.py`` sees none
of that conftest's fixtures: named ones fail with ``fixture ... not found`` and
autouse ones -- this suite's isolation resets -- are skipped silently.

The test-package layout plays no part: the same loss happens whether or not
either directory has an ``__init__.py``.

The hook below parses the conftest again for every later ``Directory`` with the
same path, so each copy gets its own fixture definitions. It is imported into
``tests/conftest.py``; ``tests/test_conftest_recollection.py`` guards it.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

# Directory path -> the first Directory collector built for it in this session.
_FIRST_DIRECTORY_COLLECTORS = pytest.StashKey[dict[Path, pytest.Directory]]()


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,
) -> Generator[None, pytest.CollectReport, pytest.CollectReport]:
    report = yield
    if isinstance(collector, pytest.Directory):
        firsts = collector.session.stash.setdefault(_FIRST_DIRECTORY_COLLECTORS, {})
        first = firsts.setdefault(collector.path, collector)
        if first is not collector:
            conftest_path = collector.path / "conftest.py"
            for name, plugin in collector.config.pluginmanager.list_name_plugin():
                if name and Path(name) == conftest_path:
                    collector.session._fixturemanager.parsefactories(
                        holder=plugin, node=collector
                    )
    return report
