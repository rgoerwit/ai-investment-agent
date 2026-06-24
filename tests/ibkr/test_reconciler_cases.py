"""Collect the IBKR reconciliation test classes that live in ``reconciler_cases.py``.

``reconciler_cases.py`` is the former ``test_reconciler.py`` — commit d860024
("Refactor oversized runtime/IBKR/validator modules...") renamed it to a name that
no longer matches ``python_files = ["test_*.py"]`` (pyproject), intending it to serve
as a shared helper module (``_make_position`` / ``_make_analysis`` / ``reconcile`` are
imported from it by ~15 other test files). The side effect was that the ~240 ``Test*``
classes still living inside it **silently stopped being collected** — they only ran
when the file was named explicitly, which CI never does. None of those classes are
duplicated under a collected name (the migration created differently-named classes such
as ``TestCheckStalenessStructural``), so they are the sole coverage for staleness,
profit-take classification, correlated-sell detection, currency accuracy, and the macro
event detectors.

This shim re-exposes them under a collected module name without touching the helper
module path the other 15 test files depend on. ``import *`` pulls in the ``Test*``
classes (which pytest then collects from this module) along with imported helpers and
symbols, which pytest ignores. A future cleanup can fully migrate the classes into
properly named ``test_*.py`` files and delete this shim.
"""

from tests.ibkr.reconciler_cases import *  # noqa: F401,F403
