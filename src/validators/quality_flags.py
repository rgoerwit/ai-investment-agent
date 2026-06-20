"""Canonical quality-flag name sets shared across validators and consumers.

Kept dependency-free (stdlib only) so lightweight callers — e.g. IBKR analysis
indexing — can import the constant without pulling in heavier packages such as
``src.agents``. Sibling to ``pfic_constants.py``.
"""

from __future__ import annotations

# Pre-screening flags that mark the earnings base as a cyclical peak / one-time
# inflated. A BUY resting on these should not initiate on a single noisy run;
# the off-watchlist BUY stability gate treats them as "unresolved quality".
PEAK_OR_TRANSIENT_FLAGS: frozenset[str] = frozenset(
    {
        "CYCLICAL_PEAK_WARNING",
        "TRANSIENT_STRENGTH_DISTORTION",
        "MOAT_BONUS_SUPPRESSED_PEAK_TRANSIENT",
        "CAPITAL_EFFICIENCY_BONUS_SUPPRESSED",
    }
)
