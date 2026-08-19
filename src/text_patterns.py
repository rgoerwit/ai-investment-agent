"""Shared text-scanning primitives.

A stdlib-only leaf: this module imports nothing from ``src``, for the same reason
``data_block_utils`` was split out of ``agents/support`` -- so that agents-free consumers
(notably ``src/ibkr/``, guarded by ``tests/ibkr/test_buy_stability.py``) can use it.

Everything here had two or more independent copies before consolidation. Add a pattern
only when a *second* consumer appears; a single-use pattern belongs at its call site.

Naming rule: **the name states the constraint**, so a later reader cannot mistake a
deliberate restriction for an oversight and "fix" it (``EXCHANGE_QUALIFIED_TICKER_RE``,
not ``TICKER_RE``).

Block-structured text (``### --- START X ---`` fences, ``KEY: value`` field lines) is
*not* served here -- that is ``src/data_block_utils.py``, which owns its own
``_NUMBER_TOKEN_PATTERN``. The two leaves stay independent on purpose; importing across
them would couple the block layer to this one for no gain.

Guarded by ``tests/test_regex_consolidation.py``, which AST-scans ``src/`` and fails if
any module re-declares one of these literals.
"""

from __future__ import annotations

import ipaddress
import re

# Exchange-qualified tickers only (``6741.T``, ``AGS.BR``, ``111770.KS``).
#
# The trailing ``.SUFFIX``/``-SUFFIX`` segment is MANDATORY and load-bearing: made
# optional, the body degrades to ``\b[A-Z0-9]{1,8}\b``, which matches every short word --
# "Related Listed Tickers: 6741.T and AGS.BR. THE PARENT NOTE says..." would yield
# ``['Related', 'Listed', 'Tickers', '6741.T', 'and', 'AGS.BR', 'THE', 'PARENT', ...]``.
# This system analyses ex-US equities, where a related listing is always
# exchange-qualified, so the constraint costs nothing real.
#
# Deliberately case-SENSITIVE. Folding case over an all-uppercase class admits prose
# ("Spun-off", "Dual-Listed") and bare domains ("nikkei.com") as ticker symbols; that was
# a live defect in one of the two copies this constant replaced.
EXCHANGE_QUALIFIED_TICKER_RE = re.compile(r"\b[A-Z0-9]{1,8}(?:[.-][A-Z0-9]{1,6})\b")

# URL harvesting from tool output and agent prose.
#
# The ``<>"`` exclusion is the point: a bare ``\S+`` swallows a trailing quote or angle
# bracket ('"https://x/a"' -> 'https://x/a"'), which then fails to fetch. Two looser
# spellings existed alongside three byte-identical copies of this one.
#
# NOTE ``)`` is deliberately NOT excluded -- URLs legitimately contain parentheses
# (Wikipedia-style paths), and callers that need to strip a wrapping paren should do it
# at the call site where the surrounding syntax is known.
URL_RE = re.compile(r'https?://[^\s<>"]+', re.IGNORECASE)

# Sentence splitter for prose auditing. Had four byte-identical copies
# (article_audit x2, pm_claim_audit, validators/financial_rules).
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# No shared numeric-token pattern lives here, deliberately. One was added during the
# consolidation and removed again for having zero consumers -- it duplicated
# ``data_block_utils._NUMBER_TOKEN_PATTERN`` while guarding nothing, which is exactly what
# the "second consumer" rule above exists to prevent. The three surviving numeric tokens
# are genuinely different and each has one caller: `data_block_utils` (block layer),
# `management_guidance._NUMBER_TOKEN_RE` (no leading ``+``), and
# `foreign_language_evidence._NUMBER_TOKEN_RE` (word/dot lookaround guards, so it does not
# match inside a version string). Sign tolerance across them is enforced behaviourally by
# tests/test_regex_consolidation.py, not by sharing one literal.

# Search-result envelopes emitted by ``tools/shared.py``. Five copies existed across two
# capture shapes; keep both here so a caller picks a shape rather than re-authoring one.
RESULT_ENVELOPE_RE = re.compile(r"(?is)<result\b[^>]*>.*?</result>")
RESULT_ENVELOPE_BODY_RE = re.compile(r"(?is)<result\b[^>]*>(.*?)</result>")

# Hostname syntax. Paired with `is_safe_public_host` below -- prefer the function.
_HOSTNAME_RE = re.compile(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?")


def is_safe_public_host(host: str) -> bool:
    """Return True when ``host`` is a syntactically valid, non-IP public hostname.

    Consolidates a guard that was duplicated verbatim -- pattern, the three-clause
    length/dot/shape check, and the ``ipaddress`` rejection -- in
    ``runtime_services.register_url`` and ``tools/official_documents``. Only the failure
    *action* differed between them (return False vs. log-and-skip), so callers keep their
    own handling and share the policy.

    Rejects a bare label ("localhost"), an over-long name, anything outside the
    LDH character set, and any literal IP address (v4 or v6) -- an IP bypasses
    host-based trust decisions.

    Callers are responsible for normalizing case and the trailing dot first.
    """
    if not host or "." not in host or len(host) > 253:
        return False
    if not _HOSTNAME_RE.fullmatch(host):
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return True
    return False
