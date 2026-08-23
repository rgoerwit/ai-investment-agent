---
paths:
  - "src/eval/**"
  - "evals/**"
  - "src/runtime_diagnostics/**"
---

# Evals, capture, and the publication contract

## Publication fails closed

A live run is stamped with the provenance contract version, so a **missing** snapshot
or decision trace is a failure rather than a silent pass. Legacy artifacts carry no
stamp and are grandfathered — present-but-invalid checks only. When adding a test
result that reaches persistence, give it a valid snapshot and trace or it is correctly
non-publishable.

Decoding is fail-closed and **type-only**: a future schema version, a type-corrupt
field, a non-finite float, a non-bool where a bool is required, or a wrong-typed
collection all raise rather than defaulting. Payload *semantics* stay with their
producers — re-deriving a business rule inside a wire codec couples them and they
drift.

## Capture acceptance follows the publication contract

An artifact the contract declares optional must not reject a capture. The load-bearing
site is the **node-level** check, not finalization: rejection latches, so exempting
only at the end is dead code. Both levels read the same source so they cannot
disagree.

Two adjacent doors have to stay shut for the exemption to hold: a generic error-marker
scan must skip the failed optional artifact's own subtree while still invalidating
unrelated markers, and accepting the run must not promote a failed seat's output to
replay-eligible — otherwise an empty output becomes the reference that good output is
later judged against.

A test that injects failed statuses at finalization **cannot** catch this. The guard
has to wrap a genuinely failing optional node.

## The layered harness

L0 static parity and L1 contract round-trip run in `make test-prompts` with no model.
L2 replays frozen fixtures. L3 is the semantic judge and costs real spend — manual or
nightly, never in the default test run.

The optional-node set is **derived** from the publication contract, not hand-listed: a
node is optional exactly when every artifact it owns is optional for publication.
Hand-maintaining that list is how switching off a supported configuration started
reporting an absent seat as a failure.

## Operator note

A dirty working tree rejects every capture bundle, so the sequence is always commit,
then capture. Capture bundles land in `evals/captures/`, which is operator-local and
absent from a fresh clone; the suite manifests under `evals/prompt_check_suites/` are
tracked.

Full history: `docs/PROVENANCE.md`.
