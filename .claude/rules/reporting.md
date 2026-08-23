---
paths:
  - "src/report_generator.py"
  - "src/reporting/**"
  - "src/article_writer.py"
  - "src/article_audit.py"
  - "src/editor_tools.py"
---

# Rendering reports and articles

## Cleanup must not delete the signal

The machine-readable resolution blocks the PM emits are often the **only** place a
cross-check reconciliation appears. Strip the machine label and unwrap the fence;
keep the content. Removing the whole block leaves an empty fence and loses the
reconciliation — a regression this repo has already shipped once.

Collapse any fence left genuinely empty, and rewrite an unresolved machine stub into
prose rather than showing the template to a reader.

## Qualify by banner, never by token rewriting

Where a claim needs softening — a coverage caveat, a data-quality note — prepend a
one-time banner to the section. **Do not rewrite tokens in place.** In-place
substitution splices a noun phrase into header, label, and predicate-adjective slots
indiscriminately and produced ungrammatical output across most of a batch.

## Follow the verdict, not a re-derivation

Compliance rows take the PM's gate verdict token, not a fresh comparison against a
threshold. A documented exception can legitimately pass a sub-threshold score, and
when the two disagree the row must say the verdict is the authority. Anchor
extraction to the gate line; a loose first-match pattern picks up a different metric
from a nearby bullet and renders a false failure.

## The article pipeline

Citations of hard fields are audited deterministically against the underlying data,
including un-backticked parentheticals, and a mismatch produces a caveat block rather
than a silent publish. Insert that block below the first heading, and make insertion
idempotent.

A claim with weak provenance asserted in language of certainty gets one caveat — the
conjunction of the two is the signal, not either alone. Broad prose number-matching
was tried and rejected: it misses the real case and fires on threshold language.

Deterministically force a revision when a reference whose fetch the editor watched
fail is still present in the draft. The prompt rule alone was ignored.

**Record which model actually wrote it.** A writer fallback between vendors is
otherwise visible only in a buried log line, and voice evaluations then measure the
wrong model. Fallback event names stay family-neutral, with the vendor in structured
fields.

Full history: `docs/AGENT_ROSTER.md`.
