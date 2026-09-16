---
paths:
  - "prompts/*.json"
  - "src/thesis_constants.py"
  - "tests/prompts/**"
---

# Prompt contract discipline

## Editing the JSON

Never write `\uXXXX` escapes — use the literal character (`≤`, `×`, `—`). Edit via
Python's `json` module with `ensure_ascii=False`; never `sed`, `awk`, `echo >`, or a
heredoc. The `system_message` strings run to tens of thousands of characters and no
line-oriented tool edits them reliably.

Parse with `json.loads()` before **and** after, then diff the resulting strings to
confirm only the intended change landed — not whitespace, escape sequences, or field
order.

## Keeping code and prose in agreement

A prompt teaches a format; code parses that format. Nothing detects silent drift
between them except the harness, so after changing any output template — block
markers, field labels, enum tokens, verdict headers — run `make test-prompts`:

- **L0** asserts prompt prose is consistent with `src/thesis_constants.py` and the
  parser enum tuples. Thresholds live in the constants module and prose must never
  restate a number the constants own.
- **L1** materializes each documented block from the `PROMPT_CONTRACTS` registry and
  feeds it to its real parser. Resolve prompt text with `prompt_text(agent_key)`, the
  on-disk canonical form — not `get_prompt()`, which env or Langfuse can mutate.

Bump the `version` field and add a `changes` line to `metadata` in the same edit.

## Fields the model must not author

A field the code *derives* must never be requested from a model: on the paths where
derivation fails, a model-authored value satisfies the validator and suppresses the
fallback, converting absent evidence into apparent positive evidence. Give every
required field a code-owned producer instead.

Full history: `docs/PROMPT_CONTRACTS.md`.
