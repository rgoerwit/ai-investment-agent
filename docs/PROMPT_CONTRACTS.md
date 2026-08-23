# Prompt contracts and the drift harness

Last updated: 2026-08-22

This file records *why* the prompt-editing rules exist and what happened when they
were absent. It explains the rationale and intended behaviour; the implementation and
its tests — chiefly `tests/prompts/` and the drift harness — are authoritative. If
this file and the repo disagree, trust the repo.

## The problem this system has

Agent behaviour is specified in natural language and consumed by deterministic code.
A prompt tells a model to emit `VERDICT: [BUY/HOLD/DO_NOT_INITIATE]`; a parser
elsewhere matches a tuple of accepted tokens. The two are written months apart, by
different reasoning, and nothing links them. When they diverge the system does not
crash — it quietly stops producing a field, and the absence looks like a legitimate
negative result.

Three failures of exactly this shape are on record:

- **Trader vocabulary drift.** `prompts/trader.json` v4.3 retired `SELL`/`REJECT` in
  favour of `ACTION: [BUY/HOLD/DO_NOT_INITIATE]`, but `check_trade_block_present`
  still required the old set, failing Stage 2 on *every* ticker whose Trader ran
  correctly. The L1 round-trip did not catch it because the contract declared
  `required_fields=()` and, as an unfenced block, skipped the `line_pattern`
  assertion — so the test always fed `ACTION: BUY`, the first enum member, and passed
  regardless of the rest of the vocabulary.
- **An advertised key nobody read.** `prompts/legal_counsel.json` documented the
  output key `other_legal_risks` while every consumer read
  `other_regulatory_risks`. A schema-compliant response therefore discarded every
  sanctions-adjacent finding: zero `REGULATORY_*` flags across 4,621 artifacts.
- **A required field with no producer.** `GUIDANCE_BRIDGE_STATUS` became
  unconditionally required in the Senior fundamentals block while appearing in no
  prompt at all — it is derived, not reported. On the paths where derivation produced
  nothing, the whole fundamentals artifact failed closed and the analysis was
  discarded.

## The four layers

Only the first two run in `make test-prompts`; they are the ones that must pass before
a prompt change is committed.

**L0 — static parity.** Prompt prose is compared against `src/thesis_constants.py` and
against the parser enum tuples. The rule behind it: a threshold has exactly one home,
the constants module, and prose may reference it but never restate the number. A test
that owns a *copy* of the contract does not guard the contract — it can drift with the
code and stay green.

**L1 — contract round-trip.** Each entry in the `PROMPT_CONTRACTS` registry
(`src/eval/prompt_contracts.py`) is materialized into its documented block and fed to
the real parser. Blocks are located with `extract_last_fenced_block`, never by
matching literal markers. Prompt text is resolved with `prompt_text(agent_key)`, the
on-disk canonical form; `get_prompt()` is wrong here because environment overrides and
Langfuse can replace it.

**L2 — deterministic replay** (`make replay`). Frozen captured outputs under
`tests/fixtures/frozen/` are pushed through the pure consumers, checking golden values
and verdict-independent invariants. No model is called.

**L3 — semantic judge** (`make eval-semantic`). A rubric comparison over the
suite manifests in `evals/prompt_check_suites/`. This one costs real LLM spend and is
manual or nightly, never part of `pytest`.

## Editing rules, and why each exists

**No `\uXXXX` escapes.** The strings are long, multilingual, and full of typographic
characters. An escape survives a round trip but makes the next diff unreadable, and
mixed conventions in one file guarantee someone eventually hand-edits one wrongly.
Write the literal character and let `ensure_ascii=False` preserve it.

**Python's `json` module only.** No `sed`, `awk`, `echo >`, or heredoc. These are
single strings of tens of thousands of characters spanning many lines; line-oriented
tools cannot address them safely. Where zsh history expansion is active — the
interactive default — there is a second reason: `!` is escaped, so a heredoc silently
converts `!=` into `\!=`. The rule above holds on every shell regardless; this is an
additional way to lose, not the only one.

**Parse and diff on both sides of the edit.** `json.loads()` before, `json.loads()`
after, then compare the strings. This catches the whole class of accidents — reordered
keys, lost escapes, a stray newline in a block marker — that are invisible in review
but fatal to a parser.

**Bump `version` and add a `changes` line.** The metadata block is the only record of
why a prompt says what it says, and several rules in the current prompts exist to
prevent a specific recurring model behaviour.

## The rule that generalizes

**A field the code derives must never be requested from a model.** On exactly the
paths where derivation fails, a model-authored value satisfies the validator and
suppresses the conservative fallback — converting absent evidence into apparent
positive evidence. Every required field needs a guaranteed code-owned producer; the
prompt may be told to copy a derived field, but only with its legal tokens spelled
out, so that naming the field cannot swap one silent failure for another.
