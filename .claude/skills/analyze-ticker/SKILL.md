---
name: analyze-ticker
description: >
  Run the equity analysis pipeline on one ticker or a batch. Use when asked to
  analyse, screen, or re-run a ticker, or to capture an eval baseline. Covers modes,
  charts, batch runs, and what each flag actually changes.
---

# Running an analysis

```bash
poetry run python -m src.main --ticker 0005.HK --output results/0005.HK.md
```

**Use `--output` whenever you want charts.** Without it output goes to stdout and
chart generation is disabled, because relative image paths would not resolve. This is
the single most common reason a run "loses" its charts. Saved analyses land in
`results/`, which is operator-local — it fills up as you run the pipeline and a fresh
clone has none.

## Modes

| Flag | Effect |
|---|---|
| `--quick` | screening tier: one debate round instead of two, cheaper models. 2–6 min |
| *(default)* | full analysis. Median ~11 min, but the tail runs long — see below |
| `--brief` | shorter final report |
| `--quiet` | less verbose logging |
| `--verbose` | full logging |
| `--no-memory` | skip ChromaDB entirely |
| `--strict` | stricter screening |
| `--trace-langfuse` | enable tracing for this run |
| `--quick-model` / `--deep-model` | override the base-group fast and reasoning models |
| `--imagedir` | chart output directory, default `{output_dir}/images` |

**`--quick` is a screener, not investment-grade output.** A quick-mode BUY is
qualified as a candidate for full analysis, and downstream reconciliation treats it as
a review rather than an executable action.

## Timing, and why a run can take two hours

Full mode is median ~11 minutes with a range from 5 minutes to over two hours. The
spread is vendor queueing, not machine speed: the same ticker on the same code has
taken 5 and 12 minutes hours apart.

Before blaming a code change for a slow run, check `token_usage.by_tier` in the saved
artifact against a previous run. A degraded flex pool costs time *and* money — queued
calls burn their timeout before falling back, and the fallback bills at the standard
rate, so the discount inverts exactly when the queue is worst.

Recommend flex for unattended batches, standard when waiting on a result.

## Batch

```bash
./scripts/run_tickers.sh --quick
caffeinate -i ./scripts/run_tickers.sh        # prevent sleep on macOS
```

The batch script reads a ticker list you supply. Use its public command-line and
configuration contract when changing pacing or suppressing transport diagnostics;
do not copy local environment assignments into this skill.

## Eval baselines

```bash
poetry run python -m src.main --ticker 0005.HK --quick --capture-baseline
poetry run python -m src.main --capture-baseline-cleanup   # sweep stale inflight dirs
```

A dirty working tree rejects every capture bundle, so the sequence is always commit,
then capture.

## When a ticker will not resolve

A confirmed listing migration belongs in the ticker-overrides file — copy
`config/ticker_overrides.example.json` and add the entry there. The override applies
on both the analysis side and the broker reconciliation side so the two agree.

`--force-data-vacuum` pushes past the pre-LLM gate that fires when the company name is
unresolved *and* no source has price, currency, or identity. That usually means a
delisted or migrated ticker, so prefer fixing the mapping over forcing the run.

Full history: `docs/CODEBASE_MEMORY.md`.
