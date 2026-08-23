---
name: diagnose-run
description: >
  Investigate a pipeline run or a suspected defect in this system, using saved
  analysis artifacts. Use before rating any defect, when a verdict looks wrong, or
  when a batch degrades. Establishes incidence from real data first.
---

# Diagnosing a run

## Measure before you rate

**Do not rate a defect you have not measured.** Nearly every question about this
system is answerable in under a minute against the saved artifacts, and the answer
routinely inverts the priority order: about half the plausible-looking findings in
this repo's history turned out to be inert, and half the real ones were invisible from
reading the source.

```bash
grep -rl 'SOME_FLAG' results/ | wc -l          # how often does this actually fire?
ls results/*_analysis.json | wc -l             # corpus size
```

`results/` is operator-local: it accumulates as you run analyses, and a fresh clone
has none. Every command here assumes you have run the pipeline at least once.

Report incidence as a fraction of the corpus, not as an anecdote. "Fires on 6 of 12
tickers" and "fires on 2 of 4,621 artifacts" call for opposite responses.

## What a saved artifact carries

`results/*_analysis.json` is the richest persisted payload. The fields worth reaching
for first:

| Field | Answers |
|---|---|
| `run_summary` | verdict, debate rounds, optional failures, model provenance, cost |
| `artifact_statuses.<field>` | did this node produce a usable artifact, and if not, `error_kind` |
| `red_flags` | the persisted flag ledger — authoritative, unlike re-derived prose |
| `analysis_snapshot`, `decision_trace` | the provenance contract; a rejected one blocks publication |
| `token_usage.by_tier` / `by_model` / `by_provider` | cost, and whether flex was actually served |
| `prediction_snapshot` | what the run predicted, for later outcome comparison |
| `prompts_metadata` | which prompt versions produced this |

Read the ledger, not the prose. A flag re-derived from a report is the thing the
provenance work exists to prevent; the persisted `red_flags` list is what the run
actually recorded.

## Batch-level tools

```bash
poetry run python scripts/eval_longitudinal_compare.py --run-log <path>
poetry run python scripts/cost_report.py --baseline <dir> --candidate <dir>
```

`eval_longitudinal_compare.py` breaks DNS failures out by operation and host and
groups classified failures by provider — use it so a genuinely new network dependency
does not hide inside a familiar-looking one. Pass the **analysis** log, not a file you
teed the wrapper script into: the latter contains only the wrapper's own echo lines
and no event records, so scanning it finds nothing by construction.

`cost_report.py` is read-only and diffs two run sets, which is how you attribute a
cost change to a lever rather than to noise.

## Attributing a change

Verdicts flip between BUY and HOLD run-to-run on identical data, so a verdict-level
A/B is noise-dominated. Attribute changes through the data layer — which source won a
field, which flags fired, which artifacts failed — and use repeats.

Before concluding that a code change caused a slowdown, compare `token_usage.by_tier`
against a prior run: vendor queueing dominates wall-clock time here.

## Degradation is quiet by design

An optional cross-check seat failing is publishable, so a run can be reported `OK`
while producing no cross-check at all. That makes the stock look *safer*, because the
flags that seat would have raised simply never appear. Check `run_summary` for
optional failures before comparing risk tallies across runs.

Full history: `docs/PROVENANCE.md`.
