---
paths:
  - "src/charts/**"
  - "src/thesis_visualizer.py"
---

# Charts

## Charts are generated after the verdict, deliberately

Chart generation runs **after** the Portfolio Manager, reading the PM block and
falling back to the fundamentals block only when it is missing. The reason is
historical and worth keeping: charts previously came from raw fundamentals *before*
risk penalties were applied, so a rejected stock got a chart showing upside. A chart
must reflect the decision that was actually reached.

Consequences that must survive any change here:

- The "our target" range is **suppressed entirely** for a do-not-initiate verdict.
- The target range is discounted by risk zone.
- Health and growth come from the PM-adjusted scores when available.

Chart generation is pure Python. It calls no model, so it must not acquire one.

## Charts require an output path

Without a file destination, relative image paths cannot resolve, so charts are
disabled. That is by design; a run that "lost" its charts almost always ran without
one.

Restrict any image manifest to the current run's chart paths when that mapping is
non-empty. Globbing the image directory lets a chart the run deliberately suppressed
reappear from a previous run of the same ticker.

## Parsing numbers out of blocks

A field that can be negative needs a sign-capable pattern. **Grep for an existing
canonical parser before writing a second regex** — a tally that subtracts bonuses was
parsed unsigned by a duplicated pattern and silently dropped its value on every
artifact that had one.

Compliance visuals follow the PM's stated gate verdict, not a re-derived comparison
against a threshold. The two legitimately disagree — a documented exception can pass a
sub-threshold score — and when they do, the label must say the verdict is the source.

Full history: `docs/AGENT_ROSTER.md`.
