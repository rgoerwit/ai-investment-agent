---
name: run-pipeline
description: Run, resume, audit, or interpret the repository's two-stage equity-screening pipeline, including Stage-1 quick BUY lists, Stage-2 full reports, reuse, and failures.
---

# Run or interpret the screening pipeline

`scripts/run_pipeline.sh` owns the two stages: quick Stage 1 writes the dated
BUY list; full Stage 2 consumes it. Same-mode reuse means quick reports never
satisfy Stage 2, and reconciliation treats quick BUYs as non-investable.

1. Read the relevant script paths first; confirm start stage, date, freshness,
   strictness, and forced reuse behavior.
2. Report Stage-1 BUY-list count separately from Stage-2 attempted, reused,
   completed, and failed work. `results/.pipeline_last_run.json` records only
   the Stage-1 BUY-list count.
3. Verify state from the dated BUY list, full reports (no `_quick` suffix),
   stage summary, and any failure/timeout records in `scratch/`.
4. Require a current, publishable full report for a purchase. On interruption,
   state the verified status and resume the appropriate stage.

When changing this workflow, inspect reuse, completeness, watchdog, failure,
and marker paths together, then run focused pipeline tests and static checks.
