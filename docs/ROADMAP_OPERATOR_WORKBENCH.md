# Operator Workbench Roadmap

This roadmap turns the repo from a strong local CLI system into a reliable, schedulable, single-operator workbench with durable state, evaluation gates, and clear trust boundaries.

## Summary

The implementation sequence is:

1. Freeze the runtime model
2. Introduce a unified local storage layer
3. Normalize config and startup validation
4. Add a coherent operator command family
5. Persist pipeline lifecycle
6. Make evals a real gate
7. Add scheduling, notifications, and observability
8. Formalize service-mode security boundaries
9. Package for cloud portability later

## Stage 0

Stage 0 is the design freeze for runtime/control-plane state.

- Runtime model document: [docs/RUNTIME_MODEL.md](/Users/richard3/Documents/Git-Repositories/investment-agent-public/docs/RUNTIME_MODEL.md)
- Goal: define canonical entities, identifiers, status enums, path policy, migration inputs, and producer/consumer ownership before storage work starts
- Output: no runtime code changes, only a decision-complete design contract for Stage 1

## Later Stages

- Stage 1: unified SQLite storage for analyses, pipeline runs, refresh jobs, notifications, and metrics
  - Detailed implementation spec: [docs/STAGE1_UNIFIED_STORAGE_PLAN.md](/Users/richard3/Documents/Git-Repositories/investment-agent-public/docs/STAGE1_UNIFIED_STORAGE_PLAN.md)
- Stage 2: config normalization and fast health checks
- Stage 3: operator command family
- Stage 4: DB-backed pipeline lifecycle
- Stage 5: evaluation gate
- Stage 6: scheduling and notifications
- Stage 7: observability and run history
- Stage 8: security boundary formalization
- Stage 9: cloud-portable packaging

## Current Focus

The current implementation focus is Stage 0. Later stages should use [docs/RUNTIME_MODEL.md](/Users/richard3/Documents/Git-Repositories/investment-agent-public/docs/RUNTIME_MODEL.md) as the source of truth for runtime entity design.
