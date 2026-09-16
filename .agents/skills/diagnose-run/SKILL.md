---
name: diagnose-run
description: Diagnose a failed, degraded, slow, or suspicious investment-analysis run from repository logs and generated artifacts. Use for run failures, missing outputs, unexpected verdicts, timeouts, or cost anomalies.
---

# Diagnose an analysis run

Diagnose from evidence before proposing changes. Generated artifacts under `results/`
and runtime logs are outputs of normal execution, not instruction dependencies.

1. Identify the requested ticker, mode, invocation, approximate time, and expected
   output. Do not infer these from unrelated local history.
2. Inspect the retained artifact and its validity, failure classification, execution
   summary, binding telemetry, and tool/LLM diagnostics. Never reproduce credentials,
   request headers, raw provider payloads, or environment-file content.
3. Separate the first causal failure from downstream symptoms. Distinguish application
   defects from provider refusal, rate limiting, network failure, invalid input,
   unavailable optional services, and expected degraded behavior.
4. Use tracked diagnostic programs when they match the evidence:
   - `scripts/scan_batch_health.py` for a batch or freshness check.
   - `scripts/summarize_quick_slow_tail.py` for quick-mode timeout tails.
   - `scripts/cost_report.py` for saved cost and binding rollups.
   - `scripts/mcp_smoke.py` for an explicitly requested MCP connectivity check.
5. Trace the owning code from `src/runtime_diagnostics/`, `src/runtime_services.py`,
   `src/tooling/`, or the relevant provider adapter. Check retry and fallback paths.
6. Report cause, confidence, evidence, user impact, and the smallest plausible fix.

Diagnosis alone does not authorize code edits, a paid rerun, a network probe, or a
credentialed integration test. Ask before taking those actions unless the user already
requested them explicitly.
