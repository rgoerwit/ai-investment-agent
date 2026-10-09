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
   For timeout incidents, inspect `src/llm_runtime/timeouts.py` and
   `src/llm_runtime/run_deadline.py`: separate request, invocation, node and ticker
   scopes, admission refusal, transport timeout and external watchdog termination.
   Inspect retained dispatch records for effective tier, remaining allowance and
   attempt limits; distinguish configured SDK limits from observed requests.
   Trace both canonical and legacy construction, plus direct writer/health callers.
   A timeout cluster alone does not establish provider capacity or physical retry
   counts. Check publication validity and recorded cost after failure or recovery.
6. Report cause, confidence, evidence, user impact, and the smallest plausible fix.

For operational timeouts, verify the allowance is resolved before request creation,
and that inspection remains bounded and fail-closed. Tool/data scopes read the
enclosing deadline without rebinding it around their judge. Embedding requests use
live configuration and the caller's allowance; the cached startup probe is a
separate scope and must not inherit a short caller deadline.

For MCP, verify timeout values reach the SDK's HTTP, session and tool-call
parameters. Check allowance again between initialization, catalog acquisition and
tool dispatch; a session's initial timeout is not the next operation's remaining
budget. Follow the spent-allowance error path outside the session body so it does
not destabilize the SDK's task-group cleanup.

Diagnosis alone does not authorize code edits, a paid rerun, a network probe, or a
credentialed integration test. Ask before taking those actions unless the user already
requested them explicitly.
