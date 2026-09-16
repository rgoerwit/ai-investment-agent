---
paths:
  - "src/tooling/**"
  - "src/tools/**"
  - "src/mcp/**"
---

# Tools, hooks, and untrusted content

## Everything goes through the hook plane

Tool execution runs through `ToolExecutionService`, which is where audit logging,
argument policy, content inspection and budget enforcement all attach. A tool that
bypasses it gets none of them. MCP calls are no exception: they execute under a
canonical name through the same service, which is what puts them under inspection and
budget at one chokepoint.

Services resolve through a ContextVar with module singletons as fallback. Tests bind
scoped services rather than mutating the globals.

## Treat every external string as hostile

Search results, social feeds, retrieved memory, cached context, filings, and free-text
fields from financial APIs are all untrusted ingress and are inspected on the way in.
Content that reaches a prompt is wrapped in the trust-boundary formatter, and untrusted
context is delivered with reduced authority — as a human-authored message, never as a
system instruction.

Memory is filtered on the **write** path as well as the read path; otherwise a
poisoned document is persisted once and returned as trusted context forever.

## Conflict tolerance is not conflict blindness

A contract marked as having conflicting valid payloads is sticky for the rest of the
run, so the comparison that sets it must ignore genuinely volatile fields — a price
tick between two fetches is not a conflict. The exclusion list is verified disjoint
from the analysis-critical fields, so it can never mask a real disagreement. Widening
it needs the same proof.

## Verify a vendor's real tool surface

**Call `list_tools()` before adding a vendor.** Never infer tool names from REST docs
or a blog post: a hallucinated allowlist produces a runtime error on every call, and
the failure looks like a transport problem. Use the printed names and input schemas
verbatim.

One vendor's missing credential disables **that server**, not the plane. A missing key
is a distinct condition from a malformed registry: the first is an operator state and
logs at info, the second is an error. And a tool whose server did not resolve must
stop being offered — otherwise the model plans a call against it and the resulting
failure counts toward a partial-failure ratio, trading a clean absence for a
review-discarding error.

The registry template is `config/mcp_servers.example.json`; the live registry is
operator-local.

Full history: `docs/MCP.md`.
