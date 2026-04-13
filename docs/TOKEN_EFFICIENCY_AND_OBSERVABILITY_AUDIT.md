# Token Efficiency & Observability Audit

*Audited: 2026-04-12. Live analysis run was in progress during audit (no --trace-langfuse).*

---

## Part 1: Token Efficiency

### Summary

The architecture is sound. The waste is mechanical — message accumulation, repeated context injection, and redundant system prompt boilerplate — not structural. Conservative estimate: **35–60K tokens wasted per ticker**. At 100 tickers/month that is 3.5–10M tokens (~$10–30/month at current Gemini pricing), growing linearly with batch size.

---

### Issue 1 — Message history snowball (HIGH)

**Location:** `src/agents/analyst_nodes.py:228-243`, `src/agents/state.py`

Every agent receives the full `state["messages"]` list with no pruning. By the time Portfolio Manager runs it sees ~20 prior agent outputs (70–150KB of accumulated text). `filter_messages_for_gemini()` only strips Gemini-incompatible formats; it does not bound the history. There is no trimming anywhere in the graph.

**Fix:** Replace `MessagesState`'s unbounded `add_messages` reducer with a capped version (keep last 7–10 messages) in `src/agents/state.py`.

---

### Issue 2 — System prompt redundancy (~40KB / ~15–20K tokens) (HIGH)

**Location:** `prompts/*.json` across 21 agents

Total system prompt text is ~219KB. ~40KB (~18%) is redundant boilerplate repeated across 8–15 agents:

| Repeated block | Agents affected | Est. chars |
|---|---|---|
| Ex-US equity context | 12 agents | ~20KB |
| "CRITICAL" / "IMPORTANT" warnings | 15 agents | ~12KB |
| Hard-fail thesis criteria | 7 agents | ~5KB |
| Analyst coverage threshold (<15) | 8 agents | ~3.2KB |

Individual prompt sizes: Fundamentals Analyst 25.7KB, Portfolio Manager 22.9KB, Consultant 20.2KB (largely duplicates PM), Auditor 16KB, News Analyst 14.9KB.

**Fix:** Create a shared template dict loaded at prompt-load time in `src/prompts.py`. Replace repeated blocks with `{INSERT_EX_US_CONTEXT}` etc. Makes thesis changes a one-line edit.

---

### Issue 3 — Analyst reports re-injected in Round 2 debate (MEDIUM)

**Location:** `src/agents/research_nodes.py:92-108`

Round-2 Bull and Bear researchers receive the same ~10KB of analyst summaries they already had in Round 1. Only the `debate_history` block is new. Total redundant injection per Round 2 pair: ~20KB.

**Fix:** For Round 2, inject only `debate_history`. The reports are already in the message context from Round 1.

---

### Issue 4 — DATA_BLOCK parsed and re-injected 4+ times (MEDIUM)

**Location:** `src/agents/consultant_nodes.py:344`, `src/agents/research_nodes.py:89,272`, `src/agents/decision_nodes.py`

The DATA_BLOCK (3–8KB) from Fundamentals Analyst flows through:
1. Summarized for Consultant (5KB max, `summarize_for_pm()`)
2. Summarized again for Research Manager (6KB max)
3. Parsed by Red Flag Detector (full text scan)
4. Re-indexed in Portfolio Manager summary table

It is parsed 3–4 times by different components, each re-scanning for the same `KEY: VALUE` patterns.

**Fix:** After Fundamentals Analyst runs, parse DATA_BLOCK to a `data_block_dict` and store it in state. Downstream agents read the dict; no re-parsing needed.

---

### Issue 5 — Extra context accumulation in Fundamentals Analyst (MEDIUM)

**Location:** `src/agents/analyst_nodes.py:258-346`

Fundamentals Analyst receives per invocation:
- System message: 25.7KB
- Full message history: 30–100KB
- Raw fundamentals from Junior: 5–15KB
- Foreign language report: 2–10KB
- News highlights: 1–5KB
- Data conflicts report: 1–3KB
- Legal report: 2–5KB
- Macro events (if applicable): 0.5–2KB

**Total: ~75–180KB per invocation.** This agent warrants large context, but the message history component is the part to fix (see Issue 1).

---

### Issue 6 — Research Manager receives all reports again (HIGH)

**Location:** `src/agents/research_nodes.py:269-279`

Research Manager re-summarizes all analyst reports (~18KB) that Bull/Bear already had. Combined with message history this is 60–130KB per invocation.

---

### Issue 7 — No input token counting (MEDIUM)

**Location:** `src/agents/runtime.py:86-115`

`invoke_with_rate_limit_handling()` extracts output tokens from the LLM response but does not count input tokens. The `token_tracker.py` module exists but is not wired into `runtime.py`. Without production input token counts, optimization is flying blind.

**Fix:** Count input tokens in `runtime.py` and record via `TokenTracker`.

---

## Part 2: Langfuse Observability

### What Langfuse can do for you

A quick primer:

- **Traces & spans**: A *trace* is one complete analysis run. *Spans* are the steps inside it (each agent, each tool call). Right now you have traces but the spans are coarse — graph nodes only.
- **Token cost dashboard**: Once usage flows through, Langfuse gives per-agent cost breakdowns automatically.
- **Scores**: Call `langfuse.score(trace_id=..., name="verdict", value=1.0)` after the PM decision. Over time this builds a dataset you can correlate with actual market outcomes — which is what the retrospective system does locally, but in a queryable UI.
- **Prompt versioning**: Langfuse has a prompt management feature. You could push `prompts/*.json` system messages there and track which version produced which outcome.
- **User-defined evaluators**: Attach an LLM-as-judge evaluator in the Langfuse UI that runs on every trace — no code needed. `src/eval/semantic_judge.py` could be registered there instead of run separately.

---

### Current state

| Component | Status | Notes |
|---|---|---|
| Basic setup & config | Working | `src/config.py:557-594`, `src/observability.py` |
| CLI flag `--trace-langfuse` | Working | `src/main.py:269-275` |
| Graph-level callback injection | Working | `src/main.py:1376-1419`, injected into `graph.ainvoke()` |
| Graph node tracing | Working | All nodes produce traces via LangGraph integration |
| LLM call tracing | Implicit | CallbackHandler receives calls through graph; no explicit per-call spans |
| Tool execution spans | **Missing** | `src/tooling/runtime.py` — no Langfuse spans |
| Token usage → Langfuse | **Missing** | Tracked locally in `token_tracker.py`, never forwarded |
| Article writer tracing | **Missing** | `src/article_writer.py` calls `llm.invoke()` post-graph, no callbacks |
| PM verdict as score | **Missing** | `langfuse.score()` never called anywhere |
| Explicit root trace | **Missing** | Relying on LangGraph's implicit span; UI hierarchy may be fragmented |
| Eval/baseline integration | **Missing** | `evals/captures/schema_v3/` is entirely separate, no shared trace IDs |
| Error handling | Working | Graceful degradation — Langfuse failure never crashes analysis |

---

### Gap 1 — Token usage not sent to Langfuse (HIGH impact)

**Location:** `src/agents/runtime.py:86-115`

Token counts are extracted from LLM responses and recorded in the local baseline capture system, but never forwarded to Langfuse. You cannot see cost per agent in the Langfuse UI.

**Fix:** In `runtime.py`, after `_extract_token_usage(result)`, forward counts through LangChain's standard `usage_metadata` or via an explicit `langfuse.generation()` call.

---

### Gap 2 — No PM verdict score (HIGH impact, trivial to add)

**Location:** `src/main.py` (post-graph result extraction)

`langfuse.score()` is never called. Portfolio Manager verdict (BUY/SELL/DNI), pre-screening result (PASS/REJECT), and debate conviction levels are captured locally but invisible to Langfuse.

**Fix:** After `graph.ainvoke()` completes, extract verdict from state and call:
```python
langfuse.score(trace_id=session_id, name="pm_verdict", value=..., comment=ticker)
```

---

### Gap 3 — No explicit root trace (MEDIUM)

**Location:** `src/main.py:1376-1419`

There is no `langfuse.trace()` call wrapping the full run. The hierarchy in the Langfuse UI depends on LangGraph's implicit top-level span, which can produce fragmented views.

**Fix:** Wrap the `graph.ainvoke()` call in an explicit `langfuse.trace(name=ticker, session_id=session_id)` context.

---

### Gap 4 — Tool execution not spanned (MEDIUM)

**Location:** `src/tooling/runtime.py:73-134`

`ToolExecutionService.execute()` runs tool invocations with no Langfuse spans. Tool calls appear in LLM response content, not as first-class observable events. Tool latency and failure rates are invisible.

**Fix:** The hook infrastructure already exists in `src/tooling/runtime.py`. Add a Langfuse-aware `ToolHook` that opens/closes a span around each tool execution.

---

### Gap 5 — Article writer is invisible (LOW-MEDIUM)

**Location:** `src/article_writer.py:499-599`

`self.llm.invoke(messages)` is called post-graph without callbacks. These OpenAI/Claude calls (often the most expensive in cost-per-token terms) do not appear in Langfuse at all.

**Fix:** Accept an optional `callbacks` parameter in `ArticleWriter.__init__()` and pass it through to `llm.invoke()`.

---

## Prioritized Action List

### Tier 1 — Mechanical, high payoff, low risk

1. **Bounded message reducer** — `src/agents/state.py`. Keep last 7–10 messages. Single change, kills the snowball.
2. **Token usage → Langfuse** — `src/agents/runtime.py`. Forward extracted token counts. Unlocks the cost dashboard.
3. **PM verdict as Langfuse score** — `src/main.py`. One `langfuse.score()` call after graph completes. Permanent observability benefit.

### Tier 2 — Structural, medium effort

4. **Stop re-injecting analyst reports in R2 debate** — `src/agents/research_nodes.py`. Pass only `debate_history` for Round 2.
5. **Deduplicate system prompt boilerplate** — `src/prompts.py` + `prompts/*.json`. Shared template dict, ~40KB savings.
6. **Explicit root Langfuse trace** — `src/main.py`. Clean top-level entry per analysis run in the UI.

### Tier 3 — Architectural

7. **Tool execution spans** — `src/tooling/runtime.py`. Langfuse-aware `ToolHook`. Hook infrastructure already present.
8. **Article writer callbacks** — `src/article_writer.py`. Pass callbacks into `llm.invoke()`.
9. **Parse DATA_BLOCK to JSON once** — Store `data_block_dict` in state; downstream agents read the dict.

---

## Enabling Langfuse on a run

```bash
poetry run python -m src.main --ticker 0005.HK --trace-langfuse --output results/0005.HK.md
```

Required env vars in `.env`:
```
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com  # or self-hosted
```
