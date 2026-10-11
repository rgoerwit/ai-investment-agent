# Codebase Memory

Last updated: 2026-10-11

This file is a durable orientation note, not the source of truth.
Use it to get context quickly, then verify against the live tree.
If this file and the repo disagree, trust the repo.

## Current Quality Baseline (August 2026)

- Ruff format and lint cover `src/`, `tests/`, and `scripts/`.
- `poetry run mypy src/` is a blocking gate and has no errors across the 235
  source files. The previous baseline was 52 errors across 31 files; fixes were
  made at owning boundaries without blanket ignores or retyping the intentionally
  shallow `AnalysisSnapshot` claims payload. The configured gate now includes
  `check_untyped_defs = true`, so typed errors inside legacy unannotated functions
  are checked. Full annotation completeness (`disallow_untyped_defs`) remains a
  separate migration: the 2026-08-05 baseline is 190 errors across 55 files.
- `make test-ci` is the deterministic, credential-free test surface. It excludes
  tests marked `slow`, `integration`, `llm`, or `api`.
- `make ci` runs installation, all static gates, and `test-ci`; `make ci-full`
  additionally runs the complete suite and may require external services.
- GitHub CI uses the same `check-all` and `test-ci` targets as local development.
- `SNYK-PYTHON-CRYPTOGRAPHY-18516621`, `CVE-2026-69247`, and
  `GHSA-g6cj-pr64-35w5` identify the same PKCS#7 EnvelopedData decryption
  finding; the JWT rationale in commit `9658029` is superseded. The repository
  does not call the affected `pkcs7_decrypt_*` APIs, but pins the fixed
  `cryptography==50.0.0`. Its seven-day release cooldown was explicitly waived
  on 2026-08-05 after applicability review and owner approval.

Legal/regulatory authority is ledger-first: report scoring consumes effective
red flags for PFIC, VIE, and CMIC before falling back to the Senior Fundamentals
DATA_BLOCK. Artifact validity is authoritative: an unavailable Legal Counsel
leaves PFIC/VIE/CMIC unassessed and emits one zero-penalty,
BUY-blocking `LEGAL_COUNSEL_UNAVAILABLE` coverage flag; provider failure is never
converted into a substantive legal finding. Retryability is owned by
`runtime_diagnostics.classify_failure()` / `FailureDetails.retryable`; the agent
runtime only retains a separate rate-limit branch because its backoff policy
differs. Consultant timeout floors use the inferred provider rather than assuming
OpenAI-compatible seats are OpenAI.

Model-response validity has two independent layers. `src/agents/runtime.py` owns
transient provider retries; the producing node then validates its output contract.
Canonical fenced blocks require their end marker and legacy fields match exact logical
keys, so `RAW_HEALTH_SCORE` cannot satisfy `HEALTH_SCORE`. Invalid responses are
classified as cap exhaustion, incomplete structure, or ordinary contract violation.
Structural recovery is text-only and bounded: full mode applies it to eligible analyst
outputs, while quick mode limits it to Foreign Language, Senior Fundamentals and Portfolio Manager.

## Comparing Analysis Quality Across Runs (October 2026)

**Pair by the latest quick-mode artifact per ticker per month.** Taking the latest
artifact of any mode and dropping non-quick ones silently removes every ticker that
went on to a full-mode stage-2 run, which is every quick BUY: a July/October
comparison done that way reported 0 July BUYs where there were 34.

**Adjacent months barely overlap.** The pipeline reuses a quick report for 60 days,
so September and October share 5 tickers, all re-runs of failures. Compare against a
month outside the reuse window (July for October: 427 pairs).

Findings from that comparison, attributed through the data layer rather than verdicts:

- **Guidance-gap recurrence (code).** `MANAGEMENT_GUIDANCE_EVIDENCE_GAP`
  (`blocks_buy`) fired on 52–62% of quick runs from August to October. Both required
  guidance searches had succeeded in every case; the demotion counted the optional
  filings adapter, which reports UNAVAILABLE on every exchange, so
  `NOT_DISCLOSED_AFTER_TARGETED_SEARCH` survived 0 of 449 runs that called it
  against 29 of 56 that did not. Fixed by `REQUIRED_GUIDANCE_SEARCHES`. It was
  present in 20 of the 31 July BUYs that were no longer BUY in October.
- **LATEST_RESULTS contract failures (reader, plus model formatting).** 29 of 512
  runs: complete blocks behind a bare heading, an END marker missing its dashes,
  or a URL with a trailing note. The FLA normaliser now re-frames an unbroken run of
  field lines; replaying the 29 saved outputs recovered 23 and changed none of the
  486 valid ones. The remaining 6 are genuinely incomplete. A separate replay of
  523 October 2–4 saved FLA reports found 501 valid latest-results blocks; the
  current normalizer repaired 17 of 22 invalid blocks without source records,
  leaving five invalid. Across 526 quick artifacts in that window, FLA call
  attempts recorded 687 successes, two timeouts,
  one provider safety block, and no output-cap failures; this sample does not
  support a general FLA token-exhaustion diagnosis.
- **Junior Fundamentals timeouts (configuration).** 2% in July, 13–18% on busy
  October days. Under `--quick` a flex attempt gets 0.4 x 60s = 24s, below
  Junior's own p90 of 26.5s, so long healthy calls were cut, re-issued at standard,
  and killed at the 60s cap. Junior is now standard-tier in quick mode.
- **Senior Fundamentals output-cap failures (model).** `gemini-3.8-flash` hit the
  cap on 8 of 523 quick Senior calls, `gemini-3.7-flash` on 0 of 522: thinking used
  15.6k–18.3k of 19,115 tokens against an 8,192 reserve. The structural retry
  raises reasoning effort, so it failed the same way; that is unfixed.

A two-arm trial produced 19 quick runs per arm across 17 tickers. Two pairs had
different prompt digests, leaving 17 prompt-matched runs per arm across 15 tickers
(including three repeats of 8174.T). With the code changes, Junior failed on none
of the 38 runs. In the prompt-matched subset, moving Senior from 3.8 to 3.7 coincided
with one recorded Senior output-cap failure on 3.8 and none on 3.7. Mean cost
fell from $0.135 to $0.103. Both arms used 3.8 for Portfolio Manager and the separately
bound structural-recovery seat; the saved binding does not record recovery effort.
All artifacts mark the code tree dirty, so matching commit and prompt digests do
not establish that non-prompt code was identical between arms.
Five prompt-matched ticker pairs had different recorded verdicts, including 2117.T,
where the 3.8 arm lacked a verdict after a cap failure. Some other flips involve
borderline growth scores or contested legal risk. The three-run 8174.T probe was
stable in both arms; the remaining single runs cannot rank model judgment.

## What This Repo Is

This is a multi-agent international equity analysis system built on LangGraph.
It combines:

- parallel analyst agents
- deterministic pre-screening via `RedFlagDetector`
- adversarial bull/bear debate
- markdown report generation with charts
- ticker-isolated ChromaDB memory plus lessons learned
- IBKR portfolio and reconciliation workflows

The system is no longer just “analyze one ticker.” It also supports:

- batch analysis
- retrospective learning
- article generation
- portfolio-aware recommendations and reconciliation

## Screening Pipeline Semantics

Stage 0 requires every enabled, applicable exchange source to succeed. Source
floors count unique normalized tickers; repeated pagination pages cannot satisfy
them. Source failure aborts before writing fresh screening output. BVB screening
uses its official Regulated Market equity table and validates the active market
selector; AeRO is excluded. Direct BVB HTTP availability remains an operational
dependency, even when its browser interface is reachable. No size threshold is
used as a substitute for exchange-segment membership.

`scripts/run_pipeline.sh` runs quick Stage 1 screening, then full Stage 2
analysis of its BUY list; same-mode reuse keeps quick reports from satisfying
Stage 2. The marker's `buy_count` is the Stage-1 list size, not Stage-2
completion. Verify full-run status from stage output and full reports; quick
BUYs are screening candidates, not investable signals.

### The watchdog contract (October 2026)

Each Stage-1 and Stage-2 child runs under a per-ticker watchdog
(`run_tracked_child` in `scripts/pipeline_signals.sh`). Before this, the child
did not know the watchdog existed. Every in-process budget assumed unlimited
time, so a retry ladder near the end kept going until SIGTERM discarded the
whole ticker. In full mode with flex on, one Portfolio Manager call can use
2 × 1,350 s, which is more than the 2,400 s Stage-2 watchdog on its own.

- **One source of truth.** `STAGE1_TICKER_TIMEOUT_SECONDS` (600) and
  `STAGE2_TICKER_TIMEOUT_SECONDS` (2400) are `Settings` fields. The script reads
  them with `python -m src.config_print <key>`, which exits non-zero on an
  unknown key and refuses any field that is not a number or a boolean. An
  exported value still wins and skips Python. A value that is not a positive
  whole number stops the pipeline before Stage 0.
- **An absolute deadline, not a duration.** `run_tracked_child` sets
  `PIPELINE_TICKER_DEADLINE_EPOCH = start + timeout` when it starts the child.
  `PIPELINE_TICKER_TIMEOUT_SECONDS` stays as the watchdog's own input, and the
  child never reads it. `src/llm_runtime/run_deadline.py` converts the epoch to a
  monotonic deadline once, minus `PUBLICATION_RESERVE_SECONDS` (30). A missing or
  malformed epoch means there is no run deadline. It is logged once and never
  raises, so a manual `src.main` run is unchanged.
- **Where it binds.** `src/main.py` binds it with `enclosing_deadline_at` around
  the ticker retrospective and around `_execute_analysis` (pre-graph fetches,
  macro brief and the graph). It does not cover the report-header lookup, render,
  persistence, rejection record, article or shutdown. Those run on the reserve.
  Every node scope inside the bound region becomes `min(node, run)`: the APAC
  node's 240 s, a caller's `overall_timeout_seconds`, and every operational
  clamp described under Content-ingress hardening, which had been inert until now. LLM admission refuses a
  request that cannot fit, and the retry loop stops a retry or backoff that
  would cross the deadline. If the epoch has already passed at start, the first
  LLM call is refused before any request is sent. The gate-critical seats then
  fail, and the child exits with an `ANALYSIS FAILED` report, so the shell
  records `no_analysis` and does not have to kill the child for a `child_exit_124`.
- **The gap between the deadline and the kill.** The loop polls every
  `PIPELINE_TICKER_WATCHDOG_POLL_SECONDS` (5). Its counter adds the poll interval
  each pass, so it under-counts real time. SIGUSR1, which asks for a
  pending-task dump and does not stop the child, therefore arrives no earlier
  than the epoch. `date +%s` floors, so the epoch is never late. SIGTERM comes
  `PIPELINE_TICKER_DUMP_GRACE_SECONDS` (5) after SIGUSR1, and SIGKILL comes
  `PIPELINE_TICKER_TERM_GRACE_SECONDS` (10) after SIGTERM. Analysis stops
  admitting work 30 s before the epoch, so publication has at least
  reserve + dump grace = 35 s, plus up to one poll interval of slack. A
  watchdog kill still writes its JSONL line to `PIPELINE_TIMEOUT_RECORD_FILE`
  and is recorded as `child_exit_124`.
- **The injection judge gets no carve-out.** Operational scopes deliberately
  never bind a deadline. A 45 s tool scope would refuse the judge
  while the agent that reads the content still had time. The judge fails open,
  so that would turn a fail-closed timeout into uninspected content. The run
  deadline is different because it is the outermost deadline. When it refuses
  the judge, `EscalatingInspector` keeps the heuristic decision, and every later
  LLM consumer is refused by the same deadline. Each class uses the same 5 s
  minimum useful attempt and 2 s margin once its cap is at least 40 s.
  `tests/llm_runtime/test_run_deadline.py` pins this.
- **The startup diagnostic.** `log_deadline_ladder` works out the critical path
  from each seat's resolved `BudgetPolicy`. It takes the slowest parallel analyst
  branch (Junior then Senior), then the debate rounds, the Research Manager,
  the Trader, the slowest risk analyst and the PM. Each step counts at
  `cap × LLM_ATTEMPTS`. If that total is more than the time left, it logs
  `deadline_ladder_truncated` once at INFO, with both the single-attempt and
  the full-retry totals. Tool loops and optional seats are left out, so the
  total is a lower bound. It shows the ceiling and does not promise the run
  completes. With shipped defaults, quick mode needs 1,320 s with retries and
  660 s with one attempt each, against about 570 s, so it always logs. Full
  mode on the standard tier needs 1,920 s against about 2,370 s and does not
  log. Full mode with Gemini flex needs 21,600 s.
- **No literal deadlines.** `tests/test_timeout_literals.py` is an AST guard. It
  flags a numeric literal that reaches a deadline parameter outside the config
  owners. The literal can arrive through a constant, a local, `or`, `min` or
  `+`. Allowlist entries are keyed by enclosing scope and parameter, not by line
  number, and an entry the scan no longer finds fails the test. The APAC 240 s
  client ceiling had two copies (`src/llms.py` and the compatible adapter); it
  became a named constant and is now derived from
  `APAC_NODE_TOTAL_TIMEOUT_SECONDS` (`apac_client_timeout_seconds`). The
  Consultant factory's 120 s default became a constant and is now the
  `OPENAI_REQUEST_CAP_SECONDS` setting, read at the call.

## Fast Orientation

Read in this order:

1. `README.md`
2. top of `CHANGELOG.md`
3. `src/main.py`
4. `src/cli.py`
5. `src/persistence.py`
6. `src/output.py`
7. `src/runtime_services.py`
8. `src/tooling/`
9. `src/graph/`
10. `src/agents/`
11. `src/tools/`
12. `src/data/fetcher.py`
13. `src/runtime_diagnostics/` (package: `failure_classification` + `artifact_status`)
14. `src/validators/red_flag_detector.py`
15. `src/validators/sector_classifier.py`
16. `src/validators/metric_extractor.py`
17. `src/validators/financial_rules.py`
18. `src/validators/supplemental_extractors.py`
19. `src/validators/supplemental_flags.py`
20. `src/memory.py`
21. `src/ibkr/`

## Runtime Spine

`src/main.py` is now orchestration-first: runtime setup, macro-context prefetch, graph execution, tracing, and mode dispatch.
`src/cli.py` owns CLI parsing, validation, and output/article path resolution.
`src/persistence.py` owns saved-artifact assembly, JSON persistence, and rejection-record helpers.
`src/output.py` owns banners, CLI/report rendering, and article generation helpers.

For runtime/control-plane state design, use `docs/RUNTIME_MODEL.md` as the canonical Stage 0 model before changing storage or orchestration seams.

`src/runtime_services.py` owns runtime-scoped service binding.
`RuntimeServices` uses `ContextVar` scoping so CLI runs, graph execution, dashboard snapshot loads, and worker jobs can bind their own tool execution, inspection, provider runtimes, and hooks without sharing mutable globals by accident.
Provider runtimes may live for a process, but evidence recorders and issuer-authority
registries are analysis-run scoped; long-lived workers rebuild those services per ticker.

`src/graph/` owns:

- routing and sync barriers
- graph component construction
- graph wiring
- graph-scoped per-agent tool-node filtering

`src/agents/` owns the node logic:

- analyst nodes
- fundamentals and validator nodes
- research/debate nodes
- PM/trader/risk nodes
- consultant/legal/auditor nodes

`src/tools/` holds the domain tool implementations.
`src/toolkit.py` is deleted.
Package roots such as `src/__init__.py`, `src/tooling/__init__.py`, and `src/tools/__init__.py` are intentionally inert; do not assume convenience re-exports.

The forensic Auditor uses `src/forensic_budget.py` as its single budget policy,
`src/tools/official_documents.py` for approved-host HTML/text/PDF evidence, and
`src/tools/forensic.py` for deterministic completeness gates and ratios. PDF
extraction is intentionally bounded and text-only: `pypdf` falls back to
`pdftotext`; approved exchange domains are built in, while direct issuer/IR
document hosts require the operator's `AUDITOR_OFFICIAL_DOCUMENT_HOSTS`
allowlist. OCR and general-purpose issuer-site crawling remain out of scope.

The Auditor maintains separate `CURRENT_STATEMENTS` and `AUDITED_BASELINE`
tracks. A signed audit opinion may establish the latest audited baseline even
when fresher interim/current statements exist, but figures are compared only
when period type/end, currency, and consolidation scope align. Fiscal-year
labels are never inferred from calendar years. Saved artifacts include the
actual Auditor budget ledger and attribution lists only models that made calls.

Management-guidance baseline evidence is owned by
`src/agents/management_guidance.py`. Before the Foreign Language Analyst runs,
it searches the latest results package and earnings bridge, checks the statutory
filing API, and in full mode extracts query-relevant passages from discovered
sources. Exchange-aware searches include the active fiscal period where the
exchange convention supports it and local filing vocabulary otherwise. An exact
ticker-matched result title can supply a bounded local listing name in any script
for the bridge search, with the previously resolved name retained only as fallback.
Search provenance is code-owned, then promoted deterministically into Senior
Fundamentals so temporary tax/regulatory benefits can affect scoring and BUY
eligibility even when no extraordinary-expense line appears. A results URL
without operating- and net-income guidance is treated as an unresolved baseline,
not evidence of durable earnings. `src/guidance_vocabulary.py` owns the
jurisdiction-specific search and excerpt vocabulary; the generic search formatter
accepts explicit priority terms and has no embedded finance-language bias.
`src/earnings_baseline.py` owns the pure status sets and scoring predicates. If
the Foreign Language Analyst returns useful research but omits or malforms only
the guidance block, code appends a conservative unresolved/search-failed block;
empty or unusable agent output still fails closed.

Foreign-language ownership and exact capacity claims are normalized by
`src/agents/foreign_language_evidence.py` against that agent's own
`ToolMessage` evidence before Senior Fundamentals can consume them. The
normalizer accepts the canonical `ToolEvidenceRecord` only; both the live agent
path and the fundamentals barrier convert run-scoped ledger records through
`message_utils.evidence_record_to_tool_evidence()`, preserving inspection status
and source authority. Deterministic evidence preloads use `ToolSource="preflight"`
while retaining the responsible seat in `agent_key`, so audit provenance does not
conflate code-owned retrieval with LLM-requested calls. The
`EntityGovernanceCard` keeps largest shareholder, controlling shareholder,
control status/basis, and ownership provenance separate. A minority stake or
equity-method relationship does not establish control; sub-50% control requires
an explicit basis plus official-filing evidence or corroboration from two
single-source tool records on distinct domains.
Related listed tickers must occur in the same supporting evidence, so an
unsupported FLA ticker cannot re-enter through Senior's restatement.

Junior Fundamentals requests its two ticker-bound tools in one model turn. Its tool
node assembles the existing raw-data wrapper from that round’s complete inspected
outputs before display caps; re-entry closes the exchange without a copying model
call. Useful inspected neighbors survive partial provider failures with a degradation notice;
failed, blocked or absent acquisition remains explicitly unavailable,
and artifact overflow fails closed while canonical metrics remain independent.

Saved decision provenance separates configured bindings and successful models used
from the accepted PM response model. Aggregate usage alone cannot identify the
producer after corrections; legacy or unstamped responses report unknown or ambiguous.

Every shared graph tool loop—Market, Sentiment, News, Junior Fundamentals, Foreign
Language, and Value Trap—is bounded by `ResearchBudgetLedger`. The policy instance is
shared by its analyst and tool node, so model turns, tool rounds, fan-out, per-tool and
purpose counts, duplicates, failed-tool/host circuits, and cumulative model-facing
evidence all close through one authority. Evidence capping truncates ToolMessage text;
it never drops the call/result pair or alters the complete structured-ingress record.
When a limit closes research, the next model invocation has no tools bound and must
synthesize from retained evidence. Saved `research_budgets.llm_calls` counts logical
application model turns; `token_usage.call_attempts` separately counts provider
transport attempts.

`src/tooling/` owns cross-cutting tool execution, audit hooks, argument policy, and untrusted-content inspection.

`src/runtime_diagnostics/` owns artifact completion/validity and publishability checks.

`src/data_block_utils.py` owns structured data-block marker vocabulary. Use
`BLOCK_SHAPES`, `fenced_start()`, `fenced_end()`, `build_fenced_block()`,
`unfenced_label()`, and the shared tolerant fenced matcher instead of hand-rolled
`--- START` / `--- END` regexes. `tests/prompts/test_marker_parity.py` is the L0
guard for prompt marker form, parser shape parity, and source-level marker drift.

## Information Flow Model

Primary agent-to-agent flow is through typed state fields, not just message history.
Parallel tool transcripts still share the graph's `messages` field, but retention and
invocation use the same agent ownership boundary. Assistant tool calls and matching
results are atomic retention units, bounded per agent and validated before any native
or compatible provider request. A permissive provider accepting malformed history is
never treated as an application compatibility feature. Legal Counsel keeps a private
bounded loop because its deterministic preflight and forced JSON synthesis are one
transaction; it has no duplicate graph tool-node path.

Material factual claims flow through the canonical claim envelope. Policy and roles
live in `src/claim_policy.py`; `src/tooling/structured_ingress.py` captures registered
tool payloads before text conversion or truncation and rejects conflicting valid
payloads instead of choosing one silently. `src/analysis_snapshot.py` owns claim
construction, monotonic refresh, and Senior fact reconciliation;
`src/score_lineage.py` owns rubric dependencies and deterministic score/report
projection. The Fundamentals Sync barrier mints the first snapshot from deterministic
raw metrics, normalized structured reports, and bound fetched evidence before Senior
Fundamentals runs. Senior may interpret those claims and calculate rubric scores, but
registered facts in its `DATA_BLOCK` are overwritten from the pre-Senior snapshot when
they conflict. Later stages may add validated derivations or stronger newly bound
evidence; they do not rebuild canonical facts from agent prose.

Claims have explicit `SUPPORT`, `GATE_INPUT`, or `CONTEXT` decision roles. The Portfolio
Manager references eligible canonical claim IDs and every active deterministic gate in
`DECISION_FACTS` / `DECISION_GATES`. `src/pm_claim_audit.py` reconciles and validates
that trace after all verdict rewrites, so the persisted trace describes the final
decision rather than the model's pre-policy draft. Invalid structured traces receive
one bounded correction attempt. Free-prose source-family lint remains advisory because
the claim IDs and gates are the load-bearing contract; existing PM/article claim audits
still caveat or block unsupported published assertions.

Revenue and earnings MRQ claims carry separate period provenance; the shared
`LATEST_QUARTER_DATE` display field is populated only when all available MRQ claim
periods agree. Source-required policies always mint explicit negative claims when
evidence is missing or unsupported, so omission cannot let Senior reintroduce a
number. Post-Senior compatibility snapshots are `DEGRADED`, cannot support BUY, and
make the analysis non-publishable. Adjusted rubric scores are `GATE_INPUT` claims,
never standalone BUY support. Every positive score credit must resolve transitively to
an eligible registered fact; otherwise the score remains visible as advisory but is
projected as N/A for decision use.

Article flow is Writer → Editor/revision → deterministic audit → atomic save.
Source-sensitive assertions carry a temporary `CLAIM_USAGE` manifest tied to eligible
canonical IDs; `src/article_audit.py` validates and strips it. A valid strict snapshot
with an invalid final decision trace, unsupported registered prose, or a failed
citation/claim audit is saved as a draft rather than published.
The Writer also receives a bounded, explicitly non-canonical Fundamentals reference
for analytical breadth; registered snapshot claims override every conflict. Report
publishability, like article publishability, requires a valid snapshot and final
decision trace when those contracts are present.

`src/tooling/evidence_recorder.py` is the run-scoped inspected-evidence ledger and the
shared tool-result status classifier. Embedded provider errors override optimistic
outer status tokens, while mixed valid/error result sets retain valid evidence with a
partial-error reason. Search-result identity establishes relevance only; a
source-required claim binds only to fetched `EVIDENCE_FOUND` content, never a
`RESULTS_FOUND` listing. Issuer source authority comes from the centralized
official-document trust policy, not a producing-tool name, model-written token, title,
ticker, jurisdiction, or inferred issuer domain.
Latest-results and management-guidance numeric decision eligibility additionally
requires the asserted values to occur in the URL-bound evidence; URL identity alone is
insufficient. HTML extraction retains the established cell-per-line text consumed by
the comparative readers. Row joining and blanket spanning-table exclusion are withdrawn;
flattened text does not establish source geometry, and ambiguous bindings stay rejected.
Approved HTML IR hubs may expose a bounded ranked list of same-host
relative disclosure paths; those children are fetched through the same URL, DNS,
redirect, size, and inspection controls before they can bind.

Important distinction:

- artifact content field: the report or degraded fallback text/json
- `artifact_statuses`: execution/completion metadata

Current semantics:

- `complete=True, ok=True`: agent ran and produced valid output
- `complete=True, ok=False`: agent ran but failed; may still leave conservative fallback content
- `complete=False`: agent did not complete

Run-summary tool counters have two intentionally different meanings:

- `tool_calls`: compatibility count of retained `ToolMessage` objects
- `tool_executions`: actual run-scoped evidence-ledger executions, with agent and
  source breakdowns

Likewise, structural-recovery token cost remains attributed to the originating analyst
in the primary rollup, while `token_usage.by_seat` and `token_usage.recovery_usage`
preserve the canonical recovery seat and origin. Use the explicit execution/recovery
fields for cost comparisons; transcript retention is not a billing ledger.

`scripts/cost_report.py --efficiency` summarizes recovery share, output-cap attempts,
legacy PM model corrections, and research-budget activity. Its A/B report also warns
when ticker baskets, code commits, prompts, thesis configuration, mode, or dirty-tree
state make a provider comparison uncontrolled; binding digests may differ as the
intended comparison axis.

Interpret research stop reasons across a batch, not as isolated failures. Predominant
`MODEL_FINAL` means the budget is protective. Predominant `TOOL_ROUND_LIMIT` with
`forced_synthesis_used=true` means the cap is binding and substituting for a natural
stopping condition; investigate the prompt or stopping behavior, or re-evaluate the
cap from paired evidence, rather than reflexively lowering it.

Graph barriers use completion, not validity.
Downstream decision logic should use valid content helpers where correctness matters.

## High-Value Files

If something breaks, check these first:

- `src/main.py`
- `src/graph/routing.py`
- `src/graph/components.py`
- `src/agents/analyst_nodes.py`
- `src/agents/decision_nodes.py`
- `src/agents/consultant_nodes.py`
- `src/data/fetcher.py`
- `src/runtime_diagnostics/`
- `src/validators/red_flag_detector.py`
- `src/ibkr/reconciler.py`
- `scripts/portfolio_manager.py`

## Major Subsystems

### Data ingestion

`src/data/fetcher.py` is the core market/fundamental data pipeline.
It merges multiple sources and is a common regression surface.
Ownership now lives across:

- `src/data/source_fetchers.py`
- `src/data/metric_extraction.py`
- `src/data/merge_policy.py`
- `src/data/gap_fill.py`

### Validator

`src/validators/red_flag_detector.py` is the thin public validator facade.
Ownership now lives in:

- `src/validators/sector_classifier.py`
- `src/validators/metric_extractor.py`
- `src/validators/financial_rules.py`
- `src/validators/supplemental_extractors.py`
- `src/validators/supplemental_flags.py`

Together they parse the fundamentals `DATA_BLOCK` and drive auto-reject or risk-penalty outcomes.

### Memory

`src/memory.py` provides ticker-isolated ChromaDB memory plus macro/lesson retrieval.
Memory outages should degrade analysis, not abort it.
Memory writes are inspected per document with `SourceKind.memory_write`; blocked writes are skipped and fully blocked batches return `False`.

Macro surfaces are intentionally split:

- `MacroEventsStore` holds sparse portfolio-detected discrete shocks
- `src/macro_context.py` holds a short cached regional regime brief under `results/.macro_context_cache/`
- saved analysis JSON records macro-context status/region metadata separately so operator review can tell whether the pre-graph summarizer ran

### Reporting

Main files:

- `src/report_generator.py`
- `src/charts/`
- `src/article_writer.py`

Large graph-state artifacts are bounded with `src/agents/output_limits.py::cap_state_value()` at the write points so oversized LLM output does not silently bloat state.

### Content-ingress hardening

The current trust-boundary model is:

- inspect tool output in the tool execution plane
- inspect cached or replayed untrusted context before it re-enters prompts
- inspect financial-API free-text fields before they become prompt-visible
- keep blocked content out of primary prompt paths rather than storing sentinel text as valid analysis

**Two tool deadlines, on purpose (October 2026).** The tool runner has its own
bound (`TOOL_EXECUTION_TIMEOUT_SECONDS`, 45 s). The `after()` hooks — content
inspection, including an LLM injection judge — run *after* that bound is
released, so a wider scope wraps the whole call, hooks included, and fails it
closed (`TOOL_TIMEOUT` in the graph). The outer scope looks redundant and is
not: it is the only bound on inspection. Since round 3 it is derived, runner
plus `TOOL_HOOKS_TIMEOUT_SECONDS` (45 + 75 = 120 s), so it can never be the
narrower one; it was a separate setting with a startup check, which made an
operator raising the runner for a slow vendor raise a second value too.
Where the wider scope lives changed in round 2; see below.

The outer scope uses `run_with_hard_timeout`, not `asyncio.wait_for`: `wait_for`
waits on the cancelled call, so a hook that swallows or outlives cancellation
held the node past its scope and, on Python 3.12, could still return its late,
uninspected result through `wait_for`. Both scopes clamp to a caller's enclosing
deadline when one is bound, but tool scopes never *bind* it: the judge fails open when admission refuses it, so a bound deadline
would turn a fail-closed timeout into uninspected content.

**The combined scope moved into the service (October 2026, round 2).** Until
then only the graph tool node had the outer scope, so the Consultant, Auditor,
Legal Counsel, article editor and evidence preflight, which call the service
directly, ran their `before()` and `after()` hooks with no bound at all.
`ToolExecutionService.execute` now wraps hooks and runner in
`run_with_hard_timeout(tool_execution_with_inspection_timeout_seconds())`, and
the graph node calls the service directly (round 3 retired its identical
wrapper, which only delayed cancellation by a loop turn; the node keeps the
`TOOL_TIMEOUT` conversion). Three details are load-bearing:

- **Nesting cannot renew the allowance.** A Consultant tool
  (`spot_check_metric_mcp_fmp`) routes its MCP call back through the service, so
  one `execute` runs inside another's runner. Each call binds its deadline in a
  tool-plane ContextVar (`tool_call_deadline` in `src/llm_runtime/timeouts.py`)
  before creating its inner task, and the runner binds its own narrower one, so
  the inner call clamps to what the outer runner has left. That ContextVar is
  deliberately *not* the LLM enclosing deadline, for the judge reason above:
  operational work reads it, LLM admission never does, and a test asserts
  hooks see no bound LLM deadline.
- **A late result is never recorded.** A hook that swallows cancellation keeps
  the orphaned chain running, and `EvidenceRecorder` is the last `after()` hook.
  The chain therefore re-checks the deadline before every `after()` hook and
  stops once it is spent, so the ledger never records what the caller was told
  timed out. That race is why MCP budget accounting moved out of the hook
  chain (round 3): `MCPRuntime` now counts each tools/call as it is sent, so a
  discarded result is still counted (`docs/MCP.md`, Budget interaction).
- **`run_with_hard_timeout` returns on time even when a hook ignores
  cancellation**, because it never awaits the cancelled task. It cannot help a
  hook that blocks the event loop synchronously, and it does not stop the
  orphan: it bounds the caller.

**One operational deadline (October 2026, round 3).** Rounds 1 and 2 left two
ContextVars, each with its own public clamp, and every call site chose one.
MCP and the official-document tool were switched to the tool-call clamp one at
a time; Tavily, market data, FX, filings, the editor's reference fetch and the
company-name lookups kept reading the LLM enclosing deadline alone. Inside a
45 s runner a research tool that searched (30 s) and then extracted (30 s) gave
the second call its full 30 s, and the runner cut it off as a bare
`TOOL_TIMEOUT` instead of the tool's own partial result. The fix is
structural, not per site: both ContextVars live in
`src/llm_runtime/timeouts.py`, `operational_deadline()` is the earlier of the
two, and `require_allowance` / `clamp_to_operational_deadline` /
`operational_time_left` are built on it, so every existing operational site
picked the runner up with no edit. `current_enclosing_deadline` stays the
LLM-only reader (dispatch, the runtime loop, the article workflow), so the
injection judge still never sees a tool deadline;
`tests/architecture/test_deadline_readers.py` keeps it there, keeps the
ContextVars private, and keeps the tool-call binder in the service.
`run_blocking_call(_sync)` applies the clamp itself and refuses with
`BlockingCallTimeoutError` before a thread or slot is taken, so a new blocking
call site cannot forget it.

Equal deadlines race: when an extractor's clamp equalled the runner's
remaining time, both timers were due at one instant, asyncio ordered them
arbitrarily, and the reply could be `TOOL_TIMEOUT` instead of
`EXTRACTION_TIMEOUT` (the same held for the 45 s MCP call under the 45 s
runner). `tool_call_deadline(seconds)` therefore binds `now + seconds -
deadline_margin(seconds)`: `DEADLINE_MARGIN_FRACTION` (5 %) of the cap, at most
`LLM_DISPATCH_MARGIN_SECONDS` -- the LLM plane's "the request's own deadline
fires before the hard cap" rule, applied to the tool plane through one helper.
The runner's hard timeout stays the full allowance. Because the combined scope
binds with the same margin, the runner always fires before the combined scope,
which is what keeps each timeout's label honest: a timeout is now classified
where it happens (`_ScopedToolTimeout`: `runner`, `before_hooks`,
`after_hooks`, `hooks_allowance_spent`) and only the combined bound's own
firing is labelled `with_inspection`; before, a hook's own `TimeoutError` was
relabelled as the combined scope.

**Market-data deadlines are grouped by scope, not by value (October 2026).**
The fields sit together in `Settings` in `src/config.py`. Equal numbers do not mean
one field: the 10 s vendor HTTP total and the 10 s ticker-resolution lookup, or the
8 s FX fetch and the 8 s Consultant spot check, bound different things and are
tuned separately. One field does cover several sites when the meaning is the same:
the four company-name sources in `src/ticker_utils.py` and the gap-fill name
lookup share `COMPANY_NAME_LOOKUP_TIMEOUT_SECONDS`, and FMP, EODHD and Alpha
Vantage share the vendor HTTP total. EODHD's filtered lookup keeps its own 5 s.
The `BlockingCallPolicy` constants in `src/blocking_io.py` keep their label and
in-flight cap, but their number is only the default; each call site applies its
field with `with_timeout()`.

Every site clamps through `require_allowance` (or
`clamp_to_operational_deadline`) in `src/llm_runtime/timeouts.py`, which holds
both deadline ContextVars so the data layer does not import tooling; see "One
operational deadline" below. A spent caller allowance raises
`TimeoutError` *before* the coroutine or thread is created, so it takes the
site's existing timeout fallback and leaves no un-awaited request. As with the
tool scopes, the aggregate metrics fetch reads the enclosing deadline but never
binds one: gap fill runs content inspection, and its LLM judge fails open when
admission refuses it. Under the pipeline the clamp is live: the ticker run
deadline is bound around the whole analysis (see the watchdog contract under
Screening Pipeline Semantics). Manual runs bind none, so there only the APAC
node's scope applies. Six
later fields joined the block with the same clamp: the ownership tool's 8 s per
yfinance section, StockTwits' 10 s request, the retrospective's 15 s fetch, the
two advisory 10 s fetches in `src/main.py` (market note and price snapshot,
kept apart) and the report-header name lookup's 6 s whole in `src/output.py`.
That lookup's inner per-candidate 5 s reuses `COMPANY_NAME_LOOKUP_TIMEOUT_SECONDS`,
because it is the same thing: one yfinance `.info` for a name.

The report-header lookup was also the sync-caller bug class. It was not a `with`
block. It called `shutdown(wait=False, cancel_futures=True)`, so the caller did
return on time and a mock-based test said so. But `concurrent.futures` joins every
executor worker at interpreter exit, so a stalled Yahoo read could hold the
process open after the report was written. It also built a fresh executor per
call, so each stall left another worker. A test that only checks the caller's
return time cannot catch this; check that the worker is a daemon.

The research deadlines follow the same rules and sit next to the market-data
block. Three 30 s values stay three fields: a Tavily search, a Tavily extract and
one filing fetcher. Two pairs are nested scopes, not duplicates. The 20 s
`get_official_filings` lookup in `src/tools/research.py` wraps the 30 s per-fetcher
bound in `src/data/filings/registry.py`, so today the outer one fires first. The
8 s DuckDuckGo wait in `src/tools/shared.py` wraps the ddgs client's own 5 s HTTP
timeout. The gap-fill search keeps its narrow 5 s as its own field and passes it
explicitly. Every Tavily entrypoint clamps an explicit per-call timeout too, not
just the default. On every path the absolute deadline is taken from the clamped
bound and covers search *and* inspection: a result that arrives with nothing
left is discarded before paid inspection starts, and an inspection that overruns
returns `None`, never the uninspected payload. The async entrypoints gained this
in October 2026 (round 2); before, their inspection had no bound of its own,
and the macro-context path, which calls `get_macroeconomic_news` outside the
tool service, had no bound on it at all.
Two caveats. httpx applies a timeout to each phase (connect, read, write, pool),
so for the editor's reference fetch the clamp narrows each phase but does not
cap the whole download. And ddgs takes whole seconds, so its clamped HTTP
timeout rounds up; the clamped wait around it is the bound that holds.

**The official-document sequence is allocated, not just bounded (October 2026,
rounds 2 and 3).** DNS check, download, pypdf, pdftotext and the content
inspection of the evidence run inside one 45 s tool runner; their ceilings sum
to more than that on purpose, and the allocation decides what is cut short.
Round 2 reserved absolute seconds: the download got what was left less
`PDF_EXTRACTION_RESERVE_SECONDS` (default the 20 s extract cap), within a 35 s
total. Two defects followed. A runner at or below the reserve (say
`TOOL_EXECUTION_TIMEOUT_SECONDS=20`) left the download nothing, so no document
was ever fetched, and nothing rejected that configuration. And inspection --
which may be a paid LLM judge -- ran after extraction with no share at all, so a
slow download plus extraction left the runner to cancel it into a bare
`TOOL_TIMEOUT`. Round 3 shares the tool's allowance (`_allocate`, read once
when the tool starts) by nested fractions, so the split holds for any runner by
construction: the download (DNS and redirects included, and httpx per phase,
since a tighter per-phase bound bought nothing) may use
`OFFICIAL_DOCUMENT_DOWNLOAD_FRACTION`; of the rest,
`OFFICIAL_DOCUMENT_INSPECTION_FRACTION` is held back from extraction; each
extractor gets its cap within what is left less that reserve, and one that
cannot start is skipped with `EXTRACTION_TIMEOUT`; inspection gets whatever is
then left through `InspectionService.check_within`, and evidence it cannot
finish inspecting fails the call with `INSPECTION_TIMEOUT` (a `FAILED` ledger
record since round 4), never returned uninspected. The defaults (0.55, 0.25) are chosen, not measured: saved
analyses record backends and failure reasons but no per-phase durations. In
the saved artifacts at the time, about 2,000 analyses carried document calls;
pypdf extracted in 283, pdftotext in none, and `EXTRACTION_TIMEOUT` appeared
in one -- so extraction is rarely the binding phase, and the download share
is the one to revisit when durations are logged.

**A failed document fetch carries its cause, and the client says who it is
(October 2026, round 3).** In saved runs `get_official_document` returned
`DOCUMENT_RETRIEVAL_FAILED` 11 times in about 1,619 network attempts (under 1%,
so a reliability nuisance, not a defect in the happy path). The causes lived only
in logs: six TLS certificate failures (certifi trust stays; the failure is now just
visible as `DOCUMENT_TLS_FAILED`), a server disconnect, a read timeout, and a few
others. One catch-all hid them, `DOCUMENT_HTTP_ERROR` dropped the status, and the
`official_document_failed` warning logged no host. The mapping now classifies by
exception class over the whole `__cause__`/`__context__` chain (the transport wraps
the ssl or OS error), applies only while the download is running (an ssl error
raised during extraction is not the network's), and is a tuple table so a new
reason is a row (`_TRANSPORT_REASONS`). The larger finding is the client sent no
User-Agent at all: `www.sec.gov` returned 0 successes in 262 calls, because EDGAR
refuses an anonymous or contactless client. `HTTP_USER_AGENT` is now the one string
(the document, editor and StockTwits clients), and sec.gov hosts
(`_CONTACT_REQUIRED_HOSTS`, matched on the registrable suffix, so `sec.gov.evil.com`
and `notsec.gov` are not SEC) get `<product token> <SEC_CONTACT_EMAIL>`, chosen per
redirect hop; with no contact the request is refused before DNS, never sent. Two
more traps: httpx validates the `Location` itself even with `follow_redirects=False`
(an unparseable one surfaces as `RemoteProtocolError`, `DOCUMENT_PROTOCOL_ERROR`), and
an https-to-http redirect to an approved host is reported as
`DOCUMENT_INSECURE_REDIRECT`, not as an unapproved host. A tool timeout used to
leave the ledger nothing (`after()` hooks run only for a result), so the URL was
lost and the model called it `EXTRACTION_TIMEOUT`; `ToolExecutionService` now tells
hooks that implement `failed()` (the `EvidenceRecorder`) of any failure, with the
timeout scope, as a `FAILED`/`UNAVAILABLE` record. Not changed: the vendor JSON
clients (aiohttp: FMP, EODHD, Alpha Vantage) and `scripts/find_gems.py`'s rotating
agents send their own headers.

`check_within` (also behind both Tavily entrypoints) binds no deadline: the
injection judge inside still sizes itself from the LLM enclosing deadline
alone, never the tool's, because refused by admission it fails *open*. Its
own bound fails closed. A test runs the real judge inside a tool deadline
with half a second left and checks its request is still admitted and sized
from its own policy.

**An inspection block is never evidence (October 2026, round 4).**
`check_within` used to return only the approved value, so a `TOOL_BLOCKED:`
placeholder came back to `get_official_document` as if it were the text; the
tool wrapped it in `STATUS: EVIDENCE_FOUND` with primary authority, the outer
inspection hook allowed that explanation, and the ledger bound a blocked
document as primary evidence. The raw `action` could not decide either: in warn
mode a `block` returns the original text. `InspectionService.evaluate` now sets
one enforced disposition on the decision it returns (`InspectionDecision.enforced`:
allowed, sanitized, blocked, inspection_failed), derived once from the decision,
the configured mode and the fail policy; a backend's raw decision carries `None`.
`check_within` returns that `(decision, approved)` pair, or `None` on timeout.
Callers branch on `enforced`, never on the approved text: the official-document
tool raises `ToolCallBlocked` for `blocked`, which `ToolExecutionService` turns
into the same blocked `ToolResult` a `before()` hook produces, so the ledger
records `BLOCKED`; it raises `ToolCallFailed` (a typed reason code) for
`inspection_failed` and a timeout, which reaches `EvidenceRecorder.failed` as
`FAILED` with that code and the model as `TOOL_ERROR: <code>`. The content hook
does the same for every tool, and Tavily returns `None` for an inspection that
failed, as for a timeout. Approved text under the useful-text minimum is
`DOCUMENT_TEXT_UNAVAILABLE`, and an empty replacement counts as a replacement.
A fail-closed backend error carries the code `INSPECTION_BACKEND_ERROR`, never the
exception text, which reaches only the log through `summarize_exception`.
Official documents get no stricter treatment than the configured mode.

The pdftotext child used to be killed only on its own timeout. The runner's
cancellation, the supported way the tool is cut short, left it running and
unreaped. Now every abnormal exit kills it and reaps it from a separate task
behind `asyncio.shield`, bounded by `SUBPROCESS_CLEANUP_TIMEOUT_SECONDS`; a
second cancellation interrupts only the wait, and a reap that outlives the
bound continues detached, held in `_PENDING_REAPS`, its failure logged by a
done-callback rather than left as "Task exception was never retrieved".

**Every child starts on the posix_spawn path (October 2026, round 3).** CPython
uses `posix_spawn` only for an absolute executable with `close_fds=False` and
no `cwd`, `start_new_session` or `preexec_fn`; anything else is
`fork()` + `exec()`. In the analysis process, threaded and with Apple's
Network.framework loaded by gRPC, `fork()` runs that framework's atfork
handler and can crash the child before `exec` (the macOS crash dialogs
`_apply_macos_fork_safety_env` in `src/config.py` mitigates). Two production
children forked: pdftotext (`close_fds` defaulted to True) and the run
fingerprint's git probe (a bare `git` and `cwd=`). Both now start in
`src/subprocesses.py`, which owns the spawn configuration and the kill-and-reap
cleanup above; git takes `-C <dir>` instead of `cwd=`.
`tests/architecture/test_subprocess_spawns.py` fails on any spawn elsewhere in
`src/`, resolving module aliases and imported names. Left as they are: the
pre-commit guards in `scripts/` (single-threaded, stdlib-only) and the mcp
SDK's own stdio-server spawn, which needs `start_new_session` to kill the
server's process tree (every registry server is streamable HTTP). The ownership tool's 8 s yfinance properties
(`src/tools/ownership.py`), StockTwits' 10 s HTTP (`src/stocktwits_api.py`) and
the retrospective's 15 s yfinance fetch are market-data reads and sit in that
block. The ddgs warm-up `DDGS(timeout=1)` runs at import, with no
request to bound.

The MCP deadlines follow the same rules, but each goes to an SDK parameter
rather than a wrapper, because the mcp SDK has one at every scope and tears the
request down itself when it fires. `src/mcp/` had no timeout of its own before.
The HTTP client ran on httpx2's implicit 5 s per phase, because the client is
supplied, and the SDK's 30 s / 300 s defaults apply only to a client it builds.
That 5 s is now `MCP_HTTP_TIMEOUT_SECONDS`, unchanged. The `ClientSession` read
timeout was `None`, so initialize, tools/list and tools/call had no protocol
bound. Now the session bound (30 s, the SDK's general-operation default) covers
the handshake and the catalog request, and `call_tool` passes its own 45 s
explicitly; without that it would inherit the handshake bound. The Consultant's
only MCP path runs inside two 45 s tool-runner scopes: the outer
`spot_check_metric_mcp_fmp` call and the inner `mcp__<server>__<tool>` call. The
outer one fires first, so the new bounds do not change its outcome. They matter
for a direct caller and for stdio, where nothing else bounded a silent server.
A spent allowance raises a non-retryable TRANSPORT `MCPCallError` before the
client or subprocess exists. If the session is already open, it is raised after
the session closes, never inside it. An `MCPCallError` raised inside the session
body comes back wrapped in the SDK's task groups and would be reclassified as a
generic error. See `docs/MCP.md`.

The one exception to "SDK parameter, never a wrapper" is the catalogue.
`list_tools()` takes no per-request timeout, and the session's was fixed at open,
before the handshake spent part of the allowance, so the listing runs under
`asyncio.timeout(time left)` and its timeout is captured inside the session body
and raised outside it. It must stay the public `list_tools()`: `send_request`
skips `_absorb_tool_listing`, which drops invalid `x-mcp-header` tools and caches
the output schemas. The production call path lists before every call for the same
reason: the SDK validates a result against the cached schema and, in a fresh
session, lists *after* the call to fetch it, under the stale session bound. MCP
reads the tool runner's deadline (`clamp_to_operational_deadline`,
`operational_deadline`) for every step and the retry backoff, never binds it.

The operational deadlines close the category, and most deliberately do **not**
clamp. The health check's 15 s probe is a top-level entrypoint, so nothing binds
a deadline around it. Its HEALTH_CHECK seat's 10 s client timeout sits inside
it, and it still calls `ainvoke` directly. The embeddings startup probe (15 s) is
never clamped because its result is cached for the process: a probe cut short by
one caller's deadline would turn embeddings off for every later caller. It is
*admitted* instead: it starts only when its full bound fits before the
operational deadline, and otherwise it is deferred with nothing cached, so the
next caller with time left probes (`embeddings_healthcheck_deferred`). Before
October 2026 it ran for its full bound even under a spent run deadline. The
probe runs once per process per binding (provider, model, dimension, key),
after success and after failure alike; it never ran once per memory instance.
The async entry points (`src/main.py`, before the retrospective and the graph,
and the retrospective-only mode) run it through
`FinancialSituationMemory.prepare_shared_embeddings` on a daemon thread
(`run_blocking_call` with a per-probe `InFlightLimit`), so the synchronous
constructors that follow find it cached. Before, a constructor ran it on the
event loop with a thread join, blocking the loop for up to the probe bound at
graph build. Sync callers (scripts, tests) still probe from the constructor,
through `run_blocking_call_sync` and the same policy. The 2 s GenAI client
close and the 2 s Langfuse flush run on the shutdown path, after any caller
deadline and inside `SHUTDOWN_HARD_TIMEOUT_SECONDS` (the cleanup half). They
keep their defaults because they set how long exit can take. Only the 30 s
embedding attempt clamps. Each attempt gets one absolute deadline that covers
the rate-limiter wait (`asyncio.timeout` around `aacquire`) and the request
together; before October 2026 the limiter wait had no bound and the request got
a fresh 30 s after it. With less than the minimum useful attempt
(`MIN_USEFUL_ATTEMPT_FRACTION`, 0.04, of the 30 s call timeout = 1.2 s; a chosen
fraction, not measured, shared with the LLM and MCP planes through
`min_useful_seconds` and always taken from the configured allowance, never the
clamped remainder) left before the caller's deadline, the
attempt raises `EmbeddingAllowanceSpent` before a token is taken, and tenacity's
retry excludes it. Without that exclusion, the three-attempt retry would add
about 6 s of backoff to a deadline that had already passed. Tenacity's backoff
is clamped with `wait_allowance` (the policy-free form of
`retry_wait_allowance`), stops once a useful attempt no longer fits, and
re-raises the last attempt's own error rather than a `RetryError`.

Embeddings take their limiter the way every LLM binding does:
`limiter_for_binding` for the embedding binding's provider and endpoint host
(`embedding_endpoint_host`, which also decides the OpenAI base URL). Until
October 2026 they always took the Gemini limiter, so OpenAI embeddings were
paced at Google's 15 RPM. `None` means the provider has no RPM ceiling, and the
request goes out without a wait. There is no unthrottled fallback any more: an
error resolving the limiter, waiting on it or sending is the attempt's error.
The fallback once caught *every* exception from the wait or the request and
sent again directly, so one attempt could make two requests; a later version
kept it only for a limiter that could not be had at setup, a case the binding
lookup no longer has. The physical requests per attempt are pinned at the HTTP layer
(`tests/memory/test_embedding_attempt_deadline.py`). OpenAI embeddings are built
with `max_retries=0` (the SDK default of two tripled every attempt). Google's
GenAI client, with `retry_options` unset, makes one tenacity attempt
(`retry_args` returns `stop_after_attempt(1)`) and sends no timeout at all, but
its async path runs on aiohttp whenever aiohttp is installed, and that path
re-sends once after a connection-level error (`ClientConnectorError`,
`ClientOSError`, `ServerDisconnectedError`, ...) following a 1-10 s sleep,
outside `retry_options`. That second send is left in place, and the attempt
deadline cuts it off. A supported switch to one send does exist: google-genai
uses httpx instead of aiohttp when `async_client_args["transport"]` is set, and
`GoogleGenerativeAIEmbeddings.client_args` is a public field passed to both the
sync and the async client. One object implementing both `httpx.BaseTransport`
and `httpx.AsyncBaseTransport` makes the disconnect a single send (verified on
google-genai 2.24.0). It is not used because an explicit transport turns off
httpx's environment proxies (`allow_env_proxies = trust_env and transport is
None`), and proxy users must keep proxying (`_apply_macos_fork_safety_env`).
We would have to re-implement `HTTPS_PROXY`/`NO_PROXY` handling and pool per
event loop to save one cheap re-send the deadline already bounds. The same
re-send exists for Gemini chat calls; any switch should be one helper serving
both planes. Note the patch point when testing: the OpenAI SDK uses the `httpx2`
fork, not `httpx`, and Google's async path uses aiohttp, not its httpx client,
so a fake `httpx` transport sees neither. The close keeps `asyncio.wait_for`, which is
safe there because `aclose()` is native async I/O, not a parked thread. The
memory timeouts read the live `src.config.config` rather than `src.memory.config`,
which about 28 memory tests replace with a `MagicMock`; a mock attribute would
reach the probe's bound and fail it. The Google embeddings client is built in one
place, `build_embeddings`, and tests patch `src.embeddings.GoogleGenerativeAIEmbeddings`.

A fresh `InMemoryRateLimiter` starts with an empty bucket and refills only after
a whole `1 / rps` interval, so every process waited about 5 s (15 RPM x 0.8)
before its first request. `create_process_rate_limiter` now starts the bucket at
one token; the pace after that is unchanged (`tests/llm_runtime/test_rate_limits.py`).
The Google RPM ceiling is read through `configured_google_rpm` (`GOOGLE_RPM_LIMIT`
under the multi-provider schema, `GEMINI_RPM_LIMIT` under the legacy one), which
the run config, the provider runtime, the unscoped fallback and the legacy proxy
all share. The unscoped fallback used to read `GOOGLE_RPM_LIMIT` under both
schemas. The run-scoped vendor list is now the fallback's own, which adds xAI:
a scoped xAI call used to go unthrottled.

**Per-ticker debate memory is inert (October 2026, left as is).** `src/main.py`
builds the graph with `cleanup_previous=True`, which deletes the ticker's
collections at the start of every run, and nothing in `src/` or `scripts/` ever
writes the bull, bear, judge, trader or risk-manager collections (`add_situations`
is called only for `lessons_learned`). Each full run still spends about four
embedding requests querying the empty bull and bear collections (R1 and R2), and
`create_trader_node`, `create_portfolio_manager_node` and
`create_research_manager_node` take a `memory` argument they never read. Retire
it or wire up the writes; do not re-derive this.

### IBKR path

Main files:

- `src/ibkr/`
- `scripts/portfolio_manager.py`

This path now includes:

- holdings + watchlist loading
- order-awareness
- recommendation/reconciliation logic
- portfolio-health and macro-event handling

### Why a rejected stock was re-analysed every run (September 2026)

A held position was re-analysed on every portfolio invocation for days while its
verdict was never in doubt, and no other position was refreshed at all. Both
symptoms came from one conflation and one scheduling bug.

`PortfolioEvidence.buy_blocking_flag_types` mixed two different claims. A
measured gate failure (`LIQUIDITY_HARD_FAIL`, minted `action=AUTO_REJECT`) is
settled for the current observation period — same-day research cannot move it.
An evidence gap (`action=REVIEW`) might close. The scheduler treated any non-empty
set as "indeterminate, so buy another analysis", making a liquidity rejection
permanently urgent. Fresh settled failures now stay out of the urgent stream and
rejoin the ordinary fair cycle near analysis expiry. The
discriminator already existed at every mint site; across 400 sampled artifacts
the `AUTO_REJECT`/`REVIEW` split was clean with no overlap, so the fix needed no
new taxonomy — only that the index projection stop flattening the two families.
This preserves the original anti-churn decision: evidence that cannot change today
does not consume an urgent slot, while evidence old enough to expire is measured
again through the same fair cycle as other aging analyses.

The backoff meant to catch this never fired. It was keyed on the *sorted
buy-blocking flag tuple*, which is downstream of search quality: across seven
consecutive analyses of the same position the set changed on six of six
transitions, so the stored key never matched, and the branch handling a changed
key *deleted* the stored entry. The mechanism was self-erasing and
anti-correlated with its purpose — flakier evidence meant more certain re-spend.
The persisted map was empty on disk every time it was inspected. It is now keyed
on the action basis, a closed enum, and the state file carries one entry per
ticker rather than a nested per-condition map.

Separately, `blocking` policy passed only the urgent stream to the planner, so
every slot urgent did not fill was wasted: one permanently urgent row plus 64
due-soon rows produced exactly one refresh per run. Strict priority is about
ordering, not throughput.

Two lessons worth keeping. Scheduler state must never be keyed on model or
search output — measure the key's stability before trusting it. And a
"conservative" default that keeps spending is not conservative: the safe
direction for a paid action is to withhold it.

Raw `contractDesc` values cross into research tickers through
`src/ibkr/ticker.py::classify_ibkr_symbol`; broker bookkeeping instruments such as
corporate-action receivables are rejected there rather than patched in callers.

Ownership is now split across:

- `src/ibkr/reconciler.py` for orchestration
- `src/ibkr/analysis_index.py` for latest-analysis cache/load/update
- `src/ibkr/reconciliation_rules.py` for FX, staleness, verdict, and sell helper rules
- `src/ibkr/position_evaluator.py` for held-position routing
- `src/ibkr/watchlist_evaluator.py` for watchlist routing
- `src/ibkr/opportunity_finder.py` for off-watchlist BUY discovery
- `src/ibkr/portfolio_health.py` for portfolio-health and correlated-sell handling
- `src/ibkr/buy_stability.py` for the BUY stability / hysteresis gate

**Agents-free invariant (load-bearing):** the IBKR reconciliation path must not
import `src.agents` (importing any `src.agents.*` submodule runs the heavy
`agents/__init__`, pulling in the whole LangGraph/LLM surface). The BUY stability
gate (`src/ibkr/buy_stability.py`) is **default-on** (`BUY_STABILITY_ENABLED`,
flipped June 2026) and runs on every reconcile, so it — and the analysis index —
parse PM verdicts via the neutral, stdlib-only `src/pm_decision_parser.py`
(`canonicalize_pm_verdict`, `parse_final_decision_scores`), never via
`src.agents.pm_verdict_metadata`. That neutral module is also the single home for
`canonicalize_pm_verdict` shared by charts/report/memo, which dissolves the old
`pm_block → src.agents.pm_verdict_metadata → agents/__init__ → decision_nodes →
pm_block` circular import. Boundary tests in `tests/ibkr/test_buy_stability.py`,
`tests/ibkr/test_analysis_index.py`, and `tests/test_pm_decision_parser.py`
enforce that these paths import no `src.agents`; do not reintroduce such an import
or move the parser back under `src.agents`.

## Testing Guidance

The test suite is broad and behavior-heavy.
Strong coverage areas include:

- graph routing
- validator behavior
- memory isolation
- fetcher/data edge cases
- chart extraction/rendering
- IBKR reconciliation

Patch owning modules in tests, not facades.
Use facade tests only for real public APIs.
Prefer patching `src.runtime_services`, `src.tools.*`, or the owning helper used at call time.

### Subdirectory fixtures that vanish under some orderings (October 2026)

`pytest tests/llm_runtime/test_timeouts.py tests/test_llms_flex.py
tests/llm_runtime/test_factory.py` passed 105 and errored 5 with `fixture
'restore_runtime_config' not found`. The fixture lives in `tests/llm_runtime/conftest.py`.
Any order without a root-level file between the two `llm_runtime` files passed.

The cause is in pytest 9.1.1, not in the layout. `--collect-only` shows two
`<Dir llm_runtime>` nodes:

- `Session.collect` (`_pytest/main.py`, around line 968) re-collects a parent
  without its duplicate cache when the remaining part is a single file
  (`handle_dupes=False`). The second argument, `test_llms_flex.py`, sits directly in
  `tests/`, so the `tests` package is collected again. That builds a new
  `Dir llm_runtime` with the same nodeid, and the third argument's items hang
  under it.
- `FixtureManager` (`_pytest/fixtures.py`, `pytest_plugin_registered` and the
  `pytest_make_collect_report` wrapper, around lines 1776–1803) defers a
  conftest until its directory is collected. It then pops it once and binds its
  fixtures to that `Dir` *object*. `_matchfactories` (around line 2155) matches by
  node identity. The second `Dir` gets nothing.

Named fixtures fail loudly. Autouse fixtures are worse: they are skipped silently,
so `tests/ibkr`, `tests/memory`, `tests/prompts`, `tests/tooling` and
`tests/advanced` can lose their isolation resets without an error. A scratch tree
reproduced it with all four `__init__.py` combinations. Making the layout
consistent would not have helped, and moving the fixture up would have hidden the
symptom while leaving the autouse loss.

The fix is `tests/helpers/conftest_recollection.py`, imported into
`tests/conftest.py`. When a later `Directory` collector appears for a path already
seen, it calls `parsefactories(holder=<that conftest>, node=<the new Dir>)`.
`tests/test_conftest_recollection.py` replays the ordering on the real files and on
a scratch tree. One of its tests asserts that pytest still drops the conftest
without the hook; when that one fails, pytest has fixed this upstream and the hook
can be retired.

### Unit tests reached Google's embeddings endpoint (October 2026)

The session sets a dummy `GOOGLE_API_KEY`, which is enough for
`FinancialSituationMemory` to build a real embeddings client and run its startup
probe. pytest-socket guards only `security`-marked tests, so the probe went out.
One full run recorded 57 `embed_content` requests from 19 tests, answered "API key
not valid" through the sandbox proxy. Offline, each probe waited out the probe
timeout instead.

Two kinds of test sent them:

- tests that build a memory directly, mostly `tests/memory/test_memory.py`;
- tests that build the graph with memory disabled. With memory off,
  `build_graph_components` constructed five shared `legacy_*` memories, and the
  first one probed (the result is cached per process). **This happened in
  production too:** a `--no-memory` run sent one embeddings probe at graph build,
  even under a spent run deadline. Since October 2026 memory off, or no ticker,
  builds no memory object at all; the no-ticker case also read collections
  shared by every ticker.

`refuse_embedding_sdks` (`tests/helpers/offline_network.py`, installed by the
`_offline_boundary` fixture in `tests/conftest.py`) patches the SDK methods to
refuse every request. It covers both providers, sync and async. The client stays
real, the probe fails as it would against an unreachable provider, and a test's own
patch of the client or its transport still wins. `integration` tests are exempt
(they need the provider), and so are `security` tests, for this refusal only: their
Step-3 tests drive the real SDK into wire fakes, while their sockets are still
refused. `tests/memory/test_embedding_probe_offline.py` guards it.

The same run measured the rest of the unit suite's egress: 147 tests made 1,309
network attempts. Most are yfinance over libcurl (568 to `fc.yahoo.com` alone),
then FX-rate fetches, Tavily DNS lookups, and the `find_gems` exchange-scrape
integration tests. pytest-socket cannot close it alone, because libcurl bypasses
Python sockets. `tests/helpers/offline_network.py` covers both and is what a test
should use to prove that it sent nothing.

### Unit-test egress, measured again (October 2026, round 3)

A refusing recorder (sockets, DNS and libcurl refused and logged per test; not
committed) over the whole suite found 564 attempts from 87 non-`integration` tests.
**Every one of those tests passed when refused**: each takes its "unreachable"
fallback, so no unit test needs the network. The cost was not the requests but
yfinance's own retry loop: `configure_yfinance_defaults` pinned
`network.retries = 3`, so each refused request slept 1 + 2 + 4 s. Those 87 tests
cost about 705 s; the retries are now 0 (the application owns retries and the fetcher
already falls through to the next source), and the same tests take seconds.

Causes, by seam (each closed where the test meets the data source, not at the socket):

- **IBKR connects (4 tests).** `IbkrSettings` and `DashboardSettings` hard-coded
  `env_file=".env"`; only `Settings` honoured `PYTEST_DISABLE_DOTENV`. The operator's
  IBKR credentials loaded, so tests that reach the pooled session tried to connect.
  `get_env_value`/`_parse_env_file` also parsed `.env` directly. All three now take
  `src.env_file.default_settings_env_file()`, and
  `tests/config/test_env_file_policy.py` scans every `BaseSettings` subclass in `src/`.
  A third, unreproduced symptom (an operator `LLM_CALL_HARD_TIMEOUT_SECONDS` reaching a
  concurrency test) was probed after every test of a full run: `Settings()` stayed at
  the shipped value and `os.environ` was clean, so the `.env`-only reader above is the
  only leak path found; if it recurs, check for a test that reloads `src.config` with
  the switch cleared.
- **Fetcher merge step.** `_repair_quote_range_from_history` (a Yahoo chart request)
  and gap filling (a Tavily client built from the dummy key, then a `.info` lookup).
- **FX.** The live tier of `get_fx_rate` on every non-USD position, plus a process
  `FxRateCache` that carried a rate between tests.
- **`src.main` around a run.** Benchmark note, price snapshot, macro context, company
  name for the header; six copies of the same patch block in `test_main_cli.py` are now
  the `offline_main_flow` fixture.
- **Tests that called the live tool** (ownership structure, foreign-source search,
  news): now fake the Yahoo property or the name resolver, and assert behaviour they
  could not before (insider trend, timeout bound). The one live yfinance canary is
  marked `integration`.
- **Hidden waits.** The Google process-fallback limiter paces 15 RPM x 0.8 = 0.2 rps, so
  memory tests waited about 5 s per acquire (100+ s across three modules).
  `_instant_process_limiters` replaces `create_process_rate_limiter`'s product for
  every non-`integration` test; `real_rate_limiter` opts a test of the real one back in.
- **Order dependence.** Two `TestMemoryIsolation` tests skipped when run alone and
  passed only with a probed embeddings client left on the class by an earlier test.
  `_reset_shared_memory_state` clears it per test; the tests now use a deterministic
  fake embedder, assert instead of skip, and add a positive control (the stored
  situation is retrievable from its own memory) so an empty cross-ticker result proves
  isolation rather than a broken store.

### The offline boundary is strict and integration is opt-in (October 2026)

The recorder above is now a fixture. `_offline_boundary` (autouse, `tests/conftest.py`)
installs `install_offline_network` for every non-`integration` test, `security` ones
included, and at teardown **fails** a test that recorded any attempt, naming targets
and thread names. Refusal alone is not a guard: production code catches the error and
degrades, so a leaking test passes while paying the timeout (the 705 s above). A test
that proves refusal installs its own recorder and asserts on that list.
`tests/test_offline_boundary.py` drives the fixture through a subprocess pytester
session (the outer session's own patches would otherwise mask it).

`integration` tests are skipped unless `--run-integration` is passed
(`pytest_collection_modifyitems`). `RUN_LIVE_EXCHANGE_PDF` was folded into it: it
guarded one canary with a second switch for the same idea, and an opt-in integration
run is already the deliberate "spend quota, use the network" act.

pytest-socket is gone (dependency and `--allow-unix-socket` addopt). Everything it
did is in `install_offline_network`, which also covers libcurl, records instead of
only raising, and allows AF_UNIX and `socketpair` (asyncio's self-pipe) by itself.
Its three other users (`security` guard, `tests/prompts/conftest.py`,
`tests/eval/test_deterministic_replay.py`) were subsumed. The old rule against
`-o addopts=""` (it dropped `--allow-unix-socket`, 156 `SocketBlockedError`s on
asyncio's self-pipe) no longer applies, though `--strict-markers` still lives there.

**The one socket created at collection is urllib3's.** `urllib3.util.connection`
runs `_has_ipv6("::1")` at import: it creates an `AF_INET6` socket and `bind`s it to
`::1`, never connecting, to set `HAS_IPV6`. Harmless, and it happens before any
fixture runs. `install_offline_network` imports that module first, so a lazy first
import inside a test cannot be mistaken for an attempt.

## Practical Notes

- Preserve the distinction between execution failure and bad business metrics.
- Do not put raw error strings into primary report fields unless that is an intentional degraded output.
- Optional cross-checkers may fail without blocking publication if core artifacts are still valid.
- For long CLI phases, keep user-visible progress at the caller layer and make deeper logging optional.

## Current Refactor State

Already split:

- `src/agents.py` -> `src/agents/`
- `src/toolkit.py` -> `src/tools/` with facade removed
- `src/graph.py` -> `src/graph/`
- `src/validators/red_flag_detector.py` -> facade plus validator ownership submodules
- `src/runtime_diagnostics.py` -> `src/runtime_diagnostics/` package (`failure_classification` + `artifact_status` behind a re-exporting `__init__` with explicit `__all__`) — Stage 6, July 2026

Provenance typing (Stage 6, July 2026): the gate-critical snapshot / decision-trace / scorecard payloads now have versioned typed codecs in `src/provenance_schema.py` (`Scorecard`, `DecisionTrace`, `SchemaStatus`) plus `AnalysisSnapshot` in `analysis_snapshot.py`. `to_dict` appends `schema_version` (the only additive wire delta); `from_dict` fails closed on a future/corrupt gate-critical payload → `build_analysis_validity` marks it non-publishable. The `detect_red_flags` dict adapter and the `pm_claim_audit`/`article_audit` modules were deliberately left as-is (see `docs/PROVENANCE.md`).

Recent completed control-plane/security work:

- runtime-scoped service container via `RuntimeServices`
- provider-neutral LLM control plane via `src/llm_runtime/`: canonical seats,
  immutable per-run binding plans, reviewed model identity/capabilities,
  separate provider/group allowlisting and evidence qualification,
  native/compatible adapters,
  provider-neutral retry/reasoning policies, conservative provider-specific rate
  ceilings (including direct-construction fallbacks), and secret-free persisted
  binding telemetry; seat execution policy also owns sampling, client bounds,
  reasoning/output semantics, and quick-mode availability, with normalized
  legacy/provider parity contracts guarding the reachable fleet
- vendor-adversarial grouping: Bull and Bear remain in `base`; Consultant,
  Forensic Auditor, and Editor form `review`; APAC is independently bound through
  `regional_review`; base and review selectors can reverse without graph changes
- provider-selectable embeddings with fingerprinted provider/model/dimension/schema
  collection names; model switches preserve legacy Chroma collections rather than
  deleting them
- memory-write inspection
- financial-API text-field inspection
- artifact bounding via `cap_state_value()`
- broader heuristic prompt-injection coverage
- `src/main.py` -> orchestration plus `src/cli.py`, `src/persistence.py`, and `src/output.py`
- `src/ibkr/reconciler.py` -> orchestration plus IBKR ownership submodules
- Snyk pre-commit redesign (July 2026): `scripts/snyk_check.sh` now takes a mode arg — `snyk-deps` BLOCKS commits (exit 1) on MEDIUM+ dependency findings (was always exit 0 / misleading `Passed`; `SNYK_ADVISORY=1` escapes), `snyk-container-base` is advisory REVIEW-REQUIRED for the base image. CI gains a token-gated, non-gating `snyk-container` job (Snyk CLI pinned in a `run:` step, scans base tag + built image) restoring the coverage the disabled Trivy jobs provided. 7-day dependency-release cooldown before remediation bumps.

Next likely large seams:

- `src/report_generator.py`

## Provenance

This note was refreshed from the current repo layout and recent refactor state.
It is intentionally shorter than older versions; use `README.md`, `CHANGELOG.md`, tests, and the live tree for deeper detail.

Financial health is assessed at `agents/verdict_policy.py` from `DecisionInputs`,
including the documented exception using both eligible canonical ratio facts from
one authoritative snapshot. `thesis_constants.py` owns the boundaries. Missing or
unreliable health is indeterminate; missing exception proof cannot rescue a reliable
low score. Active tenders exempt health/growth applicability while preserving their
raw assessments and independent initiation failures. Stored `health_gate`, PM
context and report visuals share that assessment; legacy policies without it display
unknown health. Canonical ratios retain the existing two-decimal producer precision,
so near-boundary values are assessed conservatively without a tolerance rescue.
