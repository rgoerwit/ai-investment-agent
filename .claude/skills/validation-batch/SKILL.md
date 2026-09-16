---
name: validation-batch
description: >
  Run a live validation batch of pipeline analyses against a change set, then read
  the artifacts for improvement or regression. Use when asked to run a live test
  suite, run some test analyses, validate uncommitted changes end to end, compare
  quick against full analyses, or check whether a change helped. Covers roster
  design, detached launch, iCloud and sandbox workarounds, and cross-run comparison.
---

# Running a validation batch

A batch costs roughly **$1.60–$3.50 and 90 minutes to two hours**. Spend the cheap
deterministic checks first — they catch a different and larger class of defect, and
they cost nothing.

## 1. Cheap gates first, always

```bash
poetry run python scripts/check_macos_cloud_files.py --hydrate   # see §6
poetry run ruff check src/ tests/ scripts/
```

Then the suite. See `/run-tests` for scope selection — do not reach for the full
~9,100 tests when the owning directory will do.

**A live batch is not a substitute for pytest, and has repeatedly been used as one.**
The August 2026 fast-fail regression (a stale `analysis_outcome` letting the debate
run after an established rejection) was found by diffing `debate_rounds` across a
90-minute, $1.61 batch. A unit test asserting
`pre_screening_result=REJECT → eligibility=REJECTED → routes to "PM Fast-Fail"`
would have caught it in seconds. If pytest has not run against the change set, say so
plainly and run it before spending money.

If corpus replay covers the changed area, run it before the live batch. Replay is
deterministic and free, and it correctly predicted two gate flips that then
reproduced live — the best verification-per-dollar in this workflow.

## 2. Freeze one variable

**Change the code or change the configuration, never both in one batch.** A batch that
moves `LLM_*_PROVIDER` *and* a dozen source files cannot attribute anything: a verdict
difference has two candidate causes and no way to separate them.

When both must move, run the carry-over tickers twice — once per configuration — and
treat only that pair as evidence.

Record the tree state before launching, because `git` is often unusable here (§6):

```bash
find src -name "*.py" -exec md5 -q {} \; | sort | md5 -q     # tree fingerprint
find src -name "*.py" -newermt "<previous batch start>"      # what moved since
```

Every run also stamps `code_commit` and `code_dirty` into its artifact, so the change
set is identifiable from the output even when git will not answer.

## 3. Roster design

Seven to nine runs. Each slot answers a different question, and only some of them
detect regressions at all.

| Slot | Count | Question it answers |
|---|---|---|
| **Variance probe** — one ticker, repeated | 3 quick | Is a metric stable run-to-run? |
| **Carry-over control** — recent baseline | 1–2 quick | Did anything change since last batch? |
| **Long-stale random** — untouched ≥ 8 weeks | 2 quick | Has quality drifted over months? |
| **Overlap full** — same ticker as a quick above | 1 full | Does quick→full still convert correctly? |
| **Full-only full** — rotate | 1 full | Do auditor / regional / round-2 paths work? |

Pick the roster with:

```bash
poetry run python scripts/pick_validation_roster.py --seed <n>
```

Notes on each slot, learned the hard way:

- **The variance probe is the highest-yield slot.** A single run of anything looks
  fine; defects in *stability* are invisible without repetition. The growth-score
  denominator bug (one ticker returning 40 / 60 / 66.7 / 40 / 50% on identical code)
  took four batches to surface because the ticker was run once per batch. Three runs
  in one batch would have shown it immediately. Choose a cheap, deterministic ticker —
  one whose verdict comes from a code-owned gate rather than model judgment — so
  anomalies are legible against a quiet background.
- **Carry-overs are the only regression detector.** Everything else can be judged for
  plausibility but not for change.
- **Long-stale picks must stay randomized.** Their value is longitudinal: drawing
  unpredictably from names untouched for months is what makes slow drift visible
  rather than confirming the handful of tickers you already watch. Expect them to
  produce few defects per batch; they are coverage insurance for exotic currencies,
  exchanges, and thin-coverage data shapes.
- **Do not spend both full slots on quick comparison.** Full mode runs machinery quick
  never touches — forensic auditor, regional specialist, round-2 debate, escalation
  seats. Reserve one full slot for exercising those.

## 4. Launch detached

Foreground and plain-background batches have been killed mid-run repeatedly. Only a
detached launch survives.

Write the driver with the file tools — **never a heredoc or shell redirection**; zsh
escapes `!`, silently turning `!=` into `\!=`. Then:

```bash
nohup <driver-script> > <progress-log> 2>&1 &
disown
```

- `setsid` does not exist on macOS. `nohup … & disown` is the form that works.
- `nice(5) failed: operation not permitted` on launch is benign — the sandbox
  declining a priority change. The batch starts normally.
- Have the driver echo `START`/`DONE ticker (mode) rc=N elapsed` per run to the
  progress log, and write per-run logs to a separate directory. Use `results/<TICKER>_<mode>_bN.md`
  so batches do not collide.
- `scripts/run_tickers.sh` is the maintained batch driver and sets the macOS
  fork-safety environment (`GRPC_POLL_STRATEGY`, `no_proxy`). Prefer it, or copy that
  environment block, when a batch triggers fork crashes.

Keep scratch files in the session scratchpad, not in the repo.

## 5. Watching without burning tokens

Arm a Monitor that emits one event per run and exits at completion. **Detect death by
file activity, never by process inspection** — see §6 for why process checks lie here.

```bash
sig=$( (cat "$LOGDIR"/*.log 2>/dev/null | wc -c; ls "$RES"/*_bN.md 2>/dev/null | wc -l) | tr -d ' \n')
# unchanged for ~35 polls at 60s => stalled; otherwise keep waiting
```

Expect **13–20 minutes per quick run and 17–25 per full**. An empty log for the first
minute or two is normal: structlog output is block-buffered when not a tty.

Do not poll between events. The driver writes terminal state to the progress log
regardless of whether anything is watching, so a dropped monitor loses nothing.

## 6. iCloud and sandbox failure modes

The working tree lives in an iCloud-synced folder. Cold files become `dataless`
placeholders and reads block on `fileproviderd`.

| Symptom | Cause | Workaround |
|---|---|---|
| `git status` / `git diff` hangs past 3 min | iCloud fault | Don't block on it; read `code_commit` / `code_dirty` from the artifacts |
| `fatal: mmap failed: Operation canceled` | iCloud fault | Same; retry once, then use the `find`-based fingerprint in §2 |
| `grep -r` over `src/` times out | walks cold files | `find src -name "*.py" -exec grep -n PATTERN {} +` |
| pytest fails collecting a file outside the test tree | cold placeholder | `scripts/check_macos_cloud_files.py --hydrate` |
| `pgrep` → `sysmond service not found` | sandbox | Not a liveness signal. Use file activity. |
| `ps` → `operation not permitted` | sandbox | Same. |
| `kill -0 <pid>` → rc=1 | ambiguous | **`operation not permitted` means the process EXISTS**; `no such process` means it is gone. Both return rc=1 — read stderr, not the exit code. |
| `command not found: timeout` | no GNU coreutils | Use the Bash tool's own `timeout` parameter |
| foreground `sleep` blocked | harness | Use Monitor or a background command |

Triage order for a stalled read: `dd` first, then `ls -lO` for the `dataless` flag,
then process inspection (needs the sandbox off). Never `ditto` a Chroma store.

If pytest aborts during rootdir collection with a `PermissionError` on a file outside
the test tree, that is a tool-boundary restriction, not a suite failure. Report the
suite as **unrun** — do not route around the boundary.

## 7. Reading the results

Use the maintained comparator rather than writing a new extractor:

```bash
poetry run python scripts/eval_longitudinal_compare.py \
    --tickers 2173.T 9692.T --lookback-weeks 6 --min-runs 2
poetry run python scripts/extract_run_health.py <per-run-log>   # network-health badge
```

`scripts/eval_longitudinal_compare.py` produces the per-ticker timeline with newly-appeared
red flags marked `[NEW]`, which is exactly the regression read. `--run-log` separates
transient vendor noise (DNS, auth, timeouts) from real change — do that before calling
anything a regression.

Report per run: verdict, cost, gate branch (`hard_fail` / `exception`), growth score
**with coverage** (`earned` / `available` of `rubric_total` — a percentage alone hides
a shrinking denominator), pre-screen result, `debate_rounds`, and blocking flags.

Cross-check these pairs, which have each caught a real defect:

- `pre_screening_result` against `analysis_outcome.eligibility` — they must agree.
  `REJECT` with `QUALIFIES` means a stale projection.
- `pre_screening_result: REJECT` against `debate_rounds` — a rejection should skip
  the debate; a non-zero count means the short-circuit is broken.
- Same-batch quick verdict against full verdict — a quick BUY converting to full
  HOLD is expected behaviour, not a defect.
- Growth `available` across runs — it should not move on identical code.

**Verify before reporting.** This workflow generates a lot of artifact surface and it
is easy to over-read. Several confident findings here were measurement errors:
counting a deduplicated list as an event count, grepping a span too narrow to contain
the text being called missing. Re-derive a number a second way before calling it a
defect, and check whether a flag simply was not raised on that run — LLM-raised flags
vary run to run, which is what the variance probe is for.

## 8. What a batch cannot tell you

- **N=1 per configuration cannot separate variance from effect.** One BUY that becomes
  a HOLD under a different provider may be model behaviour, a code change, or noise.
  Only the repeated slot licenses a claim about stability.
- **Inputs are uncontrolled.** Prices, news, and vendor models drift between batches. A
  cross-batch difference can be the market. Prefer replay when the question is about
  code.
- **Low incidence is weak grounds for deferring.** "Fired twice in 29 runs" across
  batches with no repeat structure is not an incidence measurement.

See `/diagnose-run` before rating any defect this batch surfaces, and
`/analyze-ticker` for what the run flags actually do.
