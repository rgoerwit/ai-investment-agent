#!/usr/bin/env bash
# A/B harness: does the optional IBKR data source actually CHANGE / improve analyses?
# Modeled on run_articles.sh. For each ticker it runs the SAME analysis twice —
# IBKR source OFF then ON — into separate RESULTS_DIRs, so the saved *_analysis.json
# artifacts can be diffed afterward. The flag is the ONLY variable.
#
#   ./run_ibkr_ab.sh
#   QUICK=0 ./run_ibkr_ab.sh                # full (non-quick) analyses (slower, costlier)
#   REPEATS=3 ./run_ibkr_ab.sh              # 3x per arm — measures LLM verdict variance
#   COOLDOWN_SECONDS=10 ./run_ibkr_ab.sh    # paid Gemini tier: shorter pause
#   TICKERS="2330.TW 005930.KS" ./run_ibkr_ab.sh   # override the basket
#
# Why these tweaks vs run_articles.sh:
#   * Each ticker runs twice (OFF then ON) into results/ab_off vs results/ab_on. The
#     analysis JSON follows RESULTS_DIR (NOT --output), so each arm gets its own
#     RESULTS_DIR or the second run would clobber the first.
#   * IBKR_DATA_SOURCE_ENABLED is set inline per run (shell env beats .env — a
#     "SHELL ENVIRONMENT OVERRIDE DETECTED" warning is expected, not a bug).
#   * --no-memory isolates the two runs (no lessons/retrospective bleed between OFF
#     and ON of the same ticker, which would confound the comparison).
#   * --quick by default: the IBKR source fires regardless of mode, so quick keeps the
#     A/B cheap while preserving the data-layer + deterministic gate signals. QUICK=0
#     for the full pipeline if you also want PM-verdict deltas (run REPEATS>1 then, to
#     separate IBKR's effect from the verdict's own sampling variance).
#   * No --article/--transparent: Medium articles add cost/noise irrelevant to the A/B.
#
# Prereqs / caveats:
#   * The ON arm only differs if IBKR is configured AND entitled for these exchanges.
#     With no creds it no-ops and both arms match — itself a valid finding (look for
#     _ibkr_advisory_status=UNAVAILABLE in the ON artifacts).
#   * Verify the snapshot FIELD CODES/UNITS on your account first (esp. marketCap
#     millions-vs-absolute, trailing-vs-forward P/E) or the mapping may be wrong.
#   * Default basket is ex-US gap-exercisers (KRX/TW/T) where yfinance vacuums. Add a
#     well-covered name (e.g. 7203.T) as a control: there IBKR can only override/diverge,
#     not gap-fill, which isolates the "is the override more accurate?" question.
#
# Run from the repo root.
set -uo pipefail   # NOT -e: one run failing must not abort the batch

export GRPC_POLL_STRATEGY=poll
export GRPC_VERBOSITY=ERROR
export GRPC_TRACE=""

# Basket: space-separated env override, else the default gap-exercising set.
read -r -a TICKERS <<< "${TICKERS:-009970.KS 030190.KS 2458.TW WDO.T 001060.KS}"

COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-60}"
QUICK="${QUICK:-1}"
REPEATS="${REPEATS:-1}"
OFF_DIR="${OFF_DIR:-results/ab_off}"
ON_DIR="${ON_DIR:-results/ab_on}"

# Optional --quick as a plain string (empty-array + set -u is unsafe on macOS bash 3.2).
QUICK_ARGS=""
[[ "$QUICK" == "1" ]] && QUICK_ARGS="--quick"

# ---- Preflight: can IBKR even serve fundamentals for this basket? ------------
# A neutered/unentitled IBKR is expected and fine — but then the ON arm equals OFF
# and the A/B is uninformative, so flag it up front instead of burning a full run.
# Exit codes from the probe: 0=serves >=1 ticker, 1=configured-but-inert, 2=not
# configured. PREFLIGHT=0 to skip; ABORT_IF_INERT=1 to stop when inert.
PREFLIGHT="${PREFLIGHT:-1}"
if [[ "$PREFLIGHT" == "1" ]]; then
    echo ">>> Preflight: probing IBKR entitlement for the basket (+controls)..."
    IBKR_DATA_SOURCE_ENABLED=true poetry run python scripts/ibkr_snapshot_probe.py \
        "${TICKERS[@]}" --with-controls --no-raw
    pf=$?
    case "$pf" in
        0) echo ">>> Preflight: IBKR serves fundamentals for >=1 ticker — A/B is meaningful." ;;
        2) echo ">>> Preflight: IBKR NOT configured — ON arm will equal OFF (no-op)." ;;
        *) echo ">>> Preflight WARNING: IBKR inert for this basket (no fundamentals returned)."
           echo ">>>   ON arm WILL equal OFF — expected if you lack market-data entitlement"
           echo ">>>   for these exchanges. The A/B cannot show an IBKR effect."
           if [[ "${ABORT_IF_INERT:-0}" == "1" ]]; then
               echo ">>>   ABORT_IF_INERT=1 — stopping before the uninformative run."
               exit 3
           fi ;;
    esac
    echo ""
fi

succeeded=()
failed=()

total=$(( ${#TICKERS[@]} * 2 * REPEATS ))   # tickers x (OFF,ON) x repeats
run_idx=0

run_one() {
    # $1 ticker  $2 flag(true|false)  $3 base_dir  $4 label
    local ticker="$1" flag="$2" base_dir="$3" label="$4"
    local safe="${ticker//./_}"
    run_idx=$((run_idx + 1))

    mkdir -p "$base_dir/images"

    echo "=============================================================="
    echo ">>> [$run_idx/$total] $label  ticker=$ticker  IBKR=$flag"
    echo ">>>      RESULTS_DIR=$base_dir"
    echo "=============================================================="

    if RESULTS_DIR="$base_dir" IBKR_DATA_SOURCE_ENABLED="$flag" \
        poetry run python -m src.main \
            --ticker   "$ticker" \
            $QUICK_ARGS \
            --no-memory \
            --output   "$base_dir/${safe}_report.md" \
            --imagedir "$base_dir/images"; then
        echo ">>> OK: $ticker ($label)"
        succeeded+=("$ticker:$label")
    else
        echo ">>> FAILED: $ticker ($label) (exit $?)"
        failed+=("$ticker:$label")
    fi

    # Cooldown between runs (skip after the very last run)
    if [[ $run_idx -lt $total ]]; then
        echo ">>> cooldown ${COOLDOWN_SECONDS}s..."
        sleep "$COOLDOWN_SECONDS"
    fi
}

for ticker in "${TICKERS[@]}"; do
    for r in $(seq 1 "$REPEATS"); do
        if [[ "$REPEATS" -gt 1 ]]; then
            off_dir="${OFF_DIR}/r${r}"
            on_dir="${ON_DIR}/r${r}"
        else
            off_dir="$OFF_DIR"
            on_dir="$ON_DIR"
        fi
        run_one "$ticker" false "$off_dir" "OFF r${r}"
        run_one "$ticker" true  "$on_dir"  "ON  r${r}"
    done
done

echo "=============================================================="
echo "Summary: ${#succeeded[@]} succeeded, ${#failed[@]} failed"
[[ ${#succeeded[@]} -gt 0 ]] && printf '  OK:     %s\n' "${succeeded[*]}"
[[ ${#failed[@]}    -gt 0 ]] && printf '  FAILED: %s\n' "${failed[*]}"
echo ""
echo "Compare the two arms (OFF=${OFF_DIR}  ON=${ON_DIR}):"
echo "  * Did IBKR contribute?   search the ON *_analysis.json for the provenance keys"
echo "    _ibkr_advisory_status / _ibkr_metrics / _field_sources / _ibkr_advisory_conflicts."
echo "  * Did it change anything? diff each ticker's report/DATA_BLOCK across arms —"
echo "    watch thesis gates (Health/Growth/PE<=18/PEG) and red-flag/PE_VS_SECTOR firings"
echo "    (deterministic), then the PM verdict (use REPEATS>1 to gauge its own variance)."
echo "  * Is it an improvement? where IBKR overrode/diverged, check its value against the"
echo "    filing or EODHD (quality 9.5) on a sample — closer-to-filing => the 9.4 override"
echo "    is justified; outlier => demote to gap-fill-only or fix the field mapping."
exit "${#failed[@]}"
