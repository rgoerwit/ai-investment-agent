"""Investment-memo builder rendered at the top of every report.

The memo is a tight, fixed-shape summary that sits above the chart and the
appendix transcript. Its slots are:

- decision (BUY / HOLD / DO_NOT_INITIATE / SELL / UNAVAILABLE)
- one-line thesis (≤ ~30 words pulled from PM rationale)
- variant view (Research Manager VARIANT_PERCEPTION when present)
- key numbers (4–6 DATA_BLOCK metrics)
- valuation (scenario IV summary when parseable, else legacy target context)
- top risks (3–4 named red flags / specific risks)
- kill criteria (parsed from Bear pre-mortem fenced block)
- confidence (one sentence summarizing which optional agents ran)

Every extractor degrades to a placeholder string rather than raising. A run
that produced no PM output renders an "Investment Memo — UNAVAILABLE" stub so
the rest of the report still publishes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from src.agents.support import extract_kill_criteria, get_bear_history
from src.charts.extractors.valuation import (
    extract_valuation_scenarios_for_fundamentals,
    format_iv,
    is_weak_buy_asymmetry,
    parse_numeric_field,
    scenario_upside_metrics,
    scenario_valuation_caveat,
)
from src.data_block_utils import extract_data_block_field, extract_last_fenced_block
from src.pm_decision_parser import canonicalize_pm_verdict
from src.reporting.source_confidence import (
    SourceRow,
    build_source_confidence_rows,
    render_source_confidence_markdown,
)
from src.reporting.state_access import (
    get_effective_red_flags,
    get_fundamentals_report,
    get_investment_plan,
    get_pm_output,
    get_valuation_params,
)

logger = structlog.get_logger(__name__)


_VERDICT_LINE = re.compile(r"VERDICT:\s*([A-Z_ ]+)")
_VERDICT_NARRATIVE = re.compile(
    r"PORTFOLIO MANAGER VERDICT:\s*(BUY|HOLD|DO NOT INITIATE|SELL)",
    re.IGNORECASE,
)
_RATIONALE_HEADER = re.compile(
    r"#+\s*DECISION RATIONALE\s*\n+(.+?)(?:\n#+\s|\n---|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_VARIANT_PLACEHOLDER = "Not explicitly stated."


@dataclass
class InvestmentMemo:
    """Structured memo content, ready for markdown rendering."""

    decision: str = "UNAVAILABLE"
    one_line_thesis: str = "Thesis unavailable."
    variant_view: str = _VARIANT_PLACEHOLDER
    key_numbers: list[str] = field(default_factory=list)
    valuation: str = "Valuation summary unavailable."
    top_risks: list[str] = field(default_factory=list)
    kill_criteria: list[str] = field(default_factory=list)
    confidence: str = "Confidence signals unavailable."
    source_confidence: list[SourceRow] = field(default_factory=list)


def extract_pm_verdict(pm_text: str) -> str:
    """Return the canonical verdict label from PM output, or 'UNAVAILABLE'."""
    if not pm_text:
        return "UNAVAILABLE"
    pm_block = extract_last_fenced_block(pm_text, "PM_BLOCK")
    if pm_block is not None:
        match = _VERDICT_LINE.search(pm_block)
        if match:
            verdict = canonicalize_pm_verdict(match.group(1))
            return "UNAVAILABLE" if verdict == "UNPARSEABLE" else verdict
    narrative = _VERDICT_NARRATIVE.search(pm_text)
    if narrative:
        verdict = canonicalize_pm_verdict(narrative.group(1))
        return "UNAVAILABLE" if verdict == "UNPARSEABLE" else verdict
    return "UNAVAILABLE"


# A period ending one of these is an abbreviation, not a sentence boundary —
# "**Operational Quality vs." rendered as the whole thesis (3393.T 2026-07-04).
_ABBREVIATION_SENTENCE_END = re.compile(
    r"\b(?:vs|etc|cf|ca|approx|incl|resp|e\.g|i\.e|inc|ltd|co|corp|u\.s|u\.k)\.$",
    re.IGNORECASE,
)


def extract_pm_thesis(pm_text: str, max_words: int = 30) -> str:
    """Pull the first sentence of the DECISION RATIONALE section, capped at max_words."""
    if not pm_text:
        return "Thesis unavailable."
    match = _RATIONALE_HEADER.search(pm_text)
    if not match:
        return "Thesis unavailable."
    body = match.group(1).strip()
    # Strip leading markdown noise that precedes the rationale prose: header
    # hashes ("#### 1."), list enumerators ("1.", "2)"), and bullets. The PM
    # almost always renders DECISION RATIONALE as a numbered list, so without
    # this the first-sentence regex latched onto a bare "1." marker and rendered
    # the thesis as "1." (systemic memo bug). Bold "**" prefixes are left intact
    # so the captured sentence keeps balanced markdown.
    body = re.sub(r"^(?:#+\s*|\d+[.)]\s+|[-•]\s+)+", "", body)
    # First sentence-ish, but never a degenerate marker-only capture (e.g. a
    # stray "1." an enumerator strip missed): require at least one letter. A
    # fragment ending in an abbreviation ("vs.", "e.g.") is not a sentence
    # boundary — keep appending the following segments.
    sentence = ""
    fragment = ""
    for candidate in re.finditer(r"(.+?[.!?])(?:\s|$)", body + " "):
        text = candidate.group(1).strip()
        if not fragment and not (re.search(r"[A-Za-z]", text) and len(text) > 3):
            continue
        fragment = f"{fragment} {text}".strip()
        if _ABBREVIATION_SENTENCE_END.search(fragment):
            continue
        sentence = fragment
        break
    if not sentence:
        sentence = fragment or body.split("\n", 1)[0]
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words]).rstrip(",;:") + "…"
    sentence = sentence.strip()
    # Balance a dangling bold marker left by abbreviation-fallback or word-cap
    # truncation so the memo line renders as markdown.
    if sentence.count("**") % 2:
        sentence += "**"
    return sentence


def extract_variant_view(state: dict) -> str:
    """Pull `VARIANT_VIEW:` / `CONSENSUS_VIEW:` from the Research Manager plan.

    Reads from either the runtime state (top-level ``investment_plan``) or the
    saved JSON shape (``investment_analysis.investment_plan``). Returns the
    placeholder string ``"Not explicitly stated."`` only when no variant
    content is found — the memo renderer omits the line entirely on that
    placeholder so the quality judge can't false-positive on it (Tier 2,
    Step 8).
    """
    plan = get_investment_plan(state)
    if not plan:
        return _VARIANT_PLACEHOLDER
    variant = re.search(
        r"VARIANT_VIEW\s*:\s*(.+?)(?:\n\n|\n[A-Z_]{3,}\s*:|\Z)",
        plan,
        re.DOTALL,
    )
    if variant:
        text = re.sub(r"\s+", " ", variant.group(1)).strip()
        words = text.split()
        if len(words) > 40:
            text = " ".join(words[:40]).rstrip(",;:") + "…"
        return text
    if "NO VARIANT" in plan.upper():
        return "Synthesis aligns with consensus — no material variant."
    return _VARIANT_PLACEHOLDER


# Each row carries one or more DATA_BLOCK field names. Real DATA_BLOCKs in the
# wild use `DE_RATIO` and `NET_DEBT_EBITDA`; the longer `DEBT_TO_EQUITY` /
# `NET_DEBT_TO_EBITDA` variants are kept as fallbacks so synthetic fixtures
# and any legacy emitters continue to populate the row.
_KEY_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("P/E (TTM)", ("PE_RATIO_TTM",)),
    ("PEG", ("PEG_RATIO",)),
    ("ROIC", ("ROIC_PERCENT",)),
    ("FCF yield", ("FCF_YIELD_PERCENT",)),
    ("Revenue growth (TTM)", ("REVENUE_GROWTH_TTM",)),
    ("Net debt / EBITDA", ("NET_DEBT_EBITDA", "NET_DEBT_TO_EBITDA")),
    ("D/E", ("DE_RATIO", "DEBT_TO_EQUITY")),
    ("Analyst coverage (EN)", ("ANALYST_COVERAGE_ENGLISH",)),
)


def extract_key_metrics(fundamentals: str, limit: int = 6) -> list[str]:
    """Pull up to `limit` non-empty key metrics from the fundamentals DATA_BLOCK.

    For each row, the first matching key wins — so legacy field names act as
    fallbacks when the canonical name (used in recent DATA_BLOCKs) is missing.
    """
    if not fundamentals:
        return []
    rows: list[str] = []
    for label, keys in _KEY_FIELDS:
        for key in keys:
            value = extract_data_block_field(fundamentals, key)
            if (
                value
                and value.strip()
                and value.strip().upper() not in {"N/A", "NA", "NONE"}
            ):
                rows.append(f"{label}: {value.strip()}")
                break
        if len(rows) >= limit:
            break
    return rows


_CURRENCY_PATTERN = re.compile(r"([A-Z]{3}|[$€£¥])")


def _detect_currency(fundamentals: str) -> str:
    """Best-effort currency tag for IV display — falls back to bare numbers."""
    for field_name in ("REPORTING_CURRENCY", "CURRENCY"):
        value = extract_data_block_field(fundamentals, field_name)
        if value:
            token = value.strip().split()[0] if value.strip() else ""
            if token and _CURRENCY_PATTERN.fullmatch(token):
                return token
    return ""


def format_scenario_summary(state: dict) -> str | None:
    """Return a one-line scenario summary if a valid VALUATION_SCENARIOS block exists.

    Returns ``None`` if scenarios aren't parseable so the memo can fall back to
    the legacy single-range string. This is the load-bearing seam for Step 6:
    when this function returns a string, the memo carries bear/base/bull IVs
    and a weighted IV in the Valuation slot; otherwise it carries the legacy
    target range.
    """
    valuation_params = get_valuation_params(state)
    if not valuation_params:
        return None

    fundamentals = get_fundamentals_report(state)

    scenarios = extract_valuation_scenarios_for_fundamentals(
        valuation_params, fundamentals
    )
    if scenarios is None:
        return None

    ccy = _detect_currency(fundamentals)
    prefix = f"{ccy} " if ccy else ""

    def _fmt(value: float) -> str:
        return f"{prefix}{format_iv(value)}"

    caveat = scenario_valuation_caveat(scenarios)
    warning = (
        " Warning: peak/distorted earnings flagged; weighted IV is conditional, "
        "not normalized fair value."
        if caveat
        else ""
    )
    current_price = parse_numeric_field(
        extract_data_block_field(fundamentals, "CURRENT_PRICE")
    )
    upside_metrics = scenario_upside_metrics(scenarios, current_price)
    if extract_pm_verdict(get_pm_output(state)) == "BUY" and upside_metrics:
        weighted_upside, downside_probability = upside_metrics
        if is_weak_buy_asymmetry(weighted_upside, downside_probability):
            warning += (
                " Warning: BUY verdict has weak valuation asymmetry; review "
                f"weighted IV upside ({weighted_upside * 100:.1f}%) and "
                f"downside probability ({downside_probability:.0f}%)."
            )

    return (
        f"Bear {_fmt(scenarios.bear_iv)} ({scenarios.bear.probability:.0f}%) / "
        f"Base {_fmt(scenarios.base_iv)} ({scenarios.base.probability:.0f}%) / "
        f"Bull {_fmt(scenarios.bull_iv)} ({scenarios.bull.probability:.0f}%); "
        f"weighted {_fmt(scenarios.weighted_iv)} "
        f"({scenarios.methodology}, sufficiency {scenarios.data_sufficiency}; "
        f"earnings basis {scenarios.earnings_basis}).{warning}"
    )


def extract_legacy_target_range(state: dict) -> str:
    """Compose a one-line valuation summary from whatever valuation context exists."""
    valuation_context = state.get("valuation_context") or ""
    if "Target Range" in valuation_context:
        match = re.search(
            r"Target Range:\s*([^\n]+)\nFair Value \(midpoint\):\s*([^\n]+)",
            valuation_context,
        )
        if match:
            return (
                f"Target range {match.group(1).strip()} (mid {match.group(2).strip()})."
            )
    # Try the DATA_BLOCK current price from either runtime state or saved JSON.
    fundamentals = get_fundamentals_report(state)
    price = extract_data_block_field(fundamentals, "CURRENT_PRICE")
    if price:
        return f"Current price {price.strip()}; target range unavailable."
    return "Valuation summary unavailable."


def extract_pm_risks(
    pm_text: str, red_flags: list[dict] | None, limit: int = 4
) -> list[str]:
    """Combine red-flag detail lines and PM-narrative risk bullets into a short list."""
    risks: list[str] = []
    for flag in red_flags or []:
        if not _is_material_risk_flag(flag):
            continue
        flag_type = flag.get("type") if isinstance(flag, dict) else None
        detail = flag.get("detail") if isinstance(flag, dict) else None
        if flag_type and detail:
            risks.append(f"{flag_type}: {detail}")
        elif flag_type:
            risks.append(str(flag_type))
        if len(risks) >= limit:
            return risks[:limit]
    if pm_text:
        # Pull bullets under "Key Risks" / "Risks" / "Top Risks" headings.
        section = re.search(
            r"#+\s*(?:TOP\s+|KEY\s+)?RISKS\b.*?\n+(.+?)(?:\n#+\s|\n---|\Z)",
            pm_text,
            re.DOTALL | re.IGNORECASE,
        )
        if section:
            for line in section.group(1).splitlines():
                stripped = re.sub(r"^[*\-•\s]+", "", line).strip()
                if stripped and len(stripped) > 4:
                    risks.append(stripped)
                if len(risks) >= limit:
                    break
    return risks[:limit]


def _is_material_risk_flag(flag: dict) -> bool:
    """True for flags that belong in the memo's Top risks list."""
    if not isinstance(flag, dict):
        return False
    if flag.get("action") == "AUTO_REJECT" or flag.get("severity") == "CRITICAL":
        return True
    penalty = flag.get("risk_penalty")
    if isinstance(penalty, bool):
        return False
    if isinstance(penalty, int | float):
        return penalty > 0
    return False


def summarize_confidence(state: dict) -> str:
    """One-sentence summary of which optional cross-checks ran."""
    run_summary = state.get("run_summary") or {}
    bits: list[str] = []
    verdict = run_summary.get("consultant_verdict")
    if verdict == "CLEAN" or (
        verdict is None and run_summary.get("consultant_successful")
    ):
        # `verdict is None` = pre-change saved JSON; fall back to the legacy
        # "ran ok" signal so old analyses still render.
        bits.append("consultant cross-check passed")
    elif verdict == "CONDITIONAL":
        bits.append("consultant approved with conditions — verify open items")
    elif verdict == "MAJOR_CONCERNS":
        bits.append("consultant raised major concerns")
    elif verdict == "REJECTED":
        bits.append("consultant did NOT approve")
    elif verdict == "ERROR":
        bits.append("consultant review failed validation")
    elif verdict == "SKIPPED":
        bits.append("consultant cross-check skipped (quick screen)")
    elif verdict == "UNPARSED":
        bits.append("consultant review unparsed")
    elif run_summary.get("consultant_completed"):
        bits.append("consultant ran but did not approve")
    if run_summary.get("auditor_successful"):
        bits.append("forensic auditor clean")
    elif run_summary.get("auditor_completed"):
        bits.append("forensic auditor ran with caveats")
    if run_summary.get("apac_specialist_successful"):
        bits.append("APAC specialist engaged")
    if not bits:
        return "Optional cross-checks did not run; rely on core agents only."
    return "Anchored on " + ", ".join(bits) + "."


def build_memo(state: dict) -> InvestmentMemo:
    """Assemble the InvestmentMemo from the analysis result dict.

    Accepts either the live AgentState dict (during graph execution) or the
    saved JSON shape (for retrospective rendering). All cross-shape lookups
    live in :mod:`src.reporting.state_access`.
    """
    pm = get_pm_output(state)
    fundamentals = get_fundamentals_report(state)
    bear_text = get_bear_history(state)
    red_flags = get_effective_red_flags(state)

    return InvestmentMemo(
        decision=extract_pm_verdict(pm),
        one_line_thesis=extract_pm_thesis(pm),
        variant_view=extract_variant_view(state),
        key_numbers=extract_key_metrics(fundamentals),
        valuation=format_scenario_summary(state) or extract_legacy_target_range(state),
        top_risks=extract_pm_risks(pm, red_flags),
        kill_criteria=extract_kill_criteria(bear_text),
        confidence=summarize_confidence(state),
        source_confidence=build_source_confidence_rows(state),
    )


def render_memo_markdown(memo: InvestmentMemo) -> str:
    """Render an InvestmentMemo as a self-contained markdown section."""
    if memo.decision == "UNAVAILABLE" and not memo.key_numbers:
        return (
            "## Investment Memo — UNAVAILABLE\n\n"
            "No Portfolio Manager output was available for this run; the appendix "
            "below contains whatever the graph did produce.\n\n---\n\n"
        )

    parts: list[str] = [f"## Investment Memo — {memo.decision}\n\n"]
    # The analyzer PM is portfolio-blind (no holdings access), so an analyzer
    # HOLD is always monitor-only — clarify up front, because the body can
    # retain the trader's pre-override sizing (e.g. 3626.T: "Initial Position
    # Size: 2.0%" below a HOLD with 0.0% recommended).
    if memo.decision == "HOLD":
        parts.append(
            "*New-candidate HOLD: monitor only — this analysis does not "
            "initiate a position.*\n\n"
        )
    parts.append(f"**Thesis.** {memo.one_line_thesis}\n\n")
    # Tranche 5, Step 8: omit the line entirely when the placeholder fires.
    # Rendering "Not explicitly stated." as a bolded section adds visual noise
    # for the reader and lets the quality judge false-positive on a marker
    # that carries no information. Honest no-variant content (`Synthesis
    # aligns with consensus — no material variant.`) is kept.
    if memo.variant_view and memo.variant_view != _VARIANT_PLACEHOLDER:
        parts.append(f"**Variant view.** {memo.variant_view}\n\n")

    if memo.key_numbers:
        parts.append("**Key numbers.**\n\n")
        parts.extend(f"- {row}\n" for row in memo.key_numbers)
        parts.append("\n")

    parts.append(f"**Valuation.** {memo.valuation}\n\n")

    if memo.top_risks:
        parts.append("**Top risks.**\n\n")
        parts.extend(f"- {risk}\n" for risk in memo.top_risks)
        parts.append("\n")

    if memo.kill_criteria:
        parts.append("**Kill criteria.**\n\n")
        parts.extend(f"- {trigger}\n" for trigger in memo.kill_criteria)
        parts.append("\n")

    parts.append(f"**Confidence.** {memo.confidence}\n\n")

    if memo.source_confidence:
        parts.append("**Source confidence.**\n\n")
        parts.append(render_source_confidence_markdown(memo.source_confidence))
        parts.append("\n")

    parts.append("---\n\n")
    return "".join(parts)


def render_memo_for_state(state: dict) -> str:
    """One-shot helper: build + render the memo from a state/result dict."""
    try:
        return render_memo_markdown(build_memo(state))
    except Exception as exc:  # pragma: no cover — defense-in-depth
        from src.error_safety import summarize_exception

        logger.warning(
            "memo_render_failed",
            **summarize_exception(exc, operation="render_memo_for_state"),
            exc_info=True,
        )
        return (
            "## Investment Memo — UNAVAILABLE\n\n"
            "Memo rendering encountered an error; see logs.\n\n---\n\n"
        )
