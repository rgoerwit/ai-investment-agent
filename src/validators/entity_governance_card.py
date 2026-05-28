"""Entity governance identity card.

Reconciles three weak signals to carry a structured view of *which legal entity*
is being analyzed through the pipeline:

  1. Deterministic hints from the yfinance/yahooquery merged dict (name tokens,
     business-summary phrases, etc.) — LLM-independent, always available.
  2. Senior Fundamentals' DATA_BLOCK GOVERNANCE fields (LISTING_ROLE,
     RELATED_LISTED_TICKERS, METRIC_SCOPE_PAYOUT, METRIC_SCOPE_OCF).
  3. Foreign Language Analyst's native-language ownership findings
     (ENTITY_ROLE_OBSERVED, Related Listed Tickers).

The card is built once after the Fundamentals Sync barrier and injected as
authoritative context for Bull/Bear/RM/Consultant/PM/Writer. Identity fields
(ticker, canonical_name) are authoritative; role inference is confidence-scored
and overrideable with cited primary evidence.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal, cast

import structlog

logger = structlog.get_logger(__name__)


# Fields we actually need from the merged yfinance dict for hint detection.
# We pull a small subset rather than parsing the entire blob.
_RAW_FIELDS_OF_INTEREST = (
    "longName",
    "shortName",
    "longBusinessSummary",
    "industry",
    "industryKey",
    "sector",
    "fullTimeEmployees",
    "totalRevenue",
)


def extract_merged_subset_from_raw(raw_fundamentals_data: str) -> dict[str, Any]:
    """Pull a small subset of yfinance fields out of the Junior TOOL 1 JSON blob.

    Junior emits raw_fundamentals_data as a stringified report containing
    `### TOOL 1: get_financial_metrics\n{<json>}` plus other sections. We parse
    only what the card hint detector needs; we tolerate parse failures because
    the card stays useful when only the structured DATA_BLOCK fires.
    """

    if not raw_fundamentals_data:
        return {}
    marker_idx = raw_fundamentals_data.find("get_financial_metrics")
    if marker_idx < 0:
        logger.debug(
            "egc_subset_marker_missing", input_chars=len(raw_fundamentals_data)
        )
        return {}
    body = raw_fundamentals_data[marker_idx:]
    start = body.find("{")
    if start < 0:
        logger.debug("egc_subset_brace_missing")
        return {}
    try:
        parsed, _ = json.JSONDecoder().raw_decode(body[start:])
    except (json.JSONDecodeError, ValueError):
        logger.debug("egc_subset_json_parse_failed")
        return {}
    if not isinstance(parsed, dict):
        logger.debug("egc_subset_not_dict", parsed_type=type(parsed).__name__)
        return {}
    return {k: parsed[k] for k in _RAW_FIELDS_OF_INTEREST if k in parsed}


EntityRole = Literal[
    "STANDALONE",
    "PURE_HOLDCO",
    "INTERMEDIATE_HOLDCO",
    "LISTED_SUBSIDIARY",
    "UNKNOWN",
]
HintRole = Literal["HOLDCO", "UNKNOWN"]
Confidence = Literal["clean", "unresolved", "conflict"]


# Holdco name tokens across the markets we routinely cover. Treated as HINTS,
# not classifiers — a hit raises one signal, not a verdict.
_HOLDCO_NAME_TOKENS = (
    "Holdings",
    "Holding",
    "Beteiligungs",
    "Participations",
    "Partecipazioni",
    "持株",
    "持株会社",
    "控股",
    "홀딩스",
    "홀딩",
)


@dataclass
class EntityGovernanceCard:
    """Authoritative identity + confidence-scored structural framing."""

    ticker: str
    canonical_name: str
    local_name: str | None = None
    entity_role: EntityRole = "UNKNOWN"
    controlling_shareholder: dict[str, Any] | None = None
    related_listed: list[dict[str, Any]] = field(default_factory=list)
    metric_scope: dict[str, str] = field(default_factory=dict)
    confidence: Confidence = "unresolved"
    deterministic_hints: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Layer 1: deterministic hints from the merged data dict
# ---------------------------------------------------------------------------


def deterministic_hints(merged: dict[str, Any]) -> tuple[list[str], HintRole]:
    """Return (hints_fired, role_guess).

    `role_guess` is HOLDCO only when at least one strong-ish signal fires; weak
    signals alone return UNKNOWN. Asset-light / royalty / financial-conglomerate
    false-positives are tolerated — Senior and FLA reconcile downstream.
    """

    hints: list[str] = []
    long_name = str(merged.get("longName") or "").strip()
    short_name = str(merged.get("shortName") or "").strip()
    summary = str(merged.get("longBusinessSummary") or "")
    industry = str(merged.get("industry") or merged.get("industryKey") or "")
    sector = str(merged.get("sector") or "")

    name_blob = f"{long_name} {short_name}".casefold()
    for token in sorted(_HOLDCO_NAME_TOKENS, key=len, reverse=True):
        if token.casefold() in name_blob:
            hints.append(f"name_token:{token}")
            break

    low_summary = summary.casefold()
    if "is a holding company" in low_summary:
        hints.append("summary:is_a_holding_company")
    if "operates as a holding company" in low_summary:
        hints.append("summary:operates_as_holding")
    if "operates as an investment holding" in low_summary:
        hints.append("summary:investment_holding")
    if "through its subsidiaries" in low_summary:
        hints.append("summary:through_subsidiaries")
    if "changed its name to" in low_summary and "holding" in low_summary:
        hints.append("summary:renamed_to_holdings")

    industry_low = industry.casefold()
    if any(
        token in industry_low
        for token in ("holding-companies", "conglomerates", "financial-conglomerates")
    ):
        hints.append(f"industry:{industry_low}")

    # Weak signals — recorded but never carry the verdict alone.
    employees = merged.get("fullTimeEmployees")
    revenue = merged.get("totalRevenue")
    if (
        isinstance(employees, int | float)
        and isinstance(revenue, int | float)
        and employees > 0
        and employees < 100
        and revenue > 100_000_000_000  # 100B local currency
    ):
        hints.append("weak:low_employees_vs_revenue")

    if sector.casefold() == "financial services" and any(
        kw in low_summary
        for kw in (
            "manufactures",
            "produces",
            "develops",
            "designs",
            "sells products",
            "provides industrial services",
        )
    ):
        hints.append("weak:financial_sector_with_industrial_summary")

    # Promote to HOLDCO guess only when a strong-ish signal fired.
    strong_signals = {
        h for h in hints if h.startswith(("name_token:", "summary:", "industry:"))
    }
    role_guess: HintRole = "HOLDCO" if strong_signals else "UNKNOWN"
    return hints, role_guess


# ---------------------------------------------------------------------------
# Layer 2/3 source extraction
# ---------------------------------------------------------------------------


def _role_from_senior(metrics: dict[str, Any]) -> EntityRole | None:
    """Pull LISTING_ROLE out of `extract_metrics()` output, return None when absent/unknown."""

    raw = metrics.get("listing_role")
    if not raw:
        return None
    return _coerce_entity_role(str(raw))


def _parse_related_listed(value: str | None) -> list[dict[str, Any]]:
    """Parse `RELATED_LISTED_TICKERS: 111770.KS:operating_subsidiary:50.5; ...` into edges."""

    if not value:
        return []
    edges: list[dict[str, Any]] = []
    for chunk in value.split(";"):
        parts = [p.strip() for p in chunk.split(":") if p.strip()]
        if not parts:
            continue
        ticker_match = _TICKER_RE.search(parts[0])
        if not ticker_match:
            continue
        edge: dict[str, Any] = {"ticker": ticker_match.group(0).upper()}
        if len(parts) >= 2:
            relationship = parts[1].strip().strip("()[]")
            if relationship:
                edge["relationship"] = relationship
        if len(parts) >= 3:
            try:
                edge["pct"] = float(parts[2].rstrip("%"))
            except ValueError:
                edge["pct_raw"] = parts[2]
        edges.append(edge)
    return edges


# Crude extractor for FLA's ENTITY_ROLE_OBSERVED line (prompt now requires it).
_FLA_ROLE_RE = re.compile(
    r"ENTITY_ROLE_OBSERVED:\s*(STANDALONE|PURE_HOLDCO|INTERMEDIATE_HOLDCO|LISTED_SUBSIDIARY|UNKNOWN)",
    re.IGNORECASE,
)
_FLA_RELATED_RE = re.compile(r"Related Listed Tickers:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FLA_CONTROLLING_RE = re.compile(
    r"Controlling Shareholder:\s*(.+?)(?:\n|$)", re.IGNORECASE
)
_VALUE_TRAP_MAJORITY_RE = re.compile(r"MAJORITY_HOLDER:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_TICKER_RE = re.compile(r"\b[A-Z0-9]{1,8}(?:[.-][A-Z0-9]{1,6})\b", re.IGNORECASE)
_HOLDER_NOT_PARENT_RE = re.compile(
    r"\b(?:founder|family|shareholder|private|vehicle|related parties|parties)\b",
    re.IGNORECASE,
)


def _coerce_entity_role(value: str) -> EntityRole | None:
    normalized = value.upper()
    if normalized in (
        "STANDALONE",
        "PURE_HOLDCO",
        "INTERMEDIATE_HOLDCO",
        "LISTED_SUBSIDIARY",
    ):
        return cast(EntityRole, normalized)
    return None


def _role_from_fla(fla_report: str) -> EntityRole | None:
    if not fla_report:
        return None
    m = _FLA_ROLE_RE.search(fla_report)
    if not m:
        return None
    val = m.group(1).upper()
    if val == "UNKNOWN":
        return None
    return _coerce_entity_role(val)


def _related_from_fla(fla_report: str) -> list[dict[str, Any]]:
    if not fla_report:
        return []
    m = _FLA_RELATED_RE.search(fla_report)
    if not m:
        return []
    value = m.group(1).strip()
    if value.upper() in ("NONE", "N/A", "UNKNOWN", ""):
        return []
    return _parse_related_listed(value)


def _controlling_from_fla(fla_report: str) -> dict[str, Any] | None:
    if not fla_report:
        return None
    m = _FLA_CONTROLLING_RE.search(fla_report)
    if not m:
        return None
    return _controller_from_text(m.group(1).strip(), source="fla_ownership")


def _controlling_from_value_trap(value_trap_report: str) -> dict[str, Any] | None:
    if not value_trap_report:
        return None
    m = _VALUE_TRAP_MAJORITY_RE.search(value_trap_report)
    if not m:
        return None
    return _controller_from_text(
        m.group(1).strip(), source="value_trap_majority_holder"
    )


def _controller_from_text(text: str, *, source: str) -> dict[str, Any] | None:
    if text.upper() in ("NONE", "N/A", "UNKNOWN", ""):
        return None
    # Try to peel a "Name (X%)" shape; fall back to the raw text.
    pct_match = re.search(r"\(([\d.]+)%\)", text)
    name = re.sub(r"\([^)]*\)", " ", text)
    name = re.sub(r"\s+", " ", name).strip()
    out: dict[str, Any] = {"name": name or text, "source": source}
    if pct_match:
        try:
            out["pct"] = float(pct_match.group(1))
        except ValueError:
            pass
    return out


def _senior_parent_company_controller(
    value: object, role: EntityRole
) -> dict[str, Any] | None:
    text = str(value or "").strip()
    if text.upper() in ("NONE", "N/A", "UNKNOWN", ""):
        return None
    if role in {"PURE_HOLDCO", "INTERMEDIATE_HOLDCO"} and (
        _HOLDER_NOT_PARENT_RE.search(text)
        or re.search(r"\((?:[0-4]?\d)(?:\.\d+)?%\)", text)
    ):
        return None
    return {"name": text, "source": "senior_parent_company"}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


_HOLDCO_FAMILY = {"PURE_HOLDCO", "INTERMEDIATE_HOLDCO", "LISTED_SUBSIDIARY"}
_HINTS_HOLDCO_SENTINEL: EntityRole = "PURE_HOLDCO"


def _reconcile_role(
    senior: EntityRole | None,
    fla: EntityRole | None,
    hints_role: HintRole,
) -> tuple[EntityRole, Confidence, str]:
    """Combine three role opinions into (role, confidence, notes_summary).

    Rules (from plan):
      - "clean": at least two of {senior, fla, hints} agree on family (HOLDCO vs STANDALONE)
        and no source explicitly disagrees.
      - "conflict": any two sources explicitly disagree on family.
      - "unresolved": only one source has an opinion, or all are UNKNOWN. This is
        intentional because Senior/FLA are advisory unless corroborated.
    """

    opinions: list[tuple[str, EntityRole]] = []
    if senior is not None:
        opinions.append(("senior", senior))
    if fla is not None:
        opinions.append(("fla", fla))
    if hints_role != "UNKNOWN":
        opinions.append(("hints", _HINTS_HOLDCO_SENTINEL))

    if not opinions:
        return "UNKNOWN", "unresolved", ""

    def is_holdco_family(r: EntityRole) -> bool:
        return r in _HOLDCO_FAMILY

    families = {
        source: ("HOLDCO_FAMILY" if is_holdco_family(role) else "STANDALONE")
        for source, role in opinions
    }
    unique_families = set(families.values())

    if len(unique_families) == 1:
        # All opinions agree on family. Pick the most specific role from
        # explicit sources (senior > fla > hints).
        family = next(iter(unique_families))
        for source_name in ("senior", "fla", "hints"):
            for s, role in opinions:
                if s == source_name and (
                    family == "STANDALONE"
                    and role == "STANDALONE"
                    or family == "HOLDCO_FAMILY"
                    and is_holdco_family(role)
                ):
                    confidence: Confidence = (
                        "clean" if len(opinions) >= 2 else "unresolved"
                    )
                    return role, confidence, ""
        # Fallback shouldn't trigger
        return opinions[0][1], "unresolved", ""

    # Mixed families → conflict
    disagreement = ", ".join(f"{s}={fam}" for s, fam in families.items())
    return "UNKNOWN", "conflict", f"role disagreement: {disagreement}"


def build_card(
    *,
    ticker: str,
    company_name: str,
    merged_data: dict[str, Any] | None,
    senior_metrics: dict[str, Any] | None,
    fla_report: str,
    value_trap_report: str = "",
) -> EntityGovernanceCard:
    """Build the card from already-available pipeline artifacts. Pure function."""

    merged = merged_data or {}
    metrics = senior_metrics or {}

    hints, hints_role = deterministic_hints(merged)
    senior_role = _role_from_senior(metrics)
    fla_role = _role_from_fla(fla_report)

    role, confidence, notes = _reconcile_role(senior_role, fla_role, hints_role)

    related_from_senior = _parse_related_listed(metrics.get("related_listed_tickers"))
    related_from_fla = _related_from_fla(fla_report)
    # Merge by ticker, prefer Senior's edge data when both surfaced the same ticker.
    related_by_ticker: dict[str, dict[str, Any]] = {}
    for edge in related_from_fla + related_from_senior:  # senior overrides
        related_by_ticker[str(edge["ticker"]).upper()] = edge
    related_listed = list(related_by_ticker.values())

    metric_scope: dict[str, str] = {}
    if metrics.get("metric_scope_payout"):
        metric_scope["payout"] = str(metrics["metric_scope_payout"])
    if metrics.get("metric_scope_ocf"):
        metric_scope["ocf"] = str(metrics["metric_scope_ocf"])

    sources: list[str] = []
    if hints:
        sources.append("yfinance_hints")
    if senior_role is not None or metrics.get("parent_company"):
        sources.append("senior_data_block")
    if fla_role is not None or fla_report:
        sources.append("fla_ownership")

    # Controller: prefer FLA's structured shape, then Value Trap majority-holder
    # data. PARENT_COMPANY remains a corporate-parent fallback only.
    controller = _controlling_from_fla(fla_report)
    if controller is None:
        controller = _controlling_from_value_trap(value_trap_report)
    if controller is None and metrics.get("parent_company"):
        controller = _senior_parent_company_controller(
            metrics["parent_company"],
            role,
        )

    canonical = (
        company_name
        or (merged.get("longName") if isinstance(merged.get("longName"), str) else None)
        or (
            merged.get("shortName")
            if isinstance(merged.get("shortName"), str)
            else None
        )
        or ticker
    )

    card = EntityGovernanceCard(
        ticker=ticker,
        canonical_name=str(canonical),
        local_name=None,
        entity_role=role,
        controlling_shareholder=controller,
        related_listed=related_listed,
        metric_scope=metric_scope,
        confidence=confidence,
        deterministic_hints=hints,
        sources=sources,
        notes=notes,
    )

    logger.info(
        "entity_governance_card_built",
        ticker=ticker,
        entity_role=role,
        confidence=confidence,
        hints_fired=len(hints),
        related_count=len(related_listed),
        sources=sources,
    )
    return card


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def card_to_prompt_block(card: EntityGovernanceCard) -> str:
    """Render a compact text block for injection into downstream agent prompts.

    Identity fields are presented as authoritative. Role/structure fields are
    confidence-scored and explicitly overrideable with cited primary evidence.
    """

    lines: list[str] = []
    lines.append("=== ENTITY GOVERNANCE CARD (authoritative for identity) ===")
    lines.append(f"Ticker: {card.ticker}")
    lines.append(f"Canonical name: {card.canonical_name}")
    if card.local_name:
        lines.append(f"Local name: {card.local_name}")
    lines.append(f"Entity role: {card.entity_role}  (confidence: {card.confidence})")

    if card.controlling_shareholder:
        cs = card.controlling_shareholder
        cs_name = cs.get("name", "?")
        cs_pct = cs.get("pct")
        suffix = f" ({cs_pct}%)" if cs_pct is not None else ""
        lines.append(f"Controlling shareholder: {cs_name}{suffix}")

    if card.related_listed:
        edges = "; ".join(
            f"{e.get('ticker')}:{e.get('relationship','?')}:{e.get('pct','?')}"
            for e in card.related_listed
        )
        lines.append(f"Related listed tickers: {edges}")

    if card.metric_scope:
        scope = ", ".join(f"{k}={v}" for k, v in card.metric_scope.items())
        lines.append(f"Senior DATA_BLOCK metric scope: {scope}")

    if card.notes:
        lines.append(f"Notes: {card.notes}")

    lines.append("")
    lines.append(
        "RULES: Ticker and canonical name are authoritative — do not contradict "
        "without quoting stronger primary evidence. Entity role is confidence-scored; "
        "if you have stronger evidence, state the override explicitly with citation. "
        "Metric scope is Senior-derived; if APAC, Consultant, or local filings cite "
        "a scope conflict, reconcile it explicitly rather than treating this card as "
        "automatic rejection authority. Do not silently re-frame the entity."
    )
    lines.append("=== END ENTITY GOVERNANCE CARD ===")
    return "\n".join(lines)


def card_from_dict(card_data: Mapping[str, Any] | None) -> EntityGovernanceCard | None:
    """Hydrate a card dict from graph state, returning None for absent/bad shapes."""

    if not isinstance(card_data, Mapping) or not card_data.get("ticker"):
        return None
    known_fields = {field_info.name for field_info in fields(EntityGovernanceCard)}
    filtered = {k: v for k, v in card_data.items() if k in known_fields}
    try:
        return EntityGovernanceCard(**filtered)
    except (TypeError, ValueError):
        return None


def card_to_prompt_block_from_dict(card_data: Mapping[str, Any] | None) -> str:
    """Render a graph-state card dict for prompt injection."""

    card = card_from_dict(card_data)
    return card_to_prompt_block(card) if card else ""


def requires_structure_disclosure(card: EntityGovernanceCard) -> bool:
    """Trigger the writer's opening disclosure paragraph?

    Yes when (a) sources conflict, or (b) entity is non-standard AND a concrete
    related listed ticker is known. Silent on standalone companies regardless.
    """

    if card.confidence == "conflict":
        return True
    if card.entity_role in _HOLDCO_FAMILY and card.related_listed:
        return True
    return False
