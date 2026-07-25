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
ControlStatus = Literal["CONTROLLED", "NOT_CONTROLLED", "UNKNOWN"]


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
    largest_shareholder: dict[str, Any] | None = None
    controlling_shareholder: dict[str, Any] | None = None
    control_status: ControlStatus = "UNKNOWN"
    control_basis: str = "UNKNOWN"
    ownership_evidence: dict[str, str] = field(default_factory=dict)
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
_FLA_LARGEST_RE = re.compile(r"Largest Shareholder:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FLA_CONTROLLING_RE = re.compile(
    r"Controlling Shareholder:\s*(.+?)(?:\n|$)", re.IGNORECASE
)
_FLA_CONTROL_STATUS_RE = re.compile(
    r"Control Status:\s*(CONTROLLED|NOT_CONTROLLED|UNKNOWN)(?:\n|$)",
    re.IGNORECASE,
)
_FLA_CONTROL_BASIS_RE = re.compile(r"Control Basis:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FLA_EVIDENCE_STATUS_RE = re.compile(
    r"Ownership Evidence Status:\s*(VERIFIED_URL|VERIFIED_OFFICIAL_FILING|NOT_FOUND|REJECTED|UNKNOWN)(?:\n|$)",
    re.IGNORECASE,
)
_FLA_SOURCE_URL_RE = re.compile(r"Ownership Source URL:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_FLA_AS_OF_RE = re.compile(r"Ownership As Of:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_TICKER_RE = re.compile(r"\b[A-Z0-9]{1,8}(?:[.-][A-Z0-9]{1,6})\b", re.IGNORECASE)


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


def _shareholder_from_fla(
    fla_report: str,
    pattern: re.Pattern[str],
    *,
    source: str,
) -> dict[str, Any] | None:
    if not fla_report:
        return None
    m = pattern.search(fla_report)
    if not m:
        return None
    return _controller_from_text(m.group(1).strip(), source=source)


def _fla_control_status(fla_report: str) -> ControlStatus:
    match = _FLA_CONTROL_STATUS_RE.search(fla_report)
    if not match:
        return "UNKNOWN"
    return cast(ControlStatus, match.group(1).upper())


def _fla_ownership_evidence(fla_report: str) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for key, pattern in (
        ("status", _FLA_EVIDENCE_STATUS_RE),
        ("source_url", _FLA_SOURCE_URL_RE),
        ("as_of", _FLA_AS_OF_RE),
    ):
        match = pattern.search(fla_report)
        if match:
            evidence[key] = match.group(1).strip()
    return evidence


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

    # Mixed families. If a clear majority emerges across senior/fla/hints (≥2 of
    # ≥3 sources on one side, with only one dissenter), treat the minority as
    # rejected and resolve to the majority's most specific role with "unresolved"
    # confidence rather than escalating to conflict. A 1-1 disagreement (no hints)
    # still goes to conflict.
    if len(opinions) >= 3:
        family_groups: dict[str, list[tuple[str, EntityRole]]] = {}
        for s, role in opinions:
            fam = "HOLDCO_FAMILY" if is_holdco_family(role) else "STANDALONE"
            family_groups.setdefault(fam, []).append((s, role))
        winning_family, winners = max(family_groups.items(), key=lambda kv: len(kv[1]))
        if len(winners) >= 2 and len(opinions) - len(winners) == 1:
            rejected = next(
                s
                for fam, lst in family_groups.items()
                if fam != winning_family
                for s, _ in lst
            )
            for src_order in ("senior", "fla", "hints"):
                for s, role in winners:
                    if s == src_order:
                        return (
                            role,
                            "unresolved",
                            f"role disagreement: hints+other corroborate {winning_family}, {rejected} rejected",
                        )

    disagreement = ", ".join(f"{s}={fam}" for s, fam in families.items())
    return "UNKNOWN", "conflict", f"role disagreement: {disagreement}"


def build_card(
    *,
    ticker: str,
    company_name: str,
    merged_data: dict[str, Any] | None,
    senior_metrics: dict[str, Any] | None,
    fla_report: str,
) -> EntityGovernanceCard:
    """Build the card from already-available pipeline artifacts. Pure function."""

    merged = merged_data or {}
    metrics = senior_metrics or {}

    hints, hints_role = deterministic_hints(merged)
    senior_role = _role_from_senior(metrics)
    fla_role = _role_from_fla(fla_report)

    role, confidence, notes = _reconcile_role(senior_role, fla_role, hints_role)

    ownership_evidence = _fla_ownership_evidence(fla_report)
    ownership_verified = ownership_evidence.get("status", "").startswith("VERIFIED")
    has_ownership_contract = bool(ownership_evidence.get("status"))

    related_from_senior = _parse_related_listed(metrics.get("related_listed_tickers"))
    related_from_fla = _related_from_fla(fla_report)
    # New-format FLA reports have deterministic evidence gating. Once that
    # contract is present, all Senior-only related-ticker edges are deliberately
    # dropped: Senior is downstream of FLA, not an independent ownership source,
    # so even a legitimate-looking restatement cannot corroborate the claim.
    related_by_ticker: dict[str, dict[str, Any]] = {}
    related_candidates = (
        related_from_fla
        if ownership_verified
        else []
        if has_ownership_contract
        else related_from_fla + related_from_senior
    )
    for edge in related_candidates:
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

    # Ownership roles are distinct. A largest shareholder is not a controller,
    # and internal Value Trap/Senior restatements are not independent evidence.
    # The FLA evidence normalizer is the sole promotion boundary.
    largest_shareholder = (
        _shareholder_from_fla(
            fla_report,
            _FLA_LARGEST_RE,
            source="fla_ownership",
        )
        if ownership_verified
        else None
    )
    control_status = (
        _fla_control_status(fla_report) if ownership_verified else "UNKNOWN"
    )
    controller = (
        _shareholder_from_fla(
            fla_report,
            _FLA_CONTROLLING_RE,
            source="fla_ownership",
        )
        if control_status == "CONTROLLED"
        else None
    )
    basis_match = _FLA_CONTROL_BASIS_RE.search(fla_report)
    control_basis = (
        basis_match.group(1).strip()
        if ownership_verified and basis_match
        else "UNKNOWN"
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
        largest_shareholder=largest_shareholder,
        controlling_shareholder=controller,
        control_status=control_status,
        control_basis=control_basis,
        ownership_evidence=ownership_evidence,
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
        control_status=control_status,
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

    if card.largest_shareholder:
        holder = card.largest_shareholder
        holder_name = holder.get("name", "?")
        holder_pct = holder.get("pct")
        holder_suffix = f" ({holder_pct}%)" if holder_pct is not None else ""
        lines.append(f"Largest shareholder: {holder_name}{holder_suffix}")

    lines.append(f"Control status: {card.control_status}")
    lines.append(f"Control basis: {card.control_basis}")
    if card.controlling_shareholder:
        cs = card.controlling_shareholder
        cs_name = cs.get("name", "?")
        cs_pct = cs.get("pct")
        suffix = f" ({cs_pct}%)" if cs_pct is not None else ""
        lines.append(f"Controlling shareholder: {cs_name}{suffix}")

    if card.related_listed:
        edges = "; ".join(
            f"{e.get('ticker')}:{e.get('relationship', '?')}:{e.get('pct', '?')}"
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
        "Largest shareholder and controlling shareholder are different concepts; "
        "do not infer control from a sub-50% stake or group affiliation. "
        "Only CONTROLLED authorizes parent/controller language. "
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
