"""Canonical, serializable claims derived from deterministic analysis state."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from src.claim_policy import (
    MATERIAL_CLAIM_POLICIES,
    RAW_FINANCIAL_METRICS_INPUT,
    Authority,
    ClaimKind,
    ClaimPolicy,
    ClaimSource,
    Coverage,
    DecisionRole,
    Exactness,
)
from src.data_block_utils import (
    extract_last_data_block,
    replace_or_append_block_line,
)
from src.provenance_schema import (
    SchemaDecodeError,
    classify_schema_version,
)
from src.tooling.evidence_recorder import bind_fetched_evidence
from src.tooling.structured_ingress import get_structured_ingress_payload

_FIELD_RE = re.compile(r"(?m)^\s*(?:[-*]\s*)?([A-Z][A-Z0-9_]{2,})\s*:\s*(.*?)\s*$")
_UNKNOWN_VALUES = frozenset({"", "N/A", "NA", "NONE", "UNKNOWN", "NOT FOUND"})
_ESTIMATE_MARKERS = ("ESTIMAT", "APPROX", "ROUGHLY", "CIRCA", "~", "≈")


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    id: str
    field: str
    value: str
    period: str | None
    authority: Authority
    exactness: Exactness
    coverage: Coverage
    source_url: str | None
    evidence_id: str
    decision_eligible: bool
    kind: ClaimKind = "FACT"
    decision_role: DecisionRole = "CONTEXT"
    source_provider: str | None = None
    lineage_ids: tuple[str, ...] = ()
    derived_from: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnalysisSnapshot:
    """Shallow, versioned wire codec for the canonical analysis snapshot.

    Deliberately shallow: ``claims`` and ``conflicts`` stay in their existing
    dict wire form (``claims`` is already ``asdict(ClaimRecord)``), so this is a
    single typed definition of the snapshot's key set + schema versioning + a
    fail-closed loader — not a re-typing of every claim.

    ``stage`` / ``commentary_status`` / ``scorecards`` are None-omitted by
    ``to_dict`` so the reduced INVALID snapshot shape is reproduced exactly (it
    carries neither ``stage`` nor ``commentary_status``). ``schema_version`` is
    appended last; nothing else in the wire shape changes.

    Only the two primary builders (``_snapshot_from_fields`` and the reduced
    INVALID form) serialize through this codec. ``add_validated_derivations`` and
    ``refresh_analysis_snapshot`` keep their ``{**snapshot, ...}`` spreads: a
    spread cannot add or drop a key, so it inherits this canonical shape (incl.
    ``schema_version``) from its codec-produced input — drift is structurally
    impossible without re-serializing (which would risk dropping an unknown key).
    """

    SCHEMA_VERSION: ClassVar[int] = 1

    version: int
    contract_status: str
    contract_reason: str | None
    claims: dict[str, Any]
    conflicts: list[Any]
    stage: str | None = None
    commentary_status: str | None = None
    scorecards: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"version": self.version}
        if self.stage is not None:
            payload["stage"] = self.stage
        payload["contract_status"] = self.contract_status
        payload["contract_reason"] = self.contract_reason
        payload["claims"] = self.claims
        payload["conflicts"] = self.conflicts
        if self.commentary_status is not None:
            payload["commentary_status"] = self.commentary_status
        if self.scorecards is not None:
            payload["scorecards"] = self.scorecards
        payload["schema_version"] = self.SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AnalysisSnapshot:
        if not isinstance(d, Mapping):
            raise SchemaDecodeError(
                f"analysis_snapshot must be a mapping, got {type(d).__name__}"
            )
        status = classify_schema_version(d.get("schema_version"), cls.SCHEMA_VERSION)
        if not status.compatible:
            raise SchemaDecodeError(
                "AnalysisSnapshot: incompatible schema_version "
                f"{d.get('schema_version')!r}"
            )
        contract_status = d.get("contract_status")
        if contract_status is None:
            # Match the historical coercion (missing status → INVALID); a
            # present-but-wrong-type status is a genuine corruption → fail closed.
            contract_status = "INVALID"
        elif not isinstance(contract_status, str):
            raise SchemaDecodeError(
                "analysis_snapshot.contract_status must be a string, got "
                f"{type(contract_status).__name__}"
            )
        version = d.get("version")
        return cls(
            version=(
                version
                if isinstance(version, int) and not isinstance(version, bool)
                else 1
            ),
            contract_status=contract_status,
            contract_reason=d.get("contract_reason"),
            claims=dict(d.get("claims") or {}),
            conflicts=list(d.get("conflicts") or []),
            stage=d.get("stage"),
            commentary_status=d.get("commentary_status"),
            scorecards=d.get("scorecards"),
        )


def _normalize_authority(value: str | None, *, default: Authority) -> Authority:
    token = str(value or "").strip().upper()
    if token in {"PRIMARY", "PRIMARY_ISSUER", "PRIMARY_REGISTRY", "FILING"}:
        return "PRIMARY"
    if token in {"SECONDARY", "THIRD_PARTY"}:
        return "SECONDARY"
    if token in {"AGGREGATOR", "JUNIOR", "CALCULATED_FROM_QUARTERLY"}:
        return "AGGREGATOR"
    if token in {"UNSUPPORTED", "QUARANTINED"}:
        return "UNSUPPORTED"
    if token in {"UNKNOWN", "N/A", "NA", "NONE", ""}:
        return default
    return "AGGREGATOR"


def claim_id(field: str, period: str | None) -> str:
    identity = f"{field}|{period or 'UNDATED'}"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    return f"claim:{field.lower()}:{suffix}"


def _bound_authority(authority: str) -> Authority:
    if authority in {"PRIMARY_REGISTRY", "PRIMARY_ISSUER"}:
        return "PRIMARY"
    if authority == "SECONDARY":
        return "SECONDARY"
    return "UNSUPPORTED"


def _cap_authority(declared: Authority, bound: Authority) -> Authority:
    rank = {
        "UNSUPPORTED": 0,
        "UNKNOWN": 0,
        "AGGREGATOR": 1,
        "SECONDARY": 2,
        "PRIMARY": 3,
    }
    return declared if rank[declared] <= rank[bound] else bound


def _coverage_from_fields(
    fields: Mapping[str, str],
    policy: ClaimPolicy,
    *,
    missing: bool,
    authority: Authority,
) -> Coverage:
    token = str(fields.get(policy.input_coverage_field or "", "")).strip().upper()
    if token in {"NOT_DISCLOSED_AFTER_TARGETED_SEARCH", "NOT_FOUND", "NOT_APPLICABLE"}:
        return "COMPLETE_NO_MATCH"
    if token == "UNRESOLVED_AFTER_TARGETED_SEARCH":
        return "SEARCHED_UNRESOLVED"
    if token in {"SEARCH_FAILED", "AUTH_ERROR", "UNAVAILABLE", "INSUFFICIENT"}:
        return "FAILED"
    if authority == "UNSUPPORTED":
        return "UNSUPPORTED"
    return "MISSING" if missing else "FOUND"


def _raw_metrics_contract_status(
    payload: Mapping[str, Any],
    ingress_error: str | None,
) -> tuple[str, str | None]:
    if ingress_error:
        return "INVALID", ingress_error
    registered = {
        policy.raw_field
        for policy in MATERIAL_CLAIM_POLICIES.values()
        if policy.source == "RAW_METRICS" and policy.raw_field
    }
    if not registered.intersection(payload):
        return "INVALID", "RAW_METRICS_PAYLOAD_HAS_NO_REGISTERED_FIELDS"
    analytic_policies = tuple(
        policy
        for policy in MATERIAL_CLAIM_POLICIES.values()
        if policy.source == "RAW_METRICS"
        and policy.raw_field
        and policy.raw_field not in {"currentPrice", "marketCap"}
    )
    usable = {
        policy.raw_field
        for policy in analytic_policies
        if policy.raw_field in payload
        and (
            (
                isinstance(payload.get(policy.raw_field), str)
                and str(payload.get(policy.raw_field)).strip().upper()
                not in _UNKNOWN_VALUES
            )
            if policy.value_format == "TEXT"
            else _is_finite_number(payload.get(policy.raw_field))
        )
    }
    if not usable:
        return "DEGRADED", "RAW_METRICS_NO_USABLE_ANALYTIC_FIELDS"
    return "VALID", None


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _snapshot_from_fields(
    fields: Mapping[str, str],
    evidence_records: Sequence[Any],
    *,
    version: int,
    stage: str,
    providers: Mapping[str, str] | None = None,
    include_sources: frozenset[ClaimSource] | None = None,
    contract_status: str = "VALID",
    contract_reason: str | None = None,
) -> dict[str, Any]:
    claims: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, str]] = []
    for field, policy in MATERIAL_CLAIM_POLICIES.items():
        if include_sources is not None and policy.source not in include_sources:
            continue
        if field not in fields and not policy.source_required:
            continue
        value = str(fields.get(field, "N/A")).strip()
        period = fields.get(policy.period_field) if policy.period_field else None
        if period and period.strip().upper() in _UNKNOWN_VALUES:
            period = None
        source_url = (
            fields.get(policy.source_url_field) if policy.source_url_field else None
        )
        if source_url and source_url.strip().upper() in _UNKNOWN_VALUES:
            source_url = None
        declared = _normalize_authority(
            fields.get(policy.authority_field) if policy.authority_field else None,
            default="UNKNOWN" if policy.source_required else "AGGREGATOR",
        )
        binding = bind_fetched_evidence(list(evidence_records), source_url)
        evidence_id = binding.evidence_id if binding else None
        authority = declared
        missing = value.upper() in _UNKNOWN_VALUES
        if policy.source_required:
            if binding:
                authority = _cap_authority(
                    declared,
                    _bound_authority(binding.authority),
                )
                source_url = binding.canonical_url
            elif not missing:
                authority = "UNSUPPORTED"
                conflicts.append(
                    {
                        "field": field,
                        "type": "SOURCE_BINDING_MISSING",
                        "detail": "Claim has no matching fetched evidence record.",
                    }
                )
        exactness: Exactness = (
            "ESTIMATED"
            if any(marker in value.upper() for marker in _ESTIMATE_MARKERS)
            else "CALCULATED"
            if policy.kind == "DERIVED_ASSESSMENT"
            or str(fields.get(policy.authority_field or "", "")).upper()
            == "CALCULATED_FROM_QUARTERLY"
            or str((providers or {}).get(field) or "").upper()
            == "CALCULATED_FROM_QUARTERLY"
            else "EXACT"
        )
        coverage = _coverage_from_fields(
            fields,
            policy,
            missing=missing,
            authority=authority,
        )
        if policy.source_required and (
            coverage != "FOUND" or authority in {"UNSUPPORTED", "UNKNOWN"}
        ):
            value = "N/A"
            missing = True
        provider = str((providers or {}).get(field) or "").strip() or None
        lineage_id = evidence_id or (
            f"raw:{provider or 'unknown'}:{policy.raw_field or field}"
            if policy.source == "RAW_METRICS"
            else f"state:{stage}:{field}"
        )
        record = ClaimRecord(
            id=claim_id(field, period),
            field=field,
            value=value,
            period=period,
            authority=authority,
            exactness=exactness,
            coverage=coverage,
            source_url=source_url,
            evidence_id=lineage_id,
            decision_eligible=(
                contract_status == "VALID"
                and not missing
                and coverage == "FOUND"
                and authority not in {"UNSUPPORTED", "UNKNOWN"}
                and (not policy.source_required or evidence_id is not None)
            ),
            kind=policy.kind,
            decision_role=policy.decision_role,
            source_provider=provider,
            lineage_ids=(lineage_id,),
        )
        claims[record.id] = asdict(record)

    return AnalysisSnapshot(
        version=version,
        stage=stage,
        contract_status=contract_status,
        contract_reason=contract_reason,
        claims=claims,
        conflicts=conflicts,
        commentary_status="NON_AUTHORITATIVE_UNLESS_CLAIM_REFERENCED",
    ).to_dict()


def build_analysis_snapshot(
    state: Mapping[str, Any],
    evidence_records: Sequence[Any] = (),
    *,
    version: int = 1,
    degraded: bool = True,
) -> dict[str, Any]:
    """Build a compatibility snapshot from the post-Senior DATA_BLOCK."""
    fundamentals = state.get("fundamentals_report")
    block = extract_last_data_block(
        fundamentals if isinstance(fundamentals, str) else ""
    )
    if not block:
        return AnalysisSnapshot(
            version=version,
            contract_status="INVALID",
            contract_reason="DATA_BLOCK_MISSING_OR_UNPARSEABLE",
            claims={},
            conflicts=[],
        ).to_dict()

    fields: dict[str, str] = {
        match.group(1): match.group(2).strip() for match in _FIELD_RE.finditer(block)
    }
    return _snapshot_from_fields(
        fields,
        evidence_records,
        version=version,
        stage="LEGACY_POST_SENIOR",
        contract_status="DEGRADED" if degraded else "VALID",
        contract_reason=("PRE_SENIOR_SNAPSHOT_UNAVAILABLE" if degraded else None),
    )


def build_pre_senior_snapshot(
    state: Mapping[str, Any],
    evidence_records: Sequence[Any] = (),
    *,
    version: int = 1,
) -> dict[str, Any]:
    """Mint code-owned base claims before Senior Fundamentals runs."""
    from src.agents.foreign_language_evidence import promote_foreign_growth_evidence
    from src.agents.fundamentals_reconciler import (
        as_float,
        format_percent_from_ratio,
        format_ratio,
    )
    from src.agents.management_guidance import promote_management_guidance

    payload, ingress_error = get_structured_ingress_payload(
        state,
        RAW_FINANCIAL_METRICS_INPUT,
    )
    payload = payload or {}
    contract_status, contract_reason = _raw_metrics_contract_status(
        payload,
        ingress_error,
    )
    foreign = state.get("foreign_language_report")
    foreign_text = foreign if isinstance(foreign, str) else ""
    promoted, _ = promote_management_guidance("", foreign_text)
    promoted, _ = promote_foreign_growth_evidence(promoted, foreign_text)
    fields = {
        match.group(1): match.group(2).strip() for match in _FIELD_RE.finditer(promoted)
    }
    providers: dict[str, str] = {}
    field_sources = payload.get("_field_sources")
    if not isinstance(field_sources, Mapping):
        field_sources = {}

    for field, policy in MATERIAL_CLAIM_POLICIES.items():
        if policy.source != "RAW_METRICS" or not policy.raw_field:
            continue
        raw_value = payload.get(policy.raw_field)
        value = None if policy.value_format == "TEXT" else as_float(raw_value)
        if field == "PE_RATIO_TTM" and payload.get("_pe_low_anomaly_quarantined"):
            value = None
        if field in {"PE_RATIO_FORWARD", "FORWARD_EPS"} and (
            payload.get("_split_sensitive_metrics_quarantined")
            or payload.get("_pe_unit_error_quarantined") == "forward"
            or (
                field == "PE_RATIO_FORWARD"
                and payload.get("_forwardPE_quarantine_reason")
            )
        ):
            value = None
        if policy.value_format == "TEXT":
            fields[field] = str(raw_value).strip() if raw_value is not None else "N/A"
        else:
            fields[field] = (
                format_percent_from_ratio(value)
                if value is not None and policy.value_format == "PERCENT_RATIO"
                else format_ratio(value)
                if value is not None
                else "N/A"
            )
        if field in {"REVENUE_GROWTH_MRQ", "EARNINGS_GROWTH_MRQ"}:
            source_marker = payload.get(
                "_revenueGrowth_MRQ_source"
                if field == "REVENUE_GROWTH_MRQ"
                else "_earningsGrowth_MRQ_source"
            )
            if source_marker == "calculated_from_quarterly":
                period = payload.get("latest_quarter_date")
                if policy.period_field:
                    fields[policy.period_field] = str(period or "N/A")
            elif policy.period_field:
                fields[policy.period_field] = "N/A"
            providers[field] = str(
                source_marker or field_sources.get(policy.raw_field) or "UNKNOWN"
            )
        else:
            providers[field] = str(
                payload.get(f"_{policy.raw_field}_source")
                or field_sources.get(policy.raw_field)
                or "UNKNOWN"
            )

    return _snapshot_from_fields(
        fields,
        evidence_records,
        version=version,
        stage="PRE_SENIOR",
        providers=providers,
        include_sources=frozenset({"RAW_METRICS", "FOREIGN_BLOCK"}),
        contract_status=contract_status,
        contract_reason=contract_reason,
    )


def reconcile_data_block_projection(
    body: str,
    snapshot: Mapping[str, Any] | None,
) -> tuple[str, list[dict[str, str]]]:
    """Make Senior's registered fact fields a projection of upstream claims."""
    if not snapshot or snapshot.get("contract_status") != "VALID":
        return body, []
    updated = body
    conflicts: list[dict[str, str]] = []
    field_order = {field: index for index, field in enumerate(MATERIAL_CLAIM_POLICIES)}
    claims = sorted(
        snapshot.get("claims", {}).values(),
        key=lambda claim: (
            field_order.get(str(claim.get("field") or ""), len(field_order))
            if isinstance(claim, Mapping)
            else len(field_order),
            str(claim.get("id") or "") if isinstance(claim, Mapping) else "",
        ),
    )
    for claim in claims:
        if not isinstance(claim, Mapping) or claim.get("kind") != "FACT":
            continue
        field = str(claim.get("field") or "")
        policy = MATERIAL_CLAIM_POLICIES.get(field)
        if not policy or policy.source == "LEGACY" or not policy.project_to_report:
            continue
        canonical = str(claim.get("value") or "N/A")
        current_match = next(
            (m for m in _FIELD_RE.finditer(updated) if m.group(1) == field),
            None,
        )
        current = current_match.group(2).strip() if current_match else None
        if current != canonical:
            conflicts.append(
                {
                    "field": field,
                    "type": "SENIOR_PROJECTION_CONFLICT",
                    "detail": f"Senior emitted {current or 'MISSING'}; canonical is {canonical}.",
                }
            )
            updated = replace_or_append_block_line(updated, field, canonical)
        if "period" in policy.projected_metadata and policy.period_field:
            updated = replace_or_append_block_line(
                updated,
                policy.period_field,
                str(claim.get("period") or "UNKNOWN"),
            )
        if "source_url" in policy.projected_metadata and policy.source_url_field:
            updated = replace_or_append_block_line(
                updated,
                policy.source_url_field,
                str(claim.get("source_url") or "N/A"),
            )
        if "authority" in policy.projected_metadata and policy.authority_field:
            provider = claim.get("source_provider")
            authority = str(claim.get("authority") or "UNKNOWN")
            projected_authority = (
                str(provider)
                if policy.source == "RAW_METRICS" and provider
                else authority
            )
            updated = replace_or_append_block_line(
                updated,
                policy.authority_field,
                projected_authority,
            )
    mrq_claims = [
        claim
        for claim in snapshot.get("claims", {}).values()
        if isinstance(claim, Mapping)
        and claim.get("field") in {"REVENUE_GROWTH_MRQ", "EARNINGS_GROWTH_MRQ"}
        and str(claim.get("value") or "").upper() not in _UNKNOWN_VALUES
    ]
    mrq_periods = {
        str(claim.get("period")) for claim in mrq_claims if claim.get("period")
    }
    shared_period = (
        next(iter(mrq_periods))
        if mrq_claims
        and len(mrq_periods) == 1
        and all(claim.get("period") for claim in mrq_claims)
        else "UNKNOWN"
    )
    updated = replace_or_append_block_line(
        updated,
        "LATEST_QUARTER_DATE",
        shared_period,
    )
    return updated, conflicts


def project_analysis_report(
    report: str,
    snapshot: Mapping[str, Any] | None,
) -> str:
    """Render canonical facts and scorecards into the fundamentals report."""
    from src.score_lineage import project_analysis_report as project

    return project(report, snapshot)


def add_validated_derivations(
    snapshot: Mapping[str, Any] | None,
    fundamentals_report: str,
    *,
    conflicts: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    """Add score assessments only when their rubric projection is coherent."""
    from src.score_lineage import add_validated_derivations as add_derivations

    return add_derivations(
        snapshot,
        fundamentals_report,
        conflicts=conflicts,
    )


def refresh_analysis_snapshot(
    prior: Mapping[str, Any] | None,
    state: Mapping[str, Any],
    evidence_records: Sequence[Any],
    *,
    version: int,
) -> dict[str, Any]:
    """Monotonically augment a snapshot without rebuilding truth from prose."""
    candidate = build_pre_senior_snapshot(
        state,
        evidence_records,
        version=version,
    )
    if not prior or prior.get("contract_status") != "VALID":
        merged = candidate
    else:
        claims = dict(prior.get("claims", {}))
        conflicts = list(prior.get("conflicts", []) or [])
        claim_ids_by_field = {
            str(claim.get("field")): str(claim_id)
            for claim_id, claim in claims.items()
            if isinstance(claim, Mapping)
        }
        authority_rank = {
            "UNSUPPORTED": 0,
            "UNKNOWN": 0,
            "AGGREGATOR": 1,
            "SECONDARY": 2,
            "PRIMARY": 3,
        }
        for claim_id, new_claim in candidate.get("claims", {}).items():
            field = str(new_claim.get("field") or "")
            old_claim_id = (
                claim_id if claim_id in claims else claim_ids_by_field.get(field)
            )
            old_claim = claims.get(old_claim_id) if old_claim_id else None
            if not isinstance(old_claim, Mapping):
                claims[claim_id] = new_claim
                claim_ids_by_field[field] = claim_id
                continue
            old_rank = authority_rank.get(str(old_claim.get("authority")), 0)
            new_rank = authority_rank.get(str(new_claim.get("authority")), 0)
            old_period = str(old_claim.get("period") or "")
            new_period = str(new_claim.get("period") or "")
            newer_iso_period = bool(
                re.fullmatch(r"\d{4}-\d{2}-\d{2}", old_period)
                and re.fullmatch(r"\d{4}-\d{2}-\d{2}", new_period)
                and new_period > old_period
            )
            if new_claim.get("decision_eligible") and (
                not old_claim.get("decision_eligible")
                or new_rank > old_rank
                or (new_rank == old_rank and newer_iso_period)
            ):
                if old_claim_id and old_claim_id != claim_id:
                    claims.pop(old_claim_id, None)
                claims[claim_id] = new_claim
                claim_ids_by_field[field] = claim_id
            elif old_claim.get("value") != new_claim.get("value") or old_claim.get(
                "period"
            ) != new_claim.get("period"):
                conflicts.append(
                    {
                        "field": field,
                        "type": "MONOTONIC_CLAIM_CONFLICT",
                        "detail": "Later state attempted to change an existing claim identity.",
                    }
                )
        merged = {
            **prior,
            "version": version,
            "stage": "POST_EVIDENCE_REFRESH",
            "claims": claims,
            "conflicts": conflicts,
        }
    fundamentals = state.get("fundamentals_report")
    if isinstance(fundamentals, str) and fundamentals:
        return add_validated_derivations(merged, fundamentals)
    return merged


def render_analysis_snapshot(snapshot: Mapping[str, Any] | None) -> str:
    """Render a bounded authoritative block for downstream prompts."""
    if not snapshot:
        return ""
    if snapshot.get("contract_status") not in {"VALID", "DEGRADED"}:
        return (
            "=== CANONICAL ANALYSIS SNAPSHOT ===\n"
            f"CONTRACT_STATUS: {snapshot.get('contract_status', 'INVALID')}\n"
            f"REASON: {snapshot.get('contract_reason', 'UNKNOWN')}\n"
        )
    lines = [
        "=== CANONICAL ANALYSIS SNAPSHOT ===",
        f"VERSION: {snapshot.get('version', 1)}",
        f"CONTRACT_STATUS: {snapshot.get('contract_status', 'INVALID')}",
        (
            "RULE: These claims override conflicting agent prose. Prose may interpret "
            "them but may not change their value, period, authority, or eligibility."
            if snapshot.get("contract_status") == "VALID"
            else "RULE: Degraded compatibility reference only. These claims do not "
            "override upstream evidence and are not decision-eligible."
        ),
    ]
    claims = snapshot.get("claims", {})
    material_fields = set(MATERIAL_CLAIM_POLICIES)
    selected = [
        claim
        for claim in claims.values()
        if isinstance(claim, Mapping) and claim.get("field") in material_fields
    ]
    for claim in selected[:40]:
        lines.append(
            f"{claim['id']} | {claim['field']}: {claim['value']} | "
            f"period={claim.get('period') or 'UNDATED'} | "
            f"authority={claim['authority']} | coverage={claim['coverage']} | "
            f"decision_eligible={'YES' if claim['decision_eligible'] else 'NO'}"
        )
    return "\n".join(lines) + "\n"


def decision_claim_ids(snapshot: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return eligible material claim IDs in stable field order."""
    if not snapshot:
        return ()
    claims = snapshot.get("claims", {})
    ordered = sorted(
        (
            claim
            for claim in claims.values()
            if isinstance(claim, Mapping)
            and claim.get("decision_eligible")
            and MATERIAL_CLAIM_POLICIES.get(
                str(claim.get("field")), ClaimPolicy()
            ).decision_role
            == "SUPPORT"
        ),
        key=lambda claim: str(claim.get("field")),
    )
    return tuple(str(claim["id"]) for claim in ordered)
