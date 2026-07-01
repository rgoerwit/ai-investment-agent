"""Public validator facade preserving the historical RedFlagDetector API."""

from __future__ import annotations

from src.validators.financial_rules import (
    detect_ocf_corroboration_flag,
    detect_red_flags,
    extract_auditor_ocf,
    is_ocf_period_mismatch_resolved,
    parse_ocf_amount,
    reconcile_ocf_period_mismatch_flags,
)
from src.validators.metric_extractor import extract_debt_to_equity, extract_metrics
from src.validators.sector_classifier import Sector, detect_sector
from src.validators.supplemental_extractors import (
    extract_capital_efficiency_signals,
    extract_legal_risks,
    extract_moat_signals,
    extract_value_trap_score,
    parse_consultant_conditions,
)
from src.validators.supplemental_flags import (
    detect_capital_efficiency_flags,
    detect_consultant_flags,
    detect_legal_flags,
    detect_material_operating_signal_flags,
    detect_moat_flags,
    detect_return_quality_fragility_flags,
    detect_shareholder_return_execution_flags,
    detect_value_trap_flags,
)


class RedFlagDetector:
    """Compatibility facade for validator parsing and flag-generation helpers."""

    detect_sector = staticmethod(detect_sector)
    extract_metrics = staticmethod(extract_metrics)
    _extract_debt_to_equity = staticmethod(extract_debt_to_equity)
    detect_red_flags = staticmethod(detect_red_flags)
    detect_ocf_corroboration_flag = staticmethod(detect_ocf_corroboration_flag)
    is_ocf_period_mismatch_resolved = staticmethod(is_ocf_period_mismatch_resolved)
    reconcile_ocf_period_mismatch_flags = staticmethod(
        reconcile_ocf_period_mismatch_flags
    )
    extract_auditor_ocf = staticmethod(extract_auditor_ocf)
    parse_ocf_amount = staticmethod(parse_ocf_amount)
    extract_legal_risks = staticmethod(extract_legal_risks)
    detect_legal_flags = staticmethod(detect_legal_flags)
    extract_value_trap_score = staticmethod(extract_value_trap_score)
    detect_value_trap_flags = staticmethod(detect_value_trap_flags)
    extract_moat_signals = staticmethod(extract_moat_signals)
    detect_moat_flags = staticmethod(detect_moat_flags)
    detect_return_quality_fragility_flags = staticmethod(
        detect_return_quality_fragility_flags
    )
    extract_capital_efficiency_signals = staticmethod(
        extract_capital_efficiency_signals
    )
    detect_capital_efficiency_flags = staticmethod(detect_capital_efficiency_flags)
    detect_shareholder_return_execution_flags = staticmethod(
        detect_shareholder_return_execution_flags
    )
    parse_consultant_conditions = staticmethod(parse_consultant_conditions)
    detect_consultant_flags = staticmethod(detect_consultant_flags)
    detect_material_operating_signal_flags = staticmethod(
        detect_material_operating_signal_flags
    )


__all__ = ["RedFlagDetector", "Sector"]
