"""
Public agent package surface.

The implementation now lives in focused submodules:
- runtime.py
- state.py
- message_utils.py
- support.py
- analyst_nodes.py
- research_nodes.py
- decision_nodes.py
- consultant_nodes.py
"""

from . import decision_nodes as _decision_nodes
from . import research_nodes as _research_nodes
from . import support as _support
from .analyst_nodes import create_analyst_node, create_valuation_calculator_node
from .consultant_nodes import (
    create_auditor_node,
    create_consultant_node,
    create_legal_counsel_node,
)
from .decision_nodes import (
    create_financial_health_validator_node,
    create_portfolio_manager_node,
    create_risk_debater_node,
    create_state_cleaner_node,
    create_trader_node,
)
from .message_utils import (
    extract_string_content,
    filter_messages_by_agent,
    filter_messages_for_gemini,
)
from .research_nodes import (
    create_research_manager_node,
    create_researcher_node,
)
from .runtime import invoke_with_rate_limit_handling
from .state import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
    merge_dicts,
    merge_invest_debate_state,
    merge_risk_state,
    take_last,
)
from .support import (
    compute_data_conflicts,
    extract_field_sources_from_messages,
    extract_news_highlights,
    extract_source_conflicts_from_messages,
    extract_value_trap_verdict,
    format_attribution_table,
    format_conflict_table,
    get_analysis_context,
    get_context_from_config,
    summarize_for_pm,
)

_STRICT_PM_ADDENDUM = _decision_nodes._STRICT_PM_ADDENDUM
_STRICT_RM_ADDENDUM = _research_nodes._STRICT_RM_ADDENDUM
_UNRESOLVED_NAME_WARNING = _support._UNRESOLVED_NAME_WARNING
_company_line = _support._company_line
_extract_sector_country = _support._extract_sector_country
_extract_sector_from_state = _support._extract_sector_from_state
_format_date_with_fy_hint = _support._format_date_with_fy_hint

__all__ = [
    "AgentState",
    "InvestDebateState",
    "RiskDebateState",
    "compute_data_conflicts",
    "create_analyst_node",
    "create_auditor_node",
    "create_consultant_node",
    "create_financial_health_validator_node",
    "create_legal_counsel_node",
    "create_portfolio_manager_node",
    "create_research_manager_node",
    "create_researcher_node",
    "create_risk_debater_node",
    "create_state_cleaner_node",
    "create_trader_node",
    "create_valuation_calculator_node",
    "extract_field_sources_from_messages",
    "extract_news_highlights",
    "extract_source_conflicts_from_messages",
    "extract_string_content",
    "extract_value_trap_verdict",
    "filter_messages_by_agent",
    "filter_messages_for_gemini",
    "format_attribution_table",
    "format_conflict_table",
    "get_analysis_context",
    "get_context_from_config",
    "invoke_with_rate_limit_handling",
    "merge_dicts",
    "merge_invest_debate_state",
    "merge_risk_state",
    "summarize_for_pm",
    "take_last",
]
