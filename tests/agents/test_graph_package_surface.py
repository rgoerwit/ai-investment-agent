from unittest.mock import patch


def test_graph_facade_exports_expected_symbols():
    from src.graph import (
        TradingContext,
        create_agent_tool_node,
        create_trading_graph,
        fan_out_to_analysts,
        fundamentals_sync_router,
        route_tools,
        should_continue_analyst,
        sync_check_router,
    )

    assert TradingContext is not None
    assert callable(create_trading_graph)
    assert callable(create_agent_tool_node)
    assert callable(should_continue_analyst)
    assert callable(route_tools)
    assert callable(fan_out_to_analysts)
    assert callable(fundamentals_sync_router)
    assert callable(sync_check_router)


@patch("src.graph.components.create_chart_generator_node")
@patch("src.graph.components.create_financial_health_validator_node")
@patch("src.graph.components.create_portfolio_manager_node")
@patch("src.graph.components.create_research_manager_node")
@patch("src.graph.components.create_researcher_node")
@patch("src.graph.components.create_trader_node")
@patch("src.graph.components.create_risk_debater_node")
@patch("src.graph.components.create_analyst_node")
@patch("src.graph.components.create_legal_counsel_node")
@patch("src.graph.components.create_agent_tool_node")
@patch("src.graph.components.create_quick_thinking_llm")
@patch("src.graph.components.create_deep_thinking_llm")
@patch("src.graph.components.get_consultant_llm")
@patch("src.graph.components.create_auditor_llm")
@patch("src.graph.components._is_auditor_enabled")
@patch("src.graph.components.toolkit")
def test_graph_facade_create_trading_graph_still_compiles(
    mock_toolkit,
    mock_auditor_enabled,
    mock_auditor_llm,
    mock_consultant_llm,
    mock_deep_llm,
    mock_quick_llm,
    mock_tool_node,
    mock_legal_counsel,
    mock_analyst,
    mock_risk,
    mock_trader,
    mock_researcher,
    mock_research_manager,
    mock_pm,
    mock_validator,
    mock_chart_generator,
):
    from src.graph import create_trading_graph

    mock_quick_llm.return_value = object()
    mock_deep_llm.return_value = object()
    mock_auditor_enabled.return_value = False
    mock_consultant_llm.return_value = None
    mock_auditor_llm.return_value = None
    mock_tool_node.return_value = lambda s, c: {}
    mock_legal_counsel.return_value = lambda s, c: {}
    mock_analyst.return_value = lambda s, c: {}
    mock_risk.return_value = lambda s, c: {}
    mock_trader.return_value = lambda s, c: {}
    mock_researcher.return_value = lambda s, c: {}
    mock_research_manager.return_value = lambda s, c: {}
    mock_pm.return_value = lambda s, c: {}
    mock_validator.return_value = lambda s, c: {}
    mock_chart_generator.return_value = lambda s, c: {}
    mock_toolkit.get_market_tools.return_value = []
    mock_toolkit.get_technical_tools.return_value = []
    mock_toolkit.get_sentiment_tools.return_value = []
    mock_toolkit.get_news_tools.return_value = []
    mock_toolkit.get_junior_fundamental_tools.return_value = []
    mock_toolkit.get_foreign_language_tools.return_value = []
    mock_toolkit.get_legal_tools.return_value = []
    mock_toolkit.get_value_trap_tools.return_value = []
    mock_toolkit.get_senior_fundamental_tools.return_value = []

    graph = create_trading_graph(enable_memory=False, max_debate_rounds=1)

    assert graph is not None
