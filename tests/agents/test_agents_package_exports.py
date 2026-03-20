from src import agents


def test_root_package_re_exports_split_modules():
    assert agents.create_analyst_node.__module__ == "src.agents.analyst_nodes"
    assert agents.create_researcher_node.__module__ == "src.agents.research_nodes"
    assert (
        agents.create_portfolio_manager_node.__module__ == "src.agents.decision_nodes"
    )
    assert agents.create_consultant_node.__module__ == "src.agents.consultant_nodes"
    assert agents.invoke_with_rate_limit_handling.__module__ == "src.agents.runtime"
