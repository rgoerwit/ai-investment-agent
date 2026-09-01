"""Fixed test_graph_execution.py - removed pytestmark from non-async tests."""

import sys
from types import ModuleType
from unittest.mock import MagicMock, Mock, patch

import pytest


def _stub_graph_component_dependencies(monkeypatch):
    import src.graph.components as components

    def stub_node(*args, **kwargs):
        def _stub_runtime_node(state, config):
            return {}

        return _stub_runtime_node

    stub_memory = tuple(MagicMock() for _ in range(5))

    monkeypatch.setattr(components, "_create_legacy_memories", lambda: stub_memory)
    monkeypatch.setattr(
        components, "create_quick_thinking_llm", lambda **kwargs: Mock()
    )
    monkeypatch.setattr(components, "create_deep_thinking_llm", lambda **kwargs: Mock())
    monkeypatch.setattr(components, "create_analyst_node", stub_node)
    monkeypatch.setattr(components, "create_apac_specialist_llm", lambda **kwargs: None)
    monkeypatch.setattr(components, "create_apac_specialist_node", stub_node)
    monkeypatch.setattr(components, "create_auditor_node", stub_node)
    monkeypatch.setattr(components, "create_consultant_node", stub_node)
    monkeypatch.setattr(components, "create_financial_health_validator_node", stub_node)
    monkeypatch.setattr(components, "create_legal_counsel_node", stub_node)
    monkeypatch.setattr(components, "create_portfolio_manager_node", stub_node)
    monkeypatch.setattr(components, "create_research_manager_node", stub_node)
    monkeypatch.setattr(components, "create_researcher_node", stub_node)
    monkeypatch.setattr(components, "create_risk_debater_node", stub_node)
    monkeypatch.setattr(components, "create_trader_node", stub_node)
    monkeypatch.setattr(components, "create_valuation_calculator_node", stub_node)
    monkeypatch.setattr(
        components, "create_chart_generator_node", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        components,
        "create_agent_tool_node",
        lambda *args, **kwargs: lambda state, config: {},
    )

    def empty_tools():
        return []

    monkeypatch.setattr(components.toolkit, "get_market_tools", empty_tools)
    monkeypatch.setattr(components.toolkit, "get_technical_tools", empty_tools)
    monkeypatch.setattr(components.toolkit, "get_sentiment_tools", empty_tools)
    monkeypatch.setattr(components.toolkit, "get_news_tools", empty_tools)
    monkeypatch.setattr(components.toolkit, "get_junior_fundamental_tools", empty_tools)
    monkeypatch.setattr(components.toolkit, "get_senior_fundamental_tools", empty_tools)
    monkeypatch.setattr(components.toolkit, "get_foreign_language_tools", empty_tools)
    monkeypatch.setattr(components.toolkit, "get_legal_tools", empty_tools)
    monkeypatch.setattr(components.toolkit, "get_value_trap_tools", empty_tools)
    monkeypatch.setattr(components.toolkit, "get_all_tools", empty_tools)

    return components


class TestGraphRouting:
    """Test graph routing functions."""

    def test_should_continue_analyst_with_tools(self):
        """Test routing when analyst has tool calls."""
        from langchain_core.messages import AIMessage

        from src.graph import should_continue_analyst

        mock_message = AIMessage(
            content="",
            tool_calls=[{"name": "tool", "args": {}, "id": "1"}],
            name="market_analyst",
        )

        state = {"messages": [mock_message], "sender": "market_analyst"}
        config = {}

        result = should_continue_analyst(state, config)
        assert result == "tools"

    def test_should_continue_analyst_without_tools(self):
        """Test routing when analyst has no tool calls."""
        from langchain_core.messages import AIMessage

        from src.graph import should_continue_analyst

        mock_message = AIMessage(content="done", name="market_analyst")

        state = {"messages": [mock_message], "sender": "market_analyst"}
        config = {}

        result = should_continue_analyst(state, config)
        assert result == "continue"

    def test_should_continue_uses_sender_owned_message_under_parallel_interleaving(
        self,
    ):
        """Another branch finishing later must not control this branch's edge."""
        from langchain_core.messages import AIMessage

        from src.graph import should_continue_analyst

        state = {
            "sender": "news_analyst",
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_news", "args": {}, "id": "news-call"}],
                    name="news_analyst",
                ),
                AIMessage(content="market complete", name="market_analyst"),
            ],
        }

        assert should_continue_analyst(state, {}) == "tools"

    def test_should_continue_ends_turn_when_sender_has_no_owned_response(self):
        from langchain_core.messages import AIMessage

        from src.graph import should_continue_analyst

        state = {
            "sender": "news_analyst",
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "tool", "args": {}, "id": "market"}],
                    name="market_analyst",
                )
            ],
        }

        assert should_continue_analyst(state, {}) == "continue"


class TestDebateRouter:
    """Test debate routing logic."""

    @patch("src.graph.components.create_agent_tool_node")
    @patch("src.graph.components.create_analyst_node")
    @patch("src.graph.components.create_researcher_node")
    @patch("src.graph.components.create_research_manager_node")
    @patch("src.graph.components.create_trader_node")
    @patch("src.graph.components.create_risk_debater_node")
    @patch("src.graph.components.create_portfolio_manager_node")
    @patch("src.graph.components.toolkit")
    def test_debate_router_alternation(
        self,
        mock_toolkit,
        mock_pm,
        mock_risk,
        mock_trader,
        mock_res_mgr,
        mock_researcher,
        mock_analyst,
        mock_tool_node,
    ):
        """Test debate router alternates correctly."""
        from src.graph import create_trading_graph

        # Mock all node creation (cleaner nodes removed in parallel refactor)
        mock_analyst.return_value = lambda s, c: {}
        mock_researcher.return_value = lambda s, c: {}
        mock_res_mgr.return_value = lambda s, c: {}
        mock_trader.return_value = lambda s, c: {}
        mock_risk.return_value = lambda s, c: {}
        mock_pm.return_value = lambda s, c: {}
        mock_tool_node.return_value = lambda s, c: {}
        mock_toolkit.get_all_tools.return_value = []

        graph = create_trading_graph(max_debate_rounds=2)

        # Test debate router is compiled into graph
        assert graph is not None


class TestSyncCheckRouter:
    """Test sync_check_router for parallel debate fan-out."""

    def test_sync_check_returns_end_when_incomplete(self):
        """Test router returns __end__ when not all analysts complete."""
        from src.graph import sync_check_router

        state = {
            "market_report": "done",
            "sentiment_report": "",  # Not complete
            "news_report": "done",
            "pre_screening_result": "PASS",
            "financial_validation_complete": True,
        }
        config = {}

        result = sync_check_router(state, config, auditor_required=False)
        assert result == "__end__"

    def test_sync_check_proceeds_when_required_branch_failed_but_completed(self):
        """Router should wait for completion, not success, at the sync barrier."""
        from src.graph import sync_check_router

        state = {
            "market_report": "Error: DNS failure",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
            "financial_validation_complete": True,
            "artifact_statuses": {
                "market_report": {
                    "complete": True,
                    "ok": False,
                    "error_kind": "dns_resolution",
                    "provider": "google",
                },
                "sentiment_report": {"ok": True, "content": "done"},
                "news_report": {"ok": True, "content": "done"},
                "value_trap_report": {"ok": True, "content": "done"},
            },
        }

        result = sync_check_router(state, {}, auditor_required=False)
        assert isinstance(result, list)
        assert "Bull Researcher R1" in result

    def test_sync_check_returns_pm_fast_fail_on_reject(self):
        """Test router returns PM Fast-Fail on REJECT (separate node to avoid edge conflicts)."""
        from src.graph import sync_check_router

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "REJECT",
            "financial_validation_complete": True,
        }
        config = {}

        result = sync_check_router(state, config, auditor_required=False)
        assert result == "PM Fast-Fail"

    def test_sync_check_waits_for_validator_even_when_parallel_gate_rejects(self):
        from src.graph import sync_check_router

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "REJECT",
            "financial_validation_complete": False,
        }

        assert sync_check_router(state, {}, auditor_required=False) == "__end__"

    def test_sync_check_routes_only_from_committed_pre_screening_state(self):
        from src.graph import sync_check_router
        from src.liquidity_assessment import LiquidityAssessment

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
            "financial_validation_complete": True,
            "liquidity_assessment": LiquidityAssessment(
                status="FAIL_INSUFFICIENT_LIQUIDITY",
                average_daily_turnover_usd=86_436,
            ).to_dict(),
        }

        result = sync_check_router(state, {}, auditor_required=False)

        assert result == ["Bull Researcher R1", "Bear Researcher R1"]

    def test_sync_check_does_not_treat_liquidity_error_as_issuer_hard_fail(self):
        from src.graph import sync_check_router
        from src.liquidity_assessment import LiquidityAssessment

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
            "financial_validation_complete": True,
            "liquidity_assessment": LiquidityAssessment(
                status="ERROR",
                reason="DNS_FAILURE",
            ).to_dict(),
        }

        result = sync_check_router(state, {}, auditor_required=False)

        assert isinstance(result, list)
        assert result == ["Bull Researcher R1", "Bear Researcher R1"]

    def test_sync_check_returns_list_for_parallel_r1(self):
        """Test router returns list for parallel Bull/Bear R1 on PASS."""
        from src.graph import sync_check_router

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
            "financial_validation_complete": True,
        }
        config = {}

        result = sync_check_router(state, config, auditor_required=False)
        assert isinstance(result, list)
        assert "Bull Researcher R1" in result
        assert "Bear Researcher R1" in result
        assert len(result) == 2


class TestAuditorIntegration:
    """Test auditor node integration with graph routing."""

    def test_fan_out_includes_auditor_when_enabled(self):
        """Test fan_out_to_analysts includes Auditor when enabled."""
        from src.graph import fan_out_to_analysts

        result = fan_out_to_analysts({}, {}, include_auditor=True)
        assert "Auditor" in result
        assert "Value Trap Detector" in result
        assert len(result) == 8  # 7 analysts + Auditor

    def test_fan_out_excludes_auditor_when_disabled(self):
        """Test fan_out_to_analysts excludes Auditor when disabled."""
        from src.graph import fan_out_to_analysts

        result = fan_out_to_analysts({}, {}, include_auditor=False)
        assert "Auditor" not in result
        assert "Value Trap Detector" in result
        assert len(result) == 7

    def test_sync_check_waits_for_auditor_when_enabled(self):
        """Test sync_check_router waits for auditor_report when enabled."""
        from src.graph import sync_check_router

        # All reports present except auditor_report
        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
            "financial_validation_complete": True,
            "auditor_report": "",  # Empty = not done
        }

        result = sync_check_router(state, {}, auditor_required=True)
        assert result == "__end__"  # Should wait

    def test_sync_check_proceeds_when_auditor_failed_but_completed(self):
        """A failed enabled auditor branch should still satisfy sync completion."""
        from src.graph import sync_check_router

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
            "financial_validation_complete": True,
            "auditor_report": "",
            "artifact_statuses": {
                "auditor_report": {
                    "complete": True,
                    "ok": False,
                    "error_kind": "timeout",
                    "provider": "openai",
                }
            },
        }

        result = sync_check_router(state, {}, auditor_required=True)
        assert isinstance(result, list)
        assert "Bull Researcher R1" in result

    def test_sync_check_proceeds_when_auditor_complete(self):
        """Test sync_check_router proceeds when auditor_report complete."""
        from src.graph import sync_check_router

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
            "financial_validation_complete": True,
            "auditor_report": "Forensic audit complete",
        }

        result = sync_check_router(state, {}, auditor_required=True)
        assert isinstance(result, list)
        assert "Bull Researcher R1" in result


class TestAuditorLLMConfiguration:
    """Tests for create_auditor_llm() parameter safety.

    Validates that the auditor LLM never sets temperature (which various
    OpenAI model families reject), and handles enable/disable correctly.
    Tests are model-agnostic — the user configures models via .env.
    """

    @patch("src.llms.config")
    def test_auditor_llm_never_sets_temperature(self, mock_config):
        """Auditor LLM should never set temperature (any model can reject it)."""
        from src.llms import create_auditor_llm

        mock_config.enable_consultant = True
        mock_config.get_openai_api_key.return_value = "fake-key"
        mock_config.auditor_model = "any-model-name"
        mock_config.consultant_model = "fallback-model"

        llm = create_auditor_llm()
        assert llm is not None
        # LangChain ChatOpenAI defaults temperature to 0.7 unless explicitly set.
        # We want the SDK default (1.0 for reasoning models, 0.7 for others)
        # — NOT 0.0 which breaks many model families.
        assert llm.temperature != 0.0

    @patch("src.llms.config")
    def test_auditor_llm_disabled_without_consultant(self, mock_config):
        """Should return None when ENABLE_CONSULTANT is false."""
        from src.llms import create_auditor_llm

        mock_config.enable_consultant = False

        llm = create_auditor_llm()
        assert llm is None

    @patch("src.llms.config")
    def test_auditor_llm_disabled_without_api_key(self, mock_config):
        """Should return None when OPENAI_API_KEY is missing."""
        from src.llms import create_auditor_llm

        mock_config.enable_consultant = True
        mock_config.get_openai_api_key.return_value = None

        llm = create_auditor_llm()
        assert llm is None

    @patch("src.llms.config")
    def test_auditor_llm_falls_back_to_consultant_model(self, mock_config):
        """Should use consultant_model when auditor_model is not set."""
        from src.llms import create_auditor_llm

        mock_config.enable_consultant = True
        mock_config.get_openai_api_key.return_value = "fake-key"
        mock_config.auditor_model = None
        mock_config.consultant_model = "some-consultant-model"

        llm = create_auditor_llm()
        assert llm is not None

    @patch("src.llms.config")
    def test_auditor_llm_prefers_auditor_model(self, mock_config):
        """Should use auditor_model over consultant_model when set."""
        from src.llms import create_auditor_llm

        mock_config.enable_consultant = True
        mock_config.get_openai_api_key.return_value = "fake-key"
        mock_config.auditor_model = "specific-auditor-model"
        mock_config.consultant_model = "some-consultant-model"

        llm = create_auditor_llm()
        assert llm is not None
        # The model set should be the auditor-specific one
        assert "auditor" in llm.model_name

    @patch("src.llms.config")
    def test_consultant_llm_never_sets_temperature(self, mock_config):
        """Consultant LLM should never set temperature (model-agnostic)."""
        from src.llms import get_consultant_llm

        mock_config.enable_consultant = True
        mock_config.get_openai_api_key.return_value = "fake-key"
        mock_config.consultant_model = "any-model-name"
        mock_config.consultant_quick_model = "any-quick-model"

        llm = get_consultant_llm()
        assert llm is not None
        assert llm.temperature != 0.0

    def test_auditor_quick_mode_gpt5_mini_uses_low_effort(self):
        """Quick-mode gpt-5-mini auditor must use 'low' (mini rejects 'minimal')."""
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("langchain-openai not installed (optional dependency)")

        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        from src.llms import create_auditor_llm

        with _patch("langchain_openai.ChatOpenAI") as mock_chatgpt:
            mock_chatgpt.return_value = MagicMock()
            with _patch("src.llms.config") as cfg:
                cfg.enable_consultant = True
                cfg.get_openai_api_key.return_value = "k"
                cfg.auditor_model = None
                cfg.auditor_quick_model = "gpt-5.4-mini"
                cfg.consultant_model = "gpt-5.4"
                create_auditor_llm(quick_mode=True)
                kw = mock_chatgpt.call_args[1]
                assert kw["model"] == "gpt-5.4-mini"
                assert kw["reasoning_effort"] == "low"

    @pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])
    def test_auditor_quick_mode_gpt56_models_use_low_effort(self, model):
        """Quick-mode GPT-5.6 variants must use their documented low effort."""
        try:
            import langchain_openai  # noqa: F401
        except ImportError:
            import pytest

            pytest.skip("langchain-openai not installed (optional dependency)")

        from unittest.mock import MagicMock
        from unittest.mock import patch as _patch

        from src.llms import create_auditor_llm

        with _patch("langchain_openai.ChatOpenAI") as mock_chatgpt:
            mock_chatgpt.return_value = MagicMock()
            with _patch("src.llms.config") as cfg:
                cfg.enable_consultant = True
                cfg.get_openai_api_key.return_value = "k"
                cfg.auditor_model = None
                cfg.auditor_quick_model = model
                cfg.consultant_model = "gpt-5.4"
                create_auditor_llm(quick_mode=True)
                kw = mock_chatgpt.call_args[1]
                assert kw["model"] == model
                assert kw["reasoning_effort"] == "low"


class TestAuditorLLMReasoning:
    """Keep the full-mode Auditor's reviewed reasoning construction unchanged."""

    def test_auditor_gpt5_uses_medium_reasoning_effort(self, monkeypatch):
        import src.llms as llms

        stub_module = ModuleType("langchain_openai")
        captured = {}

        class StubChatOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.model_name = kwargs["model"]

        stub_module.ChatOpenAI = StubChatOpenAI

        monkeypatch.setitem(sys.modules, "langchain_openai", stub_module)
        monkeypatch.setattr(llms.config, "enable_consultant", True)
        monkeypatch.setattr(
            type(llms.config), "get_openai_api_key", lambda self: "fake-key"
        )
        monkeypatch.setattr(llms.config, "auditor_model", "gpt-5-mini")
        monkeypatch.setattr(llms.config, "consultant_model", "gpt-5")

        llm = llms.create_auditor_llm()

        assert llm is not None
        assert captured["reasoning_effort"] == "medium"


class TestQuickModeGraphContracts:
    """Lock down consultant/auditor behavior in quick mode."""

    def test_build_graph_components_keeps_consultant_but_disables_auditor_in_quick_mode(
        self, monkeypatch
    ):
        from src.graph.components import build_graph_components

        components = _stub_graph_component_dependencies(monkeypatch)
        quick_consultant = Mock(name="quick-consultant")
        auditor_factory = Mock(return_value=Mock(name="auditor"))

        consultant_calls = []

        def fake_get_consultant_llm(**kwargs):
            consultant_calls.append(kwargs)
            return quick_consultant

        monkeypatch.setattr(components, "get_consultant_llm", fake_get_consultant_llm)
        monkeypatch.setattr(components, "create_auditor_llm", auditor_factory)

        graph_components = build_graph_components(
            max_debate_rounds=1,
            enable_memory=False,
            ticker="TEST",
            cleanup_previous=False,
            quick_mode=True,
            strict_mode=False,
            chart_format="png",
            transparent_charts=False,
            image_dir=None,
            skip_charts=True,
        )

        assert graph_components.consultant_enabled is True
        assert graph_components.auditor_enabled is False
        assert "Consultant" in graph_components.nodes
        assert "Auditor" not in graph_components.nodes
        assert "auditor_tools" not in graph_components.tool_nodes
        auditor_factory.assert_not_called()
        assert consultant_calls
        assert consultant_calls[0]["quick_mode"] is True

    def test_build_graph_components_quick_mode_disables_both_when_openai_path_unavailable(
        self, monkeypatch
    ):
        from src.graph.components import build_graph_components

        components = _stub_graph_component_dependencies(monkeypatch)
        monkeypatch.setattr(components, "get_consultant_llm", lambda **kwargs: None)

        graph_components = build_graph_components(
            max_debate_rounds=1,
            enable_memory=False,
            ticker="TEST",
            cleanup_previous=False,
            quick_mode=True,
            strict_mode=False,
            chart_format="png",
            transparent_charts=False,
            image_dir=None,
            skip_charts=True,
        )

        assert graph_components.consultant_enabled is False
        assert graph_components.auditor_enabled is False
        assert "Consultant" not in graph_components.nodes
        assert "Auditor" not in graph_components.nodes

    def test_full_mode_raises_when_auditor_routing_and_creation_disagree(
        self, monkeypatch
    ):
        from src.graph.components import build_graph_components

        components = _stub_graph_component_dependencies(monkeypatch)
        monkeypatch.setattr(components, "get_consultant_llm", lambda **kwargs: Mock())
        monkeypatch.setattr(
            components,
            "_is_auditor_enabled",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(components, "create_auditor_llm", lambda **kwargs: None)

        with pytest.raises(RuntimeError, match="Auditor routing was enabled"):
            build_graph_components(
                max_debate_rounds=1,
                enable_memory=False,
                ticker="TEST",
                cleanup_previous=False,
                quick_mode=False,
                strict_mode=False,
                chart_format="png",
                transparent_charts=False,
                image_dir=None,
                skip_charts=True,
            )

    def test_build_graph_components_uses_full_mode_consultant_when_not_quick(
        self, monkeypatch
    ):
        from src.graph.components import build_graph_components

        components = _stub_graph_component_dependencies(monkeypatch)
        consultant_calls = []

        def fake_get_consultant_llm(**kwargs):
            consultant_calls.append(kwargs)
            return Mock(name="full-consultant")

        monkeypatch.setattr(components, "get_consultant_llm", fake_get_consultant_llm)
        monkeypatch.setattr(
            components,
            "_is_auditor_enabled",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(components, "create_auditor_llm", lambda **kwargs: Mock())

        graph_components = build_graph_components(
            max_debate_rounds=2,
            enable_memory=False,
            ticker="TEST",
            cleanup_previous=False,
            quick_mode=False,
            strict_mode=False,
            chart_format="png",
            transparent_charts=False,
            image_dir=None,
            skip_charts=True,
        )

        assert graph_components.consultant_enabled is True
        assert graph_components.auditor_enabled is True
        assert consultant_calls
        assert consultant_calls[0]["quick_mode"] is False

    def test_build_graph_components_adds_apac_specialist_only_in_full_mode(
        self, monkeypatch
    ):
        from src.graph.components import build_graph_components

        components = _stub_graph_component_dependencies(monkeypatch)
        calls = []

        def fake_create_apac_llm(**kwargs):
            calls.append(kwargs)
            return None if kwargs.get("quick_mode") else Mock(name="apac")

        monkeypatch.setattr(components, "get_consultant_llm", lambda **kwargs: None)
        monkeypatch.setattr(
            components,
            "_is_auditor_enabled",
            lambda *_args, **_kwargs: False,
        )
        monkeypatch.setattr(
            components, "create_apac_specialist_llm", fake_create_apac_llm
        )

        full = build_graph_components(
            max_debate_rounds=2,
            enable_memory=False,
            ticker="7203.T",
            cleanup_previous=False,
            quick_mode=False,
            strict_mode=False,
            chart_format="png",
            transparent_charts=False,
            image_dir=None,
            skip_charts=False,
        )
        quick = build_graph_components(
            max_debate_rounds=1,
            enable_memory=False,
            ticker="7203.T",
            cleanup_previous=False,
            quick_mode=True,
            strict_mode=False,
            chart_format="png",
            transparent_charts=False,
            image_dir=None,
            skip_charts=True,
        )

        assert full.apac_specialist_enabled is True
        assert "APAC Regional Specialist" in full.nodes
        assert quick.apac_specialist_enabled is False
        assert "APAC Regional Specialist" not in quick.nodes
        assert [call["quick_mode"] for call in calls] == [False, False]
        assert calls[0].get("thinking_enabled", True) is True
        assert calls[1]["thinking_enabled"] is False

    def _count_thinking_bumps(self, monkeypatch, *, quick_mode):
        """Build components with a recording quick-LLM stub; return bump count.

        Every quick-thinking construction is captured; the count of calls
        passing ``thinking_level_bump=True`` is the wiring assertion target.
        """
        from src.graph.components import build_graph_components

        components = _stub_graph_component_dependencies(monkeypatch)
        monkeypatch.setattr(components, "get_consultant_llm", lambda **kwargs: None)
        monkeypatch.setattr(
            components,
            "_is_auditor_enabled",
            lambda *_args, **_kwargs: False,
        )

        bump_flags = []

        def recording_quick_llm(**kwargs):
            bump_flags.append(bool(kwargs.get("thinking_level_bump", False)))
            return Mock()

        monkeypatch.setattr(
            components, "create_quick_thinking_llm", recording_quick_llm
        )

        build_graph_components(
            max_debate_rounds=1 if quick_mode else 2,
            enable_memory=False,
            ticker="TEST",
            cleanup_previous=False,
            quick_mode=quick_mode,
            strict_mode=False,
            chart_format="png",
            transparent_charts=False,
            image_dir=None,
            skip_charts=True,
        )
        return sum(bump_flags)

    def test_only_value_trap_detector_bumps_thinking_in_full_mode(self, monkeypatch):
        """Full mode: exactly one quick-tier agent (Value Trap Detector) is bumped."""
        assert self._count_thinking_bumps(monkeypatch, quick_mode=False) == 1

    def test_no_thinking_bump_in_quick_mode(self, monkeypatch):
        """Quick mode: no quick-tier agent receives the bump (cheap screening)."""
        assert self._count_thinking_bumps(monkeypatch, quick_mode=True) == 0

    def test_non_google_base_keeps_retry_and_value_trap_adjustment(self, monkeypatch):
        """Provider swaps must not silently disable the full-mode quality paths."""
        from src.config import Settings
        from src.graph.components import build_graph_components
        from src.llm_runtime.bindings import resolve_binding_plan
        from src.llm_runtime.seats import SeatId

        components = _stub_graph_component_dependencies(monkeypatch)
        analyst_kwargs = []

        def analyst_node(*args, **kwargs):
            analyst_kwargs.append(kwargs)
            return lambda state, config: {}

        monkeypatch.setattr(components, "create_analyst_node", analyst_node)
        plan = resolve_binding_plan(
            Settings(
                _env_file=None,
                llm_base_provider="openai",
                llm_review_provider="google",
                llm_regional_provider="deepseek",
                google_api_key="g",
                openai_api_key="o",
                claude_api_key="a",
                deepseek_api_key="d",
                llm_consultant_mode="off",
                llm_auditor_mode="off",
                llm_editor_mode="off",
                llm_apac_mode="off",
            )
        )
        requests = []

        class RecordingFactory:
            def build(self, request):
                requests.append(request)
                return Mock(name=request.seat.seat_id.value)

        build_graph_components(
            max_debate_rounds=2,
            enable_memory=False,
            ticker="TEST",
            cleanup_previous=False,
            quick_mode=False,
            strict_mode=False,
            chart_format="png",
            transparent_charts=False,
            image_dir=None,
            skip_charts=True,
            binding_plan=plan,
            model_factory=RecordingFactory(),
        )

        by_seat = {request.seat.seat_id: request for request in requests}
        assert by_seat[SeatId.ANALYST_RETRY].binding.provider == "openai"
        assert by_seat[SeatId.VALUE_TRAP].reasoning_value == "medium"
        assert analyst_kwargs
        assert all(kwargs["allow_retry"] is True for kwargs in analyst_kwargs)
        assert all(kwargs["retry_llm"] is not None for kwargs in analyst_kwargs)

    def test_quick_mode_arms_recovery_only_for_gate_critical_seats(self, monkeypatch):
        from src.config import Settings
        from src.graph.components import build_graph_components
        from src.llm_runtime.bindings import resolve_binding_plan
        from src.llm_runtime.seats import SeatId

        components = _stub_graph_component_dependencies(monkeypatch)
        analyst_calls = []
        pm_calls = []

        def analyst_node(_llm, agent_key, _tools, _field, **kwargs):
            analyst_calls.append((agent_key, kwargs))
            return lambda state, config: {}

        def pm_node(*args, **kwargs):
            pm_calls.append(kwargs)
            return lambda state, config: {}

        monkeypatch.setattr(components, "create_analyst_node", analyst_node)
        monkeypatch.setattr(components, "create_portfolio_manager_node", pm_node)
        plan = resolve_binding_plan(
            Settings(
                _env_file=None,
                llm_base_provider="openai",
                llm_review_provider="google",
                llm_regional_provider="deepseek",
                google_api_key="g",
                openai_api_key="o",
                claude_api_key="a",
                deepseek_api_key="d",
                llm_consultant_mode="off",
                llm_auditor_mode="off",
                llm_editor_mode="off",
                llm_apac_mode="off",
            )
        )
        requests = []

        class RecordingFactory:
            def build(self, request):
                requests.append(request)
                return Mock(
                    name=f"{request.seat.seat_id.value}-{request.output_tokens}"
                )

        build_graph_components(
            max_debate_rounds=1,
            enable_memory=False,
            ticker="TEST",
            cleanup_previous=False,
            quick_mode=True,
            strict_mode=False,
            chart_format="png",
            transparent_charts=False,
            image_dir=None,
            skip_charts=True,
            binding_plan=plan,
            model_factory=RecordingFactory(),
        )

        retry_requests = [
            request
            for request in requests
            if request.seat.seat_id is SeatId.ANALYST_RETRY
        ]
        assert {request.output_tokens for request in retry_requests} == {10923, 16384}
        by_agent = dict(analyst_calls)
        assert by_agent["fundamentals_analyst"]["allow_retry"] is True
        assert by_agent["fundamentals_analyst"]["retry_llm"] is not None
        for agent_key, kwargs in analyst_calls:
            if agent_key != "fundamentals_analyst":
                assert kwargs["allow_retry"] is False
                assert kwargs["retry_llm"] is None
        assert len(pm_calls) == 1
        assert pm_calls[0]["recovery_llm"] is not None

    def test_legacy_pre_gemini_3_floor_keeps_retry_disabled(self, monkeypatch):
        """The compatibility bridge must preserve the old retry eligibility gate."""
        from types import SimpleNamespace

        from src.config import Settings
        from src.graph.components import build_graph_components
        from src.llm_runtime.bindings import resolve_binding_plan

        components = _stub_graph_component_dependencies(monkeypatch)
        analyst_kwargs = []

        def analyst_node(*args, **kwargs):
            analyst_kwargs.append(kwargs)
            return lambda state, config: {}

        monkeypatch.setattr(components, "create_analyst_node", analyst_node)
        monkeypatch.setattr(
            components,
            "get_runtime_config",
            lambda settings: SimpleNamespace(
                quick_think_llm="gemini-2.5-flash",
                deep_think_llm="gemini-2.5-pro",
            ),
        )
        plan = resolve_binding_plan(
            Settings(
                _env_file=None,
                google_api_key="g",
                quick_think_llm="gemini-2.5-flash",
                deep_think_llm="gemini-2.5-pro",
            )
        )

        build_graph_components(
            max_debate_rounds=2,
            enable_memory=False,
            ticker="TEST",
            cleanup_previous=False,
            quick_mode=False,
            strict_mode=False,
            chart_format="png",
            transparent_charts=False,
            image_dir=None,
            skip_charts=True,
            binding_plan=plan,
        )

        assert analyst_kwargs
        assert all(kwargs["allow_retry"] is False for kwargs in analyst_kwargs)
        assert all(kwargs["retry_llm"] is None for kwargs in analyst_kwargs)


class TestDebateReasoningHandoffWiring:
    """The opt-in policy must alter only the three intended debate seats."""

    def test_inactive_policy_leaves_every_debate_seat_untouched(self, monkeypatch):
        """Inert by default, pinned rather than re-derived by a reader.

        With the policy off no seat may request reasoning output and no
        researcher may receive the recovery clients — the fallback exists only
        to repair a response the capsule contract degraded, so carrying one
        here would arm a retry path that has nothing to recover.
        """
        from src.config import Settings
        from src.graph.components import build_graph_components
        from src.llm_runtime.bindings import resolve_binding_plan

        components = _stub_graph_component_dependencies(monkeypatch)
        researcher_calls: list[dict] = []

        def researcher_node(*args, **kwargs):
            researcher_calls.append(kwargs)
            return lambda state, runtime: {}

        monkeypatch.setattr(components, "create_researcher_node", researcher_node)

        settings = Settings(
            _env_file=None,
            google_api_key="g",
            quick_think_llm="gemini-2.5-flash",
            deep_think_llm="gemini-2.5-pro",
        )
        requests = []

        class RecordingFactory:
            def build(self, request):
                requests.append(request)
                return Mock(name=request.seat.seat_id.value)

        build_graph_components(
            max_debate_rounds=2,
            enable_memory=False,
            ticker="TEST",
            cleanup_previous=False,
            quick_mode=False,
            strict_mode=False,
            chart_format="png",
            transparent_charts=False,
            image_dir=None,
            skip_charts=True,
            binding_plan=resolve_binding_plan(settings),
            model_factory=RecordingFactory(),
        )

        assert len(researcher_calls) == 4
        assert not any(r.include_reasoning_output for r in requests)
        for kwargs in researcher_calls:
            assert "fallback_llm" not in kwargs
            assert "structured_repair_llm" not in kwargs
            assert "handoff_policy" not in kwargs

    def test_active_policy_builds_two_r1_reasoning_clients_and_expands_budgets(
        self, monkeypatch
    ):
        from src.config import Settings, config
        from src.graph.components import build_graph_components
        from src.llm_budgets import get_agent_output_budget
        from src.llm_runtime.bindings import resolve_binding_plan
        from src.llm_runtime.seats import SeatId
        from src.runtime_config import RuntimeConfig, use_runtime_config

        components = _stub_graph_component_dependencies(monkeypatch)
        researcher_calls: list[tuple[tuple, dict]] = []
        manager_calls: list[tuple[tuple, dict]] = []

        def researcher_node(*args, **kwargs):
            researcher_calls.append((args, kwargs))
            return lambda state, runtime: {}

        def manager_node(*args, **kwargs):
            manager_calls.append((args, kwargs))
            return lambda state, runtime: {}

        monkeypatch.setattr(components, "create_researcher_node", researcher_node)
        monkeypatch.setattr(components, "create_research_manager_node", manager_node)

        settings = Settings(
            _env_file=None,
            llm_base_provider="openai",
            llm_review_provider="google",
            llm_regional_provider="deepseek",
            google_api_key="g",
            openai_api_key="o",
            claude_api_key="a",
            deepseek_api_key="d",
            llm_consultant_mode="off",
            llm_auditor_mode="off",
            llm_editor_mode="off",
            llm_apac_mode="off",
        )
        plan = resolve_binding_plan(settings)
        requests = []

        class RecordingFactory:
            def build(self, request):
                requests.append(request)
                return Mock(name=f"{request.seat.seat_id.value}-{len(requests)}")

        runtime_config = RuntimeConfig.from_config(settings).with_overrides(
            debate_reasoning_handoffs=True
        )
        with use_runtime_config(runtime_config):
            graph_components = build_graph_components(
                max_debate_rounds=2,
                enable_memory=False,
                ticker="TEST",
                cleanup_previous=False,
                quick_mode=False,
                strict_mode=False,
                chart_format="png",
                transparent_charts=False,
                image_dir=None,
                skip_charts=True,
                binding_plan=plan,
                model_factory=RecordingFactory(),
            )

        bull_requests = [r for r in requests if r.seat.seat_id is SeatId.BULL]
        bear_requests = [r for r in requests if r.seat.seat_id is SeatId.BEAR]
        manager_requests = [
            r for r in requests if r.seat.seat_id is SeatId.RESEARCH_MANAGER
        ]
        assert [r.include_reasoning_output for r in bull_requests] == [
            False,
            True,
            False,
        ]
        assert [r.include_reasoning_output for r in bear_requests] == [
            False,
            True,
            False,
        ]
        assert [r.include_reasoning_output for r in manager_requests] == [False]

        base_tokens = config.llm_base_output_tokens
        full_bull_budget = (
            get_agent_output_budget("Bull Researcher", base_tokens) + 1_024
        )
        full_bear_budget = (
            get_agent_output_budget("Bear Researcher", base_tokens) + 1_024
        )
        assert [r.output_tokens for r in bull_requests] == [
            full_bull_budget,
            full_bull_budget,
            1_024,
        ]
        assert [r.output_tokens for r in bear_requests] == [
            full_bear_budget,
            full_bear_budget,
            1_024,
        ]
        for repair_request in (bull_requests[-1], bear_requests[-1]):
            assert repair_request.quick_mode is True
            assert repair_request.include_reasoning_output is False
            assert repair_request.callbacks[0].output_token_cap == 1_024
        assert manager_requests[0].output_tokens == (
            get_agent_output_budget("Research Manager", base_tokens) + 1_024
        )

        by_role_round = {
            (args[2], kwargs["round_num"]): (args, kwargs)
            for args, kwargs in researcher_calls
        }
        for role in ("bull_researcher", "bear_researcher"):
            r1_args, r1_kwargs = by_role_round[(role, 1)]
            r2_args, r2_kwargs = by_role_round[(role, 2)]
            assert (
                r1_kwargs["handoff_policy"] is graph_components.debate_reasoning_policy
            )
            assert (
                r2_kwargs["handoff_policy"] is graph_components.debate_reasoning_policy
            )
            assert r1_kwargs["fallback_llm"] is r2_args[0]
            assert r1_kwargs["structured_repair_llm"] not in {
                r1_args[0],
                r2_args[0],
            }
            assert r1_args[0] is not r2_args[0]
            assert "fallback_llm" not in r2_kwargs
            assert "structured_repair_llm" not in r2_kwargs
        assert manager_calls[0][1]["handoff_policy"] is (
            graph_components.debate_reasoning_policy
        )


class TestTradingContext:
    """Test TradingContext dataclass."""

    def test_trading_context_creation(self):
        """Test TradingContext creation."""
        from src.graph import TradingContext

        context = TradingContext(
            ticker="AAPL", trade_date="2024-01-01", quick_mode=False, enable_memory=True
        )

        assert context.ticker == "AAPL"
        assert context.trade_date == "2024-01-01"
        assert context.max_debate_rounds == 2

    def test_trading_context_quick_mode(self):
        """Test TradingContext in quick mode."""
        from src.graph import TradingContext

        context = TradingContext(
            ticker="AAPL",
            trade_date="2024-01-01",
            quick_mode=True,
            max_debate_rounds=1,  # Quick mode uses 1 round
        )

        assert context.quick_mode is True
        assert context.max_debate_rounds == 1


class TestGraphCompilation:
    """Test graph compilation."""

    @patch("src.graph.components.create_agent_tool_node")
    @patch("src.graph.components.create_quick_thinking_llm")
    @patch("src.graph.components.create_deep_thinking_llm")
    @patch("src.graph.components.toolkit")
    def test_create_trading_graph(
        self, mock_toolkit, mock_deep_llm_func, mock_quick_llm_func, mock_tool_node
    ):
        """Test trading graph creation."""
        from src.graph import create_trading_graph

        # Mock the LLM creation functions to return mock LLMs
        mock_quick_llm = MagicMock()
        mock_deep_llm = MagicMock()
        mock_quick_llm_func.return_value = mock_quick_llm
        mock_deep_llm_func.return_value = mock_deep_llm

        mock_toolkit.get_technical_tools.return_value = []
        mock_toolkit.get_sentiment_tools.return_value = []
        mock_toolkit.get_news_tools.return_value = []
        mock_toolkit.get_fundamental_tools.return_value = []
        mock_toolkit.get_all_tools.return_value = []

        # Mock create_agent_tool_node to return a dummy function
        mock_tool_node.return_value = lambda s, c: {}

        graph = create_trading_graph(max_debate_rounds=2, enable_memory=True)

        assert graph is not None
        # Graph should be compiled and ready to invoke


class TestStrictGraphWiring:
    """Smoke tests: strict_mode threads correctly through graph construction."""

    @patch("src.graph.components.create_financial_health_validator_node")
    @patch("src.graph.components.create_portfolio_manager_node")
    @patch("src.graph.components.create_research_manager_node")
    @patch("src.graph.components.create_analyst_node")
    @patch("src.graph.components.create_researcher_node")
    @patch("src.graph.components.create_trader_node")
    @patch("src.graph.components.create_risk_debater_node")
    @patch("src.graph.components.create_agent_tool_node")
    @patch("src.graph.components.toolkit")
    def test_strict_mode_reaches_validator_factory(
        self,
        mock_toolkit,
        mock_tool_node,
        mock_risk,
        mock_trader,
        mock_researcher,
        mock_analyst,
        mock_rm,
        mock_pm,
        mock_validator,
    ):
        """strict_mode=True is forwarded to create_financial_health_validator_node."""
        from src.graph import create_trading_graph

        for m in (
            mock_analyst,
            mock_researcher,
            mock_rm,
            mock_trader,
            mock_risk,
            mock_tool_node,
        ):
            m.return_value = lambda s, c: {}
        mock_pm.return_value = lambda s, c: {}
        mock_validator.return_value = lambda s, c: {}
        mock_toolkit.get_all_tools.return_value = []
        mock_toolkit.get_market_tools.return_value = []
        mock_toolkit.get_sentiment_tools.return_value = []
        mock_toolkit.get_news_tools.return_value = []
        mock_toolkit.get_junior_fundamental_tools.return_value = []
        mock_toolkit.get_senior_fundamental_tools.return_value = []
        mock_toolkit.get_foreign_language_tools.return_value = []
        mock_toolkit.get_legal_tools.return_value = []
        mock_toolkit.get_value_trap_tools.return_value = []

        create_trading_graph(strict_mode=True, enable_memory=False)
        mock_validator.assert_called_once_with(strict_mode=True)

    @patch("src.graph.components.create_financial_health_validator_node")
    @patch("src.graph.components.create_portfolio_manager_node")
    @patch("src.graph.components.create_research_manager_node")
    @patch("src.graph.components.create_analyst_node")
    @patch("src.graph.components.create_researcher_node")
    @patch("src.graph.components.create_trader_node")
    @patch("src.graph.components.create_risk_debater_node")
    @patch("src.graph.components.create_agent_tool_node")
    @patch("src.graph.components.toolkit")
    def test_strict_mode_reaches_pm_factory(
        self,
        mock_toolkit,
        mock_tool_node,
        mock_risk,
        mock_trader,
        mock_researcher,
        mock_analyst,
        mock_rm,
        mock_pm,
        mock_validator,
    ):
        """strict_mode=True reaches the sole discretionary PM factory."""
        from src.graph import create_trading_graph

        for m in (
            mock_analyst,
            mock_researcher,
            mock_rm,
            mock_trader,
            mock_risk,
            mock_tool_node,
            mock_validator,
        ):
            m.return_value = lambda s, c: {}
        mock_pm.return_value = lambda s, c: {}
        mock_toolkit.get_all_tools.return_value = []
        mock_toolkit.get_market_tools.return_value = []
        mock_toolkit.get_sentiment_tools.return_value = []
        mock_toolkit.get_news_tools.return_value = []
        mock_toolkit.get_junior_fundamental_tools.return_value = []
        mock_toolkit.get_senior_fundamental_tools.return_value = []
        mock_toolkit.get_foreign_language_tools.return_value = []
        mock_toolkit.get_legal_tools.return_value = []
        mock_toolkit.get_value_trap_tools.return_value = []

        create_trading_graph(strict_mode=True, enable_memory=False)
        # The rejection path is deterministic and does not construct another PM.
        calls = mock_pm.call_args_list
        assert len(calls) == 1
        assert calls[0].kwargs.get("strict_mode") is True

    @patch("src.graph.components.create_financial_health_validator_node")
    @patch("src.graph.components.create_portfolio_manager_node")
    @patch("src.graph.components.create_research_manager_node")
    @patch("src.graph.components.create_analyst_node")
    @patch("src.graph.components.create_researcher_node")
    @patch("src.graph.components.create_trader_node")
    @patch("src.graph.components.create_risk_debater_node")
    @patch("src.graph.components.create_agent_tool_node")
    @patch("src.graph.components.toolkit")
    def test_strict_mode_reaches_rm_factory(
        self,
        mock_toolkit,
        mock_tool_node,
        mock_risk,
        mock_trader,
        mock_researcher,
        mock_analyst,
        mock_rm,
        mock_pm,
        mock_validator,
    ):
        """strict_mode=True is forwarded to create_research_manager_node."""
        from src.graph import create_trading_graph

        for m in (
            mock_analyst,
            mock_researcher,
            mock_trader,
            mock_risk,
            mock_tool_node,
            mock_validator,
            mock_pm,
        ):
            m.return_value = lambda s, c: {}
        mock_rm.return_value = lambda s, c: {}
        mock_toolkit.get_all_tools.return_value = []
        mock_toolkit.get_market_tools.return_value = []
        mock_toolkit.get_sentiment_tools.return_value = []
        mock_toolkit.get_news_tools.return_value = []
        mock_toolkit.get_junior_fundamental_tools.return_value = []
        mock_toolkit.get_senior_fundamental_tools.return_value = []
        mock_toolkit.get_foreign_language_tools.return_value = []
        mock_toolkit.get_legal_tools.return_value = []
        mock_toolkit.get_value_trap_tools.return_value = []

        create_trading_graph(strict_mode=True, enable_memory=False)
        mock_rm.assert_called_once()
        call_kwargs = mock_rm.call_args.kwargs
        assert call_kwargs.get("strict_mode") is True


class TestPostResearchSync:
    """Regression coverage for the post-research fan-in before Trader."""

    @pytest.mark.parametrize("liquidity_reject", [False, True])
    @pytest.mark.asyncio
    async def test_trader_tail_runs_once_after_valuation_and_consultant_complete(
        self, monkeypatch, liquidity_reject
    ):
        import src.graph.components as components
        from src.runtime_diagnostics import success_artifact

        calls: list[str] = []

        def artifact_node(field: str, value: str):
            async def _node(state, config):
                calls.append(field)
                return success_artifact(field, value, provider="test")

            return _node

        def analyst_node(_llm, _agent_key, _tools, output_field, **_kwargs):
            node = artifact_node(output_field, f"{output_field} done")
            if output_field != "market_report" or not liquidity_reject:
                return node

            async def _illiquid_market(state, config):
                from src.liquidity_assessment import (
                    LiquidityAssessment,
                    liquidity_fast_fail_update,
                )

                result = await node(state, config)
                assessment = LiquidityAssessment(
                    status="FAIL_INSUFFICIENT_LIQUIDITY",
                    average_daily_turnover_usd=86_436,
                )
                result["liquidity_assessment"] = assessment.to_dict()
                result.update(liquidity_fast_fail_update(assessment.to_dict()))
                return result

            return _illiquid_market

        async def validator_node(state, config):
            calls.append("validator")
            return {
                "pre_screening_result": "PASS",
                "analysis_outcome": {
                    "schema_version": 1,
                    "eligibility": "QUALIFIES",
                    "run_status": "COMPLETED",
                    "reason_codes": [],
                },
                "financial_validation_complete": True,
            }

        def researcher_node(_llm, _memory, agent_key, round_num=1):
            prefix = "bull" if agent_key == "bull_researcher" else "bear"
            field = f"{prefix}_round{round_num}"

            async def _node(state, config):
                calls.append(field)
                return {
                    "investment_debate_state": {
                        field: f"{field} done",
                        "count": state.get("investment_debate_state", {}).get(
                            "count", 0
                        )
                        + 1,
                    }
                }

            return _node

        async def research_manager_node(state, config):
            calls.append("research_manager")
            return {"investment_plan": "RECOMMENDATION: BUY"}

        def risk_node(_llm, agent_key):
            field = {
                "risky_analyst": "current_risky_response",
                "safe_analyst": "current_safe_response",
                "neutral_analyst": "current_neutral_response",
            }[agent_key]

            async def _node(state, config):
                calls.append(agent_key)
                return {
                    "risk_debate_state": {
                        field: f"{agent_key} done",
                        "latest_speaker": agent_key,
                    }
                }

            return _node

        monkeypatch.setattr(components, "_create_legacy_memories", lambda: (None,) * 5)
        monkeypatch.setattr(
            components,
            "_is_auditor_enabled",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(components, "create_quick_thinking_llm", lambda **_: Mock())
        monkeypatch.setattr(components, "create_deep_thinking_llm", lambda **_: Mock())
        monkeypatch.setattr(components, "get_consultant_llm", lambda **_: Mock())
        monkeypatch.setattr(components, "create_auditor_llm", lambda **_: Mock())
        monkeypatch.setattr(
            components, "create_apac_specialist_llm", lambda **_: Mock()
        )
        monkeypatch.setattr(components, "create_analyst_node", analyst_node)
        monkeypatch.setattr(
            components,
            "create_legal_counsel_node",
            lambda *_args, **_kwargs: artifact_node("legal_report", "legal done"),
        )
        monkeypatch.setattr(
            components,
            "create_auditor_node",
            lambda *_args, **_kwargs: artifact_node("auditor_report", "auditor done"),
        )
        monkeypatch.setattr(
            components,
            "create_financial_health_validator_node",
            lambda **_kwargs: validator_node,
        )
        monkeypatch.setattr(components, "create_researcher_node", researcher_node)
        monkeypatch.setattr(
            components,
            "create_research_manager_node",
            lambda *_args, **_kwargs: research_manager_node,
        )
        monkeypatch.setattr(
            components,
            "create_valuation_calculator_node",
            lambda *_args, **_kwargs: artifact_node(
                "valuation_params", "valuation done"
            ),
        )
        monkeypatch.setattr(
            components,
            "create_apac_specialist_node",
            lambda *_args, **_kwargs: artifact_node(
                "apac_regional_report", "apac done"
            ),
        )
        monkeypatch.setattr(
            components,
            "create_consultant_node",
            lambda *_args, **_kwargs: artifact_node(
                "consultant_review", "consultant done"
            ),
        )
        monkeypatch.setattr(
            components,
            "create_trader_node",
            lambda *_args, **_kwargs: artifact_node(
                "trader_investment_plan", "trader done"
            ),
        )
        monkeypatch.setattr(components, "create_risk_debater_node", risk_node)
        monkeypatch.setattr(
            components,
            "create_portfolio_manager_node",
            lambda *_args, **_kwargs: artifact_node("final_trade_decision", "pm done"),
        )
        monkeypatch.setattr(
            components,
            "create_chart_generator_node",
            lambda *_args, **_kwargs: lambda _state, _config=None: {"chart_paths": {}},
        )
        monkeypatch.setattr(
            components,
            "create_agent_tool_node",
            lambda *_args, **_kwargs: lambda _state, _config: {},
        )

        for name in (
            "get_market_tools",
            "get_technical_tools",
            "get_sentiment_tools",
            "get_news_tools",
            "get_junior_fundamental_tools",
            "get_senior_fundamental_tools",
            "get_foreign_language_tools",
            "get_legal_tools",
            "get_value_trap_tools",
            "get_all_tools",
        ):
            monkeypatch.setattr(components.toolkit, name, lambda: [])

        from src.graph import create_trading_graph

        graph = create_trading_graph(enable_memory=False, ticker="TEST")
        result = await graph.ainvoke({"company_of_interest": "TEST"})

        if liquidity_reject:
            assert "VERDICT: DO NOT INITIATE" in result["final_trade_decision"]
            assert result["decision_policy"]["source"] == "deterministic_screen"
            assert calls.count("final_trade_decision") == 0
            assert result["pre_screening_result"] == "REJECT"
            assert calls.count("trader_investment_plan") == 0
            assert calls.count("research_manager") == 0
            assert calls.count("risky_analyst") == 0
            assert calls.count("safe_analyst") == 0
            assert calls.count("neutral_analyst") == 0
        else:
            assert result["final_trade_decision"] == "pm done"
            assert calls.count("final_trade_decision") == 1
            assert calls.count("trader_investment_plan") == 1
            assert calls.count("risky_analyst") == 1
            assert calls.count("safe_analyst") == 1
            assert calls.count("neutral_analyst") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAuditorGateFollowsTheBindingPlan:
    """The binding plan owns full-versus-quick Auditor availability."""

    @staticmethod
    def _new_schema_settings(**over):
        from src.config import Settings

        return Settings(
            _env_file=None,
            google_api_key="g",
            moonshot_api_key="m",
            finnhub_api_key="f",
            tavily_api_key="t",
            llm_base_provider="google",
            llm_review_provider="moonshot",
            **over,
        )

    def test_auditor_is_full_only_on_a_non_openai_review_plane(self):
        from src.graph.routing import dispatch_destinations
        from src.llm_runtime.bindings import resolve_binding_plan
        from src.llm_runtime.seats import SeatId

        settings = self._new_schema_settings()
        plan = resolve_binding_plan(settings)

        assert plan.status_for(SeatId.AUDITOR).enabled is True
        assert plan.status_for(SeatId.AUDITOR, quick_mode=True).enabled is False
        assert "Auditor" in dispatch_destinations(
            include_auditor=plan.status_for(SeatId.AUDITOR).enabled
        )
        assert "Auditor" not in dispatch_destinations(
            include_auditor=plan.status_for(SeatId.AUDITOR, quick_mode=True).enabled
        )

    def test_auditor_disabled_when_the_seat_mode_is_off(self):
        from src.llm_runtime.bindings import resolve_binding_plan
        from src.llm_runtime.seats import SeatId

        settings = self._new_schema_settings(llm_auditor_mode="off")
        plan = resolve_binding_plan(settings)

        assert plan.status_for(SeatId.AUDITOR).enabled is False
        assert plan.status_for(SeatId.AUDITOR, quick_mode=True).enabled is False

    def test_auditor_disabled_when_the_review_credential_is_missing(self):
        from src.config import Settings
        from src.llm_runtime.bindings import resolve_binding_plan
        from src.llm_runtime.seats import SeatId

        settings = Settings(
            _env_file=None,
            google_api_key="g",
            finnhub_api_key="f",
            tavily_api_key="t",
            llm_base_provider="google",
            llm_review_provider="moonshot",
        )
        plan = resolve_binding_plan(settings)

        assert plan.status_for(SeatId.AUDITOR).enabled is False
        assert plan.status_for(SeatId.AUDITOR, quick_mode=True).enabled is False

    def test_legacy_schema_uses_the_same_full_only_policy(self):
        from src.config import Settings
        from src.llm_runtime.bindings import resolve_binding_plan
        from src.llm_runtime.seats import SeatId

        legacy = Settings(
            _env_file=None,
            google_api_key="g",
            openai_api_key="o",
            finnhub_api_key="f",
            tavily_api_key="t",
            enable_consultant=True,
        )
        plan = resolve_binding_plan(legacy)

        assert plan.status_for(SeatId.AUDITOR).enabled is True
        assert plan.status_for(SeatId.AUDITOR, quick_mode=True).enabled is False

    def test_legacy_schema_disables_auditor_without_the_openai_credential(self):
        from src.config import Settings
        from src.llm_runtime.bindings import resolve_binding_plan
        from src.llm_runtime.seats import SeatId

        legacy = Settings(
            _env_file=None,
            google_api_key="g",
            openai_api_key="",
            finnhub_api_key="f",
            tavily_api_key="t",
            enable_consultant=True,
        )
        plan = resolve_binding_plan(legacy)

        assert plan.status_for(SeatId.AUDITOR).enabled is False
        assert plan.status_for(SeatId.AUDITOR, quick_mode=True).enabled is False
