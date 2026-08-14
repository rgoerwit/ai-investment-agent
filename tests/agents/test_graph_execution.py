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
        from src.graph import should_continue_analyst

        mock_message = MagicMock()
        mock_message.tool_calls = ["tool1"]

        state = {"messages": [mock_message]}
        config = {}

        result = should_continue_analyst(state, config)
        assert result == "tools"

    def test_should_continue_analyst_without_tools(self):
        """Test routing when analyst has no tool calls."""
        from src.graph import should_continue_analyst

        mock_message = MagicMock()
        mock_message.tool_calls = []

        state = {"messages": [mock_message]}
        config = {}

        result = should_continue_analyst(state, config)
        assert result == "continue"


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

    @patch("src.graph.routing.config")
    def test_sync_check_returns_end_when_incomplete(self, mock_config):
        """Test router returns __end__ when not all analysts complete."""
        from src.graph import sync_check_router

        mock_config.enable_consultant = False

        state = {
            "market_report": "done",
            "sentiment_report": "",  # Not complete
            "news_report": "done",
            "pre_screening_result": "PASS",
        }
        config = {}

        result = sync_check_router(state, config)
        assert result == "__end__"

    @patch("src.graph.routing.config")
    def test_sync_check_proceeds_when_required_branch_failed_but_completed(
        self, mock_config
    ):
        """Router should wait for completion, not success, at the sync barrier."""
        from src.graph import sync_check_router

        mock_config.enable_consultant = False

        state = {
            "market_report": "Error: DNS failure",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
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

        result = sync_check_router(state, {})
        assert isinstance(result, list)
        assert "Bull Researcher R1" in result

    @patch("src.graph.routing.config")
    def test_sync_check_returns_pm_fast_fail_on_reject(self, mock_config):
        """Test router returns PM Fast-Fail on REJECT (separate node to avoid edge conflicts)."""
        from src.graph import sync_check_router

        mock_config.enable_consultant = False

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "REJECT",
        }
        config = {}

        result = sync_check_router(state, config)
        assert result == "PM Fast-Fail"

    @patch("src.graph.routing.config")
    def test_sync_check_returns_list_for_parallel_r1(self, mock_config):
        """Test router returns list for parallel Bull/Bear R1 on PASS."""
        from src.graph import sync_check_router

        mock_config.enable_consultant = False

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
        }
        config = {}

        result = sync_check_router(state, config)
        assert isinstance(result, list)
        assert "Bull Researcher R1" in result
        assert "Bear Researcher R1" in result
        assert len(result) == 2


class TestAuditorIntegration:
    """Test auditor node integration with graph routing."""

    @patch("src.graph.routing.config")
    def test_is_auditor_enabled_when_consultant_disabled(self, mock_config):
        """Test _is_auditor_enabled returns False when consultant disabled."""
        from src.graph.routing import _is_auditor_enabled

        mock_config.enable_consultant = False
        mock_config.get_openai_api_key.return_value = "test-key"

        assert _is_auditor_enabled() is False

    @patch("src.graph.routing.is_openai_consultant_available")
    @patch("src.graph.routing.config")
    def test_is_auditor_enabled_when_no_api_key(self, mock_config, mock_available):
        """Test _is_auditor_enabled returns False when API key missing."""
        from src.graph.routing import _is_auditor_enabled

        mock_config.enable_consultant = True
        mock_available.return_value = False

        assert _is_auditor_enabled() is False

    @patch("src.graph.routing.is_openai_consultant_available")
    @patch("src.graph.routing.config")
    def test_is_auditor_enabled_when_all_conditions_met(
        self, mock_config, mock_available
    ):
        """Test _is_auditor_enabled returns True when all conditions met."""
        from src.graph.routing import _is_auditor_enabled

        mock_config.enable_consultant = True
        mock_available.return_value = True

        assert _is_auditor_enabled() is True

    @patch("src.graph.routing._is_auditor_enabled")
    def test_fan_out_includes_auditor_when_enabled(self, mock_auditor_enabled):
        """Test fan_out_to_analysts includes Auditor when enabled."""
        from src.graph import fan_out_to_analysts

        mock_auditor_enabled.return_value = True

        result = fan_out_to_analysts({}, {})
        assert "Auditor" in result
        assert "Value Trap Detector" in result
        assert len(result) == 8  # 7 analysts + Auditor

    @patch("src.graph.routing._is_auditor_enabled")
    def test_fan_out_excludes_auditor_when_disabled(self, mock_auditor_enabled):
        """Test fan_out_to_analysts excludes Auditor when disabled."""
        from src.graph import fan_out_to_analysts

        mock_auditor_enabled.return_value = False

        result = fan_out_to_analysts({}, {})
        assert "Auditor" not in result
        assert "Value Trap Detector" in result
        assert len(result) == 7

    @patch("src.graph.routing._is_auditor_enabled")
    def test_sync_check_waits_for_auditor_when_enabled(self, mock_auditor_enabled):
        """Test sync_check_router waits for auditor_report when enabled."""
        from src.graph import sync_check_router

        mock_auditor_enabled.return_value = True

        # All reports present except auditor_report
        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "pre_screening_result": "PASS",
            "auditor_report": "",  # Empty = not done
        }

        result = sync_check_router(state, {})
        assert result == "__end__"  # Should wait

    @patch("src.graph.routing._is_auditor_enabled")
    def test_sync_check_proceeds_when_auditor_failed_but_completed(
        self, mock_auditor_enabled
    ):
        """A failed enabled auditor branch should still satisfy sync completion."""
        from src.graph import sync_check_router

        mock_auditor_enabled.return_value = True

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
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

        result = sync_check_router(state, {})
        assert isinstance(result, list)
        assert "Bull Researcher R1" in result

    @patch("src.graph.routing._is_auditor_enabled")
    def test_sync_check_proceeds_when_auditor_complete(self, mock_auditor_enabled):
        """Test sync_check_router proceeds when auditor_report complete."""
        from src.graph import sync_check_router

        mock_auditor_enabled.return_value = True

        state = {
            "market_report": "done",
            "sentiment_report": "done",
            "news_report": "done",
            "value_trap_report": "done",
            "pre_screening_result": "PASS",
            "auditor_report": "Forensic audit complete",
        }

        result = sync_check_router(state, {})
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


class TestAuditorEnablementContract:
    """Ensure routing and LLM creation stay aligned on auditor availability."""

    def test_auditor_disabled_contract_stays_aligned(self, monkeypatch):
        import src.graph.routing as routing
        import src.llms as llms

        monkeypatch.setattr(routing, "is_openai_consultant_available", lambda: False)
        monkeypatch.setattr(routing.config, "enable_consultant", False)
        monkeypatch.setattr(llms.config, "enable_consultant", False)

        assert routing._is_auditor_enabled() is False
        assert llms.create_auditor_llm() is None

    def test_auditor_enabled_contract_stays_aligned(self, monkeypatch):
        import src.graph.routing as routing
        import src.llms as llms

        stub_module = ModuleType("langchain_openai")

        class StubChatOpenAI:
            def __init__(self, **kwargs):
                self.model_name = kwargs["model"]

        stub_module.ChatOpenAI = StubChatOpenAI

        monkeypatch.setitem(sys.modules, "langchain_openai", stub_module)
        monkeypatch.setattr(routing, "is_openai_consultant_available", lambda: True)
        monkeypatch.setattr(routing.config, "enable_consultant", True)
        monkeypatch.setattr(llms.config, "enable_consultant", True)
        monkeypatch.setattr(
            type(llms.config), "get_openai_api_key", lambda self: "fake-key"
        )
        monkeypatch.setattr(llms.config, "auditor_model", "gpt-5-mini")
        monkeypatch.setattr(llms.config, "consultant_model", "gpt-5")

        assert routing._is_auditor_enabled() is True
        llm = llms.create_auditor_llm()
        assert llm is not None
        assert llm.model_name == "gpt-5-mini"

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

    def test_build_graph_components_keeps_consultant_and_auditor_in_quick_mode(
        self, monkeypatch
    ):
        from src.graph.components import build_graph_components

        components = _stub_graph_component_dependencies(monkeypatch)
        quick_consultant = Mock(name="quick-consultant")
        auditor_llm = Mock(name="auditor")

        consultant_calls = []

        def fake_get_consultant_llm(**kwargs):
            consultant_calls.append(kwargs)
            return quick_consultant

        monkeypatch.setattr(components, "get_consultant_llm", fake_get_consultant_llm)
        monkeypatch.setattr(components, "_is_auditor_enabled", lambda: True)
        monkeypatch.setattr(
            components, "create_auditor_llm", lambda **kwargs: auditor_llm
        )

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
        assert graph_components.auditor_enabled is True
        assert "Consultant" in graph_components.nodes
        assert "Auditor" in graph_components.nodes
        assert consultant_calls
        assert consultant_calls[0]["quick_mode"] is True

    def test_build_graph_components_quick_mode_disables_both_when_openai_path_unavailable(
        self, monkeypatch
    ):
        from src.graph.components import build_graph_components

        components = _stub_graph_component_dependencies(monkeypatch)
        monkeypatch.setattr(components, "get_consultant_llm", lambda **kwargs: None)
        monkeypatch.setattr(components, "_is_auditor_enabled", lambda: False)

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

    def test_quick_mode_raises_when_auditor_routing_and_creation_disagree(
        self, monkeypatch
    ):
        from src.graph.components import build_graph_components

        components = _stub_graph_component_dependencies(monkeypatch)
        monkeypatch.setattr(components, "get_consultant_llm", lambda **kwargs: Mock())
        monkeypatch.setattr(components, "_is_auditor_enabled", lambda: True)
        monkeypatch.setattr(components, "create_auditor_llm", lambda **kwargs: None)

        with pytest.raises(RuntimeError, match="Auditor routing was enabled"):
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
        monkeypatch.setattr(components, "_is_auditor_enabled", lambda: True)
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
        monkeypatch.setattr(components, "_is_auditor_enabled", lambda: False)
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
        monkeypatch.setattr(components, "_is_auditor_enabled", lambda: False)

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
        """strict_mode=True is forwarded to create_portfolio_manager_node (both instances)."""
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
        # PM factory is called twice (main PM + fast-fail PM)
        calls = mock_pm.call_args_list
        assert len(calls) == 2
        assert all(call.kwargs.get("strict_mode") is True for call in calls)

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

    @pytest.mark.asyncio
    async def test_trader_tail_runs_once_after_valuation_and_consultant_complete(
        self, monkeypatch
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
            return artifact_node(output_field, f"{output_field} done")

        async def validator_node(state, config):
            calls.append("validator")
            return {"pre_screening_result": "PASS"}

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
        monkeypatch.setattr(components, "_is_auditor_enabled", lambda: True)
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

        assert result["final_trade_decision"] == "pm done"
        assert calls.count("trader_investment_plan") == 1
        assert calls.count("risky_analyst") == 1
        assert calls.count("safe_analyst") == 1
        assert calls.count("neutral_analyst") == 1
        assert calls.count("final_trade_decision") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestAuditorGateFollowsTheBindingPlan:
    """The router and the graph builder must agree on the auditor seat.

    The pre-existing guard only catches the *loud* direction — routing enabled
    while creation returns None raises. The opposite direction is **silent**: the
    node gets wired and simply never receives control, so the cross-check
    disappears with no error and only `auditor_review_status='NOT_RUN'` in the
    saved artifact to show for it. That is what happened on a migrated Moonshot
    review plane, where OPENAI_API_KEY is legitimately absent because the vendor
    key is MOONSHOT_API_KEY.
    """

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

    def test_auditor_enabled_on_a_non_openai_review_plane(self, monkeypatch):
        from src.graph import routing

        settings = self._new_schema_settings()
        monkeypatch.setattr(routing, "config", settings)
        # The legacy predicate would say no: there is no OpenAI key to find.
        monkeypatch.setattr(routing, "is_openai_consultant_available", lambda: False)

        assert routing._is_auditor_enabled() is True
        assert "Auditor" in routing.dispatch_destinations(
            include_auditor=routing._is_auditor_enabled()
        )

    def test_auditor_disabled_when_the_seat_mode_is_off(self, monkeypatch):
        from src.graph import routing

        settings = self._new_schema_settings(llm_auditor_mode="off")
        monkeypatch.setattr(routing, "config", settings)

        assert routing._is_auditor_enabled() is False

    def test_auditor_disabled_when_the_review_credential_is_missing(self, monkeypatch):
        from src.config import Settings
        from src.graph import routing

        settings = Settings(
            _env_file=None,
            google_api_key="g",
            finnhub_api_key="f",
            tavily_api_key="t",
            llm_base_provider="google",
            llm_review_provider="moonshot",
        )
        monkeypatch.setattr(routing, "config", settings)

        assert routing._is_auditor_enabled() is False

    def test_legacy_schema_still_requires_the_openai_key(self, monkeypatch):
        from src.config import Settings
        from src.graph import routing

        legacy = Settings(
            _env_file=None,
            google_api_key="g",
            finnhub_api_key="f",
            tavily_api_key="t",
            enable_consultant=True,
        )
        monkeypatch.setattr(routing, "config", legacy)
        monkeypatch.setattr(routing, "is_openai_consultant_available", lambda: False)

        assert routing._is_auditor_enabled() is False
