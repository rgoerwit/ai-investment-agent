"""The real CLI flow under a run deadline that is already spent.

The pipeline watchdog hands the child ``PIPELINE_TICKER_DEADLINE_EPOCH``; when it is
already past, ``run_deadline()`` yields a spent deadline, admission refuses every
LLM request, and the run must fail closed: diagnostics kept, nothing publishable,
a technical-failure exit code. ``tests/test_main_cli.py`` checks the binding with
``_execute_analysis`` stubbed; this test runs the real graph through
``run_with_args``.

Nothing may leave the process: every non-unix socket, name lookup and libcurl
request is refused and recorded, and the test asserts the record is empty.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from tests.helpers.offline_network import install_offline_network


@pytest.fixture
def network_attempts(monkeypatch):
    return install_offline_network(monkeypatch)


@pytest.mark.parametrize(
    "memory_flag", ["--no-memory", None], ids=["no_memory", "memory"]
)
def test_spent_run_deadline_fails_closed_through_the_real_main_flow(
    monkeypatch, tmp_path, capsys, network_attempts, memory_flag
):
    from src import cli
    from src.config import config
    from src.llm_runtime import run_deadline as run_deadline_mod
    from src.main import run_with_args
    from src.observability import NoopObservabilityRuntime

    # The real env path: the watchdog's epoch, a minute in the past. Nothing
    # from an operator .env may stand in for it.
    monkeypatch.setattr("src.config._cached_env_file_values", lambda: {})
    monkeypatch.setenv("PIPELINE_TICKER_DEADLINE_EPOCH", str(time.time() - 60))
    monkeypatch.setattr(run_deadline_mod, "_LOGGED", set())
    monkeypatch.setattr(config, "results_dir", tmp_path / "results")
    # Startup DNS checks and tracing are outside this contract.
    monkeypatch.setattr("src.main.run_provider_preflight", lambda: {})
    monkeypatch.setattr(
        "src.observability.get_observability_runtime",
        lambda *args, **kwargs: NoopObservabilityRuntime(),
    )
    monkeypatch.setattr("src.observability.flush_traces", lambda: None)

    # Publication runs on the reserve, outside the run deadline, so its
    # company-name lookup would otherwise query Yahoo.
    async def no_company_name(_ticker):
        return None

    monkeypatch.setattr(
        "src.main.output._load_company_name_for_output_async", no_company_name
    )

    import src.main as main_module

    seen_results: list[dict] = []
    real_attach = main_module._attach_run_summary

    def spy_attach(result, args, preflight):
        real_attach(result, args, preflight)
        seen_results.append(result)

    monkeypatch.setattr(main_module, "_attach_run_summary", spy_attach)

    # Memory work under the spent deadline: constructions, embedding clients
    # built and startup probes sent.
    import src.memory as memory_module

    constructed: list[str] = []
    real_init = memory_module.FinancialSituationMemory.__init__

    def spy_init(self, name):
        constructed.append(name)
        real_init(self, name)

    monkeypatch.setattr(memory_module.FinancialSituationMemory, "__init__", spy_init)
    built: list[object] = []
    real_build = memory_module.build_embeddings
    monkeypatch.setattr(
        memory_module,
        "build_embeddings",
        lambda *a, **k: built.append(a) or real_build(*a, **k),
    )
    memory_module.FinancialSituationMemory._reset_shared_state_for_tests()

    args = cli.build_arg_parser().parse_args(
        [
            "--ticker",
            "7203.T",
            "--quick",
            "--quiet",
            "--no-charts",
            # Name resolution is refused offline; without this the pre-LLM
            # data-vacuum gate stops the run before the graph.
            "--force-data-vacuum",
            *([memory_flag] if memory_flag else []),
        ]
    )

    exit_code = asyncio.run(run_with_args(args))
    out = capsys.readouterr().out

    assert network_attempts == []
    # No embeddings client is built, so no probe can be sent: with memory off
    # no memory object exists, and with memory on the spent deadline defers
    # the probe (and caches nothing) instead of running it for its full bound.
    assert built == []
    if memory_flag:
        assert constructed == []
    else:
        # The memory path really ran: at least the lessons memory was built.
        assert constructed
    memory_module.FinancialSituationMemory._reset_shared_state_for_tests()

    # A saved but technically invalid run, never a success.
    assert exit_code == 2
    (result,) = seen_results
    validity = result["analysis_validity"]
    assert validity["publishable"] is False
    failures = validity["required_failures"]
    for artifact in ("market_report", "news_report", "sentiment_report"):
        failure = failures[artifact]
        assert failure["ok"] is False, (artifact, failure)
        assert failure["error_kind"] == "timeout", (artifact, failure)
        assert "exceeded its overall LLM deadline" in failure["message"], artifact

    # The diagnostics reach the operator, and no verdict is presented.
    assert "ANALYSIS FAILED" in out
    assert "did not produce a publishable analysis" in out
    assert "Market Analyst exceeded its overall LLM deadline" in out
    for verdict in ("**BUY**", "**HOLD**", "DO NOT INITIATE**"):
        assert verdict not in out
