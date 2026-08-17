"""Hygiene guards for configuration settings.

The codebase accumulated two embarrassing classes of config bug:

1. **Dead config** — a field (and the env var it advertises) declared on a
   Settings model but never read, so e.g. ``IBKR_CASH_BUFFER_PCT`` silently did
   nothing and ``MAX_POSITION_SIZE`` / ``MAX_RISK_DISCUSS_ROUNDS`` were inert.
2. **Duplicated / drifting defaults** — the same logical default re-literalized
   across CLI argparse defaults, function signatures, ``IbkrSettings`` and the
   dashboard, which drifted (cash buffer was 0.03 in some signatures while the
   operative CLI/dashboard default was 0.05).

These tests fail if either pattern reappears. They complement (do not duplicate)
the existing coverage:
  * downstream effect of ``cash_buffer`` → ``tests/ibkr/test_portfolio.py``
    (``test_zero_cash_buffer`` / ``test_high_cash_buffer``);
  * env→rate-limiter behavior → ``tests/config/test_rate_limit_configuration.py``;
  * single source of truth for the constants → ``tests/ibkr/test_portfolio_defaults.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from src.config import Settings
from src.ibkr_config import IbkrSettings
from src.runtime_config import RuntimeConfig
from src.web.ibkr_dashboard.settings import DashboardPreferences, DashboardSettings

_REPO = Path(__file__).resolve().parents[2]
_SRC = _REPO / "src"
_SCRIPTS = _REPO / "scripts"


def _source_lines(*, exclude_paths: set[Path] | None = None) -> list[str]:
    lines: list[str] = []
    for root in (_SRC, _SCRIPTS):
        for path in root.rglob("*.py"):
            if exclude_paths and path in exclude_paths:
                continue
            lines.extend(path.read_text(encoding="utf-8").splitlines())
    return lines


_SOURCE_LINES = _source_lines()


def _field_is_consumed(field: str, *, lines: list[str] | None = None) -> bool:
    """True if ``field`` appears on a non-declaration line in src/ or scripts/.

    The model declaration line is ``name: Type = ...`` (annotation → colon), which
    is excluded. Any other occurrence — ``self.name``, ``config.name``, a
    ``name=...`` keyword argument, or a ``get_name`` body referencing ``self.name``
    — counts as a real consumer.
    """
    token = re.compile(rf"\b{re.escape(field)}\b")
    decl = re.compile(rf"^\s*{re.escape(field)}\s*:")
    return any(
        token.search(ln) and not decl.match(ln) for ln in (lines or _SOURCE_LINES)
    )


_MODELS = [Settings, IbkrSettings, DashboardSettings, DashboardPreferences]
_FIELD_CASES = [
    pytest.param(cls.__name__, name, id=f"{cls.__name__}.{name}")
    for cls in _MODELS
    for name in cls.model_fields
]


@pytest.mark.parametrize("model, field", _FIELD_CASES)
def test_no_dead_config(model, field):
    """Every settings field must be consumed somewhere in src/ or scripts/.

    A failure means a config field (and any env override it advertises) is
    declared but never read. Wire it into the code path it is meant to control,
    or delete it (and its .env.example entry).
    """
    assert _field_is_consumed(field), (
        f"{model}.{field} is declared but never consumed in src/ or scripts/. "
        f"Wire it up or remove it — do not ship dead config."
    )


_RUNTIME_CONFIG_CONSUMER_LINES = _source_lines(
    exclude_paths={_SRC / "runtime_config.py"}
)
_RUNTIME_CONFIG_FIELD_CASES = [
    pytest.param(field.name, id=f"RuntimeConfig.{field.name}")
    for field in dataclass_fields(RuntimeConfig)
]


@pytest.mark.parametrize("field", _RUNTIME_CONFIG_FIELD_CASES)
def test_runtime_config_field_is_consumed_outside_runtime_config(field):
    """Run-scoped config must be used outside its own construction helpers.

    This catches accepted-but-ignored runtime fields that are copied into
    RuntimeConfig but never change actual analyzer behavior.
    """
    assert _field_is_consumed(field, lines=_RUNTIME_CONFIG_CONSUMER_LINES), (
        f"RuntimeConfig.{field} is not consumed outside src/runtime_config.py. "
        f"Use it in a runtime path or remove it."
    )


# --- Env-var overrides are honored ------------------------------------------

_SETTINGS_ENV_CASES = [
    ("api_retry_attempts", "API_RETRY_ATTEMPTS", "7", 7),
    ("gemini_rpm_limit", "GEMINI_RPM_LIMIT", "123", 123),
    ("enable_consultant", "ENABLE_CONSULTANT", "false", False),
    ("max_debate_rounds", "MAX_DEBATE_ROUNDS", "1", 1),
    (
        "apex_model",
        "APEX_MODEL",
        "gemini-3.1-pro-preview",
        "gemini-3.1-pro-preview",
    ),
    (
        "apex_quick_model",
        "APEX_QUICK_MODEL",
        "gemini-3.5-flash",
        "gemini-3.5-flash",
    ),
    ("apex_thinking_level", "APEX_THINKING_LEVEL", "medium", "medium"),
    (
        "retrospective_max_evaluations_per_run",
        "RETROSPECTIVE_MAX_EVALUATIONS_PER_RUN",
        "20",
        20,
    ),
]


@pytest.mark.parametrize("field, alias, raw, expected", _SETTINGS_ENV_CASES)
def test_settings_env_override_is_honored(monkeypatch, field, alias, raw, expected):
    monkeypatch.setenv(alias, raw)
    assert getattr(Settings(), field) == expected


_IBKR_ENV_CASES = [
    ("ibkr_cash_buffer_pct", "IBKR_CASH_BUFFER_PCT", "0.07", 0.07),
    ("ibkr_max_analysis_age_days", "IBKR_MAX_ANALYSIS_AGE_DAYS", "21", 21),
    ("ibkr_drift_threshold_pct", "IBKR_DRIFT_THRESHOLD_PCT", "9.5", 9.5),
]


@pytest.mark.parametrize("field, alias, raw, expected", _IBKR_ENV_CASES)
def test_ibkr_settings_env_override_is_honored(
    monkeypatch, field, alias, raw, expected
):
    monkeypatch.setenv(alias, raw)
    assert getattr(IbkrSettings(), field) == expected


def test_dashboard_env_prefix_override(monkeypatch):
    monkeypatch.setenv("IBKR_DASHBOARD_CASH_BUFFER", "0.08")
    monkeypatch.setenv("IBKR_DASHBOARD_MAX_AGE_DAYS", "30")
    settings = DashboardSettings()
    assert settings.cash_buffer == 0.08
    assert settings.max_age_days == 30


# --- Downstream effect + CLI/env precedence for the portfolio knobs ----------


def test_cash_buffer_env_changes_available_cash(monkeypatch):
    """End-to-end: IBKR_CASH_BUFFER_PCT must flow through to the available-cash
    computation, not just sit on the model."""
    from src.ibkr.portfolio import build_portfolio_summary

    ledger = {"BASE": {"cashbalance": 10_000, "netliquidationvalue": 100_000}}
    monkeypatch.setenv("IBKR_CASH_BUFFER_PCT", "0.10")
    pct = IbkrSettings().ibkr_cash_buffer_pct
    summary = build_portfolio_summary(ledger, [], "U1", cash_buffer_pct=pct)
    # available = 10_000 - 100_000 * 0.10 = 0
    assert summary.available_cash_usd == 0.0


@pytest.mark.parametrize(
    "attr, flag, env_alias, env_attr, env_value, expected_default, flag_value",
    [
        (
            "cash_buffer",
            "--cash-buffer",
            "IBKR_CASH_BUFFER_PCT",
            "ibkr_cash_buffer_pct",
            0.07,
            0.07,
            "0.02",
        ),
        (
            "max_age",
            "--max-age",
            "IBKR_MAX_ANALYSIS_AGE_DAYS",
            "ibkr_max_analysis_age_days",
            21,
            21,
            "3",
        ),
        (
            "drift_pct",
            "--drift-pct",
            "IBKR_DRIFT_THRESHOLD_PCT",
            "ibkr_drift_threshold_pct",
            9.5,
            9.5,
            "1.0",
        ),
    ],
)
def test_cli_default_from_env_and_flag_overrides(
    monkeypatch,
    attr,
    flag,
    env_alias,
    env_attr,
    env_value,
    expected_default,
    flag_value,
):
    """CLI flag > env (IbkrSettings) > constant. The argparse default must come
    from the env-aware config, and the matching flag must still override it."""
    import scripts.portfolio_manager as pm
    from src.ibkr_config import ibkr_config

    monkeypatch.setattr(ibkr_config, env_attr, env_value, raising=False)

    # No flag → default reflects the env-derived config value.
    assert getattr(pm.parse_args([]), attr) == expected_default

    # Flag present → overrides the env-derived default.
    got = getattr(pm.parse_args([flag, flag_value]), attr)
    assert got == type(expected_default)(flag_value)


def test_removed_risk_discussion_rounds_knob_stays_removed():
    """The graph never implemented a risk-discussion loop, so the accepted-but-
    ignored MAX_RISK_DISCUSS_ROUNDS knob must not come back as dead config."""
    assert "max_risk_discuss_rounds" not in (_SRC / "main.py").read_text()
    assert "max_risk_discuss_rounds" not in (_SRC / "graph" / "builder.py").read_text()


def test_quick_mode_active_is_runtime_only(monkeypatch):
    """QUICK_MODE_ACTIVE must not be env-backed on Settings; --quick owns it."""
    from src.runtime_config import RuntimeConfig

    monkeypatch.setenv("QUICK_MODE_ACTIVE", "true")
    settings = Settings(_env_file=None)

    assert "quick_mode_active" not in Settings.model_fields
    assert not hasattr(settings, "quick_mode_active")
    assert RuntimeConfig.from_config(settings).quick_mode_active is False


# --- No re-literalized duplicate defaults ------------------------------------

# Scoped to the portfolio-reconciliation domain — ``max_age_days`` is also used for
# an unrelated ticker-history cache (ticker_history_resolver.py, 120 days) that has
# nothing to do with the portfolio knob, so the scan must not reach outside this domain.
_PORTFOLIO_DOMAIN_LINES = [
    ln
    for path in [
        *(_SRC / "ibkr").rglob("*.py"),
        *(_SRC / "web" / "ibkr_dashboard").rglob("*.py"),
        _SCRIPTS / "portfolio_manager.py",
    ]
    for ln in path.read_text(encoding="utf-8").splitlines()
]

_SHARED_KNOB_PARAMS = [
    "cash_buffer_pct",
    "max_age_days",
    "drift_threshold_pct",
    "sector_limit_pct",
    "exchange_limit_pct",
    "overweight_threshold_pct",
    "underweight_threshold_pct",
]


@pytest.mark.parametrize("param", _SHARED_KNOB_PARAMS)
def test_no_duplicate_literal_default_for_portfolio_knob(param):
    """A shared portfolio knob must never be given a numeric literal default in a
    function signature — it must reference a ``portfolio_defaults`` constant."""
    bad = re.compile(rf"\b{re.escape(param)}\s*:\s*(?:float|int)\s*=\s*[0-9]")
    offenders = [ln.strip() for ln in _PORTFOLIO_DOMAIN_LINES if bad.search(ln)]
    assert not offenders, (
        f"{param} has a re-literalized numeric default; reference the "
        f"src.ibkr.portfolio_defaults constant instead. Offenders: {offenders}"
    )


# --- Override precedence: CLI flag > shell env > .env > hardcoded default -----
#
# Canonical order (confirmed with the user). These tests differentiate which knobs
# expose which override surfaces, and assert the precedence actually holds.

import tempfile  # noqa: E402

import src.ibkr.portfolio_defaults as _pd  # noqa: E402

# Portfolio CLI flag → (argparse dest, IbkrSettings env alias or None when CLI-only).
_PORTFOLIO_OVERRIDE_MATRIX = {
    "--cash-buffer": ("cash_buffer", "IBKR_CASH_BUFFER_PCT"),
    "--max-age": ("max_age", "IBKR_MAX_ANALYSIS_AGE_DAYS"),
    "--drift-pct": ("drift_pct", "IBKR_DRIFT_THRESHOLD_PCT"),
    "--sector-limit": ("sector_limit", None),
    "--exchange-limit": ("exchange_limit", None),
    "--refresh-limit": ("refresh_limit", None),
}


def _parse_pm(argv: list[str]):
    import scripts.portfolio_manager as pm

    return pm.parse_args(argv)


@pytest.mark.parametrize("flag", list(_PORTFOLIO_OVERRIDE_MATRIX))
def test_every_portfolio_cli_flag_overrides_its_default(monkeypatch, flag):
    """Each portfolio CLI flag must actually override (CLI is top of the chain)."""
    dest, _ = _PORTFOLIO_OVERRIDE_MATRIX[flag]
    sample = {"cash_buffer": "0.011", "drift_pct": "7.5"}.get(dest, "5")
    args = _parse_pm([flag, sample])
    assert float(getattr(args, dest)) == float(sample)


@pytest.mark.parametrize(
    "flag", [f for f, (_, env) in _PORTFOLIO_OVERRIDE_MATRIX.items() if env is None]
)
def test_cli_only_knob_has_no_env_surface(flag):
    """Differentiator: sector/exchange/refresh limits are CLI-only — no IbkrSettings
    env field feeds them, so their default is the portfolio_defaults constant."""
    dest, _ = _PORTFOLIO_OVERRIDE_MATRIX[flag]
    const = {
        "sector_limit": _pd.DEFAULT_SECTOR_LIMIT_PCT,
        "exchange_limit": _pd.DEFAULT_EXCHANGE_LIMIT_PCT,
        "refresh_limit": _pd.DEFAULT_REFRESH_LIMIT,
    }[dest]
    # No IbkrSettings field advertises an env override for these.
    aliases = {str(f.validation_alias) for f in IbkrSettings.model_fields.values()}
    assert dest.upper() not in {a.upper() for a in aliases}
    # And the field default is the centralized constant.
    import argparse

    from src.ibkr.cli_options import add_common_portfolio_request_args

    parser = argparse.ArgumentParser()
    add_common_portfolio_request_args(
        parser,
        read_only_help="x",
        account_id_help="x",
        results_dir_help="x",
        watchlist_help="x",
    )
    assert getattr(parser.parse_args([]), dest) == const


def test_env_only_setting_has_no_portfolio_cli_flag(monkeypatch):
    """Differentiator: a pure env knob (GEMINI_RPM_LIMIT) is overridable by env but
    has no portfolio CLI flag."""
    monkeypatch.setenv("GEMINI_RPM_LIMIT", "777")
    assert Settings().gemini_rpm_limit == 777  # env override works
    args = _parse_pm([])
    assert not hasattr(args, "gemini_rpm_limit")  # no CLI surface on this parser


def test_gpt56_model_ids_load_from_dotenv(tmp_path, monkeypatch):
    dotenv_path = tmp_path / "gpt56.env"
    dotenv_path.write_text(
        "\n".join(
            (
                "CONSULTANT_MODEL=gpt-5.6-terra",
                "CONSULTANT_QUICK_MODEL=gpt-5.6-luna",
                "AUDITOR_MODEL=gpt-5.6-sol",
                "AUDITOR_QUICK_MODEL=gpt-5.6-luna",
                "EDITOR_MODEL=gpt-5.6-terra",
            )
        ),
        encoding="utf-8",
    )
    for env_name in (
        "CONSULTANT_MODEL",
        "CONSULTANT_QUICK_MODEL",
        "AUDITOR_MODEL",
        "AUDITOR_QUICK_MODEL",
        "EDITOR_MODEL",
    ):
        monkeypatch.delenv(env_name, raising=False)

    settings = Settings(_env_file=dotenv_path)

    assert settings.consultant_model == "gpt-5.6-terra"
    assert settings.consultant_quick_model == "gpt-5.6-luna"
    assert settings.auditor_model == "gpt-5.6-sol"
    assert settings.auditor_quick_model == "gpt-5.6-luna"
    assert settings.editor_model == "gpt-5.6-terra"


def test_shell_env_beats_dotenv_beats_default():
    """The middle of the chain: shell env > .env file > hardcoded default."""
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write("IBKR_CASH_BUFFER_PCT=0.055\n")
        dotenv_path = f.name

    import os

    os.environ.pop("IBKR_CASH_BUFFER_PCT", None)
    # default only (point at an empty/nonexistent dotenv)
    assert (
        IbkrSettings(_env_file=None).ibkr_cash_buffer_pct == _pd.DEFAULT_CASH_BUFFER_PCT
    )
    # .env beats default
    assert IbkrSettings(_env_file=dotenv_path).ibkr_cash_buffer_pct == 0.055
    # shell env beats .env
    os.environ["IBKR_CASH_BUFFER_PCT"] = "0.066"
    try:
        assert IbkrSettings(_env_file=dotenv_path).ibkr_cash_buffer_pct == 0.066
    finally:
        os.environ.pop("IBKR_CASH_BUFFER_PCT", None)
        os.unlink(dotenv_path)


def test_portfolio_manager_results_dir_precedence_shell_env_and_flag(monkeypatch):
    """portfolio_manager must follow analyzer RESULTS_DIR, with CLI still winning."""
    import scripts.portfolio_manager as pm

    monkeypatch.setenv("RESULTS_DIR", "/tmp/env_results")
    analyzer_config = Settings(_env_file=None)
    ibkr_settings = SimpleNamespace(
        ibkr_max_analysis_age_days=_pd.DEFAULT_MAX_AGE_DAYS,
        ibkr_cash_buffer_pct=_pd.DEFAULT_CASH_BUFFER_PCT,
        ibkr_drift_threshold_pct=_pd.DEFAULT_DRIFT_PCT,
    )

    assert Path(
        pm.parse_args(
            [],
            analyzer_config=analyzer_config,
            ibkr_settings=ibkr_settings,
        ).results_dir
    ) == Path("/tmp/env_results")

    assert Path(
        pm.parse_args(
            ["--results-dir", "/tmp/flag_results"],
            analyzer_config=analyzer_config,
            ibkr_settings=ibkr_settings,
        ).results_dir
    ) == Path("/tmp/flag_results")


def test_results_dir_shell_env_beats_dotenv_for_settings(monkeypatch):
    """The analyzer's RESULTS_DIR chain is shell env > .env > default."""
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
        f.write("RESULTS_DIR=/tmp/dotenv_results\n")
        dotenv_path = f.name

    import os

    monkeypatch.delenv("RESULTS_DIR", raising=False)
    try:
        assert Settings(_env_file=None).results_dir == Path("results")
        assert Settings(_env_file=dotenv_path).results_dir == Path(
            "/tmp/dotenv_results"
        )
        monkeypatch.setenv("RESULTS_DIR", "/tmp/shell_results")
        assert Settings(_env_file=dotenv_path).results_dir == Path("/tmp/shell_results")
    finally:
        os.unlink(dotenv_path)


def test_main_analyzer_cli_flag_overrides_config():
    """The other entry point: build_runtime_config applies CLI flags on top of the
    env/.env-driven base config (CLI > config for the 8 run-scoped fields)."""
    from types import SimpleNamespace

    from src.config import config
    from src.runtime_config import build_runtime_config

    # No flag → inherits base config (env/.env/default chain).
    base = build_runtime_config(SimpleNamespace(), config)
    assert base.quick_think_llm == config.quick_think_llm
    # Flag present → overrides.
    overridden = build_runtime_config(
        SimpleNamespace(quick_model="zzz-quick", deep_model="zzz-deep"), config
    )
    assert overridden.quick_think_llm == "zzz-quick"
    assert overridden.deep_think_llm == "zzz-deep"


class TestRetrospectiveEvaluationBudget:
    """A probe batch must not require a source edit.

    The ceiling was a module constant, so judging a policy change on a small
    batch meant editing `retrospective.py` — which is exactly the moment you
    least want an uncommitted diff in the file under test.
    """

    def test_the_shipped_default_is_unchanged(self):
        """Read the field default, never `Settings()`.

        Instantiating loads the operator's `.env`, so an operator who has set a
        documented override would fail this on a correct configuration.
        """
        field = Settings.model_fields["retrospective_max_evaluations_per_run"]
        assert field.default == 400

    @pytest.mark.parametrize("bad", ["0", "-1"])
    def test_a_non_positive_budget_is_rejected(self, monkeypatch, bad):
        """A ceiling of zero silently evaluates nothing, which reads as success."""
        monkeypatch.setenv("RETROSPECTIVE_MAX_EVALUATIONS_PER_RUN", bad)
        with pytest.raises(ValidationError):
            Settings()

    def test_the_orchestrator_resolves_it_at_call_time(self, monkeypatch):
        """`None` means "ask config", so the value cannot be bound at import.

        A signature default would freeze whatever config held when the module was
        first imported — invisible, and wrong for exactly the operator who sets
        the variable to run a probe.
        """
        import inspect

        from src.retrospective import run_retrospective

        parameter = inspect.signature(run_retrospective).parameters["max_evaluations"]
        assert parameter.default is None

    @pytest.mark.asyncio
    async def test_the_configured_value_actually_bounds_selection(
        self, monkeypatch, tmp_path
    ):
        """The signature default proves nothing on its own.

        `None` could reach the budget as `None`, or be resolved from a stale
        import-time copy, and the signature test would still pass. This drives
        the real orchestrator against real snapshot files and asserts the
        configured ceiling reached `_select_within_budget`.
        """
        import src.retrospective as retro

        for index in range(5):
            (tmp_path / f"T{index}.T_20260101_00000{index}_analysis.json").write_text(
                json.dumps(
                    {
                        "prediction_snapshot": {
                            "ticker": f"T{index}.T",
                            "analysis_date": "2026-01-01",
                            "verdict": "HOLD",
                            "current_price": 100.0,
                            "currency": "JPY",
                            "benchmark_index": "^N225",
                        }
                    }
                )
            )

        monkeypatch.setattr(retro.config, "retrospective_max_evaluations_per_run", 2)
        budgets: list[int] = []
        real_select = retro._select_within_budget
        monkeypatch.setattr(
            retro,
            "_select_within_budget",
            lambda candidates, budget: (
                budgets.append(budget) or real_select(candidates, budget)
            ),
        )

        await retro.run_retrospective(
            None,
            tmp_path,
            lessons_memory=SimpleNamespace(available=False),
            memo_path=tmp_path / "memo.json",
            dry_run=True,
        )
        assert budgets == [2], (
            "run_retrospective(max_evaluations=None) must resolve the ceiling "
            f"from config at call time; got {budgets}"
        )
