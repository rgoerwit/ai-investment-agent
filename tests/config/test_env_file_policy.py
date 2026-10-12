"""No settings class or `.env` reader may ingest the operator's `.env` under pytest.

`Settings` honoured ``PYTEST_DISABLE_DOTENV``; `IbkrSettings` and
`DashboardSettings` hard-coded ``env_file=".env"`` and `get_env_value` parsed
``.env`` directly, so the operator's IBKR credentials reached unit tests (four of
them tried to connect) and `.env`-only values leaked through the reader.
`src.env_file.default_settings_env_file` is now the single answer; this module
guards it structurally (every `BaseSettings` subclass) and behaviourally (a real
temporary `.env` with a distinctive value).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src import config as config_module
from src.env_file import DISABLE_DOTENV_VARIABLE, default_settings_env_file
from src.ibkr_config import IbkrSettings
from src.web.ibkr_dashboard.settings import DashboardSettings

SRC = Path(__file__).resolve().parents[2] / "src"
POLICY_FUNCTION = "default_settings_env_file"


def _settings_classes_missing_policy(source: str) -> list[str]:
    """Names of `BaseSettings` subclasses whose `env_file` is not the policy call."""
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            isinstance(base, ast.Name) and base.id == "BaseSettings"
            for base in node.bases
        ):
            continue
        if not any(_env_file_is_policy(value) for value in _env_file_values(node)):
            offenders.append(node.name)
    return offenders


def _env_file_values(cls: ast.ClassDef) -> list[ast.expr]:
    """The `env_file` expression of every `model_config` (call or dict) in `cls`."""
    values: list[ast.expr] = []
    for stmt in cls.body:
        target = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
        if not (isinstance(target, ast.Name) and target.id == "model_config"):
            continue
        config_value = stmt.value
        if isinstance(config_value, ast.Call):
            values.extend(
                kw.value for kw in config_value.keywords if kw.arg == "env_file"
            )
        elif isinstance(config_value, ast.Dict):
            values.extend(
                v
                for k, v in zip(config_value.keys, config_value.values, strict=True)
                if isinstance(k, ast.Constant) and k.value == "env_file"
            )
    return values


def _env_file_is_policy(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == POLICY_FUNCTION
    )


def test_every_basesettings_subclass_in_src_takes_env_file_from_the_policy():
    offenders = {}
    scanned = 0
    for path in sorted(SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "BaseSettings" not in source:
            continue
        scanned += 1
        bad = _settings_classes_missing_policy(source)
        if bad:
            offenders[str(path.relative_to(SRC))] = bad
    assert scanned >= 3  # config, ibkr_config, ibkr_dashboard.settings
    assert offenders == {}


@pytest.mark.parametrize(
    "source",
    [
        # the exact shapes IbkrSettings / DashboardSettings used to have
        "class S(BaseSettings):\n    model_config = SettingsConfigDict(env_file='.env')\n",
        "class S(BaseSettings):\n    model_config = {'env_file': '.env'}\n",
        # no env_file at all would read the pydantic default (none) -- still flagged:
        # a new class must opt in to the policy explicitly
        "class S(BaseSettings):\n    model_config = SettingsConfigDict(extra='ignore')\n",
        "class S(BaseSettings):\n    x: int = 1\n",
    ],
)
def test_the_guard_flags_planted_defects(source):
    assert _settings_classes_missing_policy(source) == ["S"]


@pytest.mark.parametrize(
    "source",
    [
        "class S(BaseSettings):\n    model_config = SettingsConfigDict(env_file=default_settings_env_file())\n",
        "class S(BaseSettings):\n    model_config = {'env_file': default_settings_env_file()}\n",
        "class S(BaseModel):\n    model_config = {'env_file': '.env'}\n",
    ],
)
def test_the_guard_accepts_compliant_or_unrelated_classes(source):
    assert _settings_classes_missing_policy(source) == []


def test_policy_returns_none_only_when_the_harness_switch_is_exactly_true(monkeypatch):
    monkeypatch.setenv(DISABLE_DOTENV_VARIABLE, "true")
    assert default_settings_env_file() is None
    for other in ("True", "1", "false", ""):
        monkeypatch.setenv(DISABLE_DOTENV_VARIABLE, other)
        assert default_settings_env_file() == ".env"
    monkeypatch.delenv(DISABLE_DOTENV_VARIABLE)
    assert default_settings_env_file() == ".env"


@pytest.fixture
def operator_dotenv(tmp_path, monkeypatch):
    """A working-directory `.env` carrying distinctive operator values."""
    (tmp_path / ".env").write_text(
        "LLM_CALL_HARD_TIMEOUT_SECONDS=777\n"
        "IBKR_ACCOUNT_ID=U-OPERATOR\n"
        "IBKR_DASHBOARD_PORT=7777\n"
        "GITHUB_RAW_BASE=https://operator.invalid/raw\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    config_module._cached_env_file_values.cache_clear()
    yield tmp_path
    config_module._cached_env_file_values.cache_clear()


def test_the_suite_runs_with_the_dotenv_switch_on():
    assert default_settings_env_file() is None


def test_main_settings_ignore_an_operator_dotenv(operator_dotenv):
    assert config_module.Settings().llm_call_hard_timeout_seconds == 120.0


def test_ibkr_settings_ignore_an_operator_dotenv(operator_dotenv):
    assert IbkrSettings().ibkr_account_id != "U-OPERATOR"


def test_dashboard_settings_ignore_an_operator_dotenv(operator_dotenv):
    assert DashboardSettings().port == 5050


def test_get_env_value_does_not_serve_the_operator_dotenv(operator_dotenv):
    assert config_module._parse_env_file() == {}
    assert config_module.get_env_value("GITHUB_RAW_BASE") is None


def test_the_same_dotenv_is_read_once_the_switch_is_lifted(
    operator_dotenv, monkeypatch
):
    """Proves the fixtures above bite: with the switch off the file is visible."""
    monkeypatch.delenv(DISABLE_DOTENV_VARIABLE)
    assert config_module._parse_env_file()["GITHUB_RAW_BASE"] == (
        "https://operator.invalid/raw"
    )
    assert IbkrSettings.model_config["env_file"] is None  # fixed at import
    assert (
        IbkrSettings(_env_file=operator_dotenv / ".env").ibkr_account_id == "U-OPERATOR"
    )
