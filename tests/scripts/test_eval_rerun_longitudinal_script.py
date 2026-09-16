"""Black-box argument-contract tests for eval_rerun_longitudinal.sh."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "eval_rerun_longitudinal.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


@pytest.fixture
def runner_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    """Use inert command doubles; no analysis, network, or real Chroma is touched."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "python_calls.txt"
    _write_executable(
        bin_dir / "python",
        """#!/bin/bash
{
    echo CALL
    for arg in "$@"; do
        printf 'ARG=%s\\n' "$arg"
    done
    echo END
} >> "$FAKE_PYTHON_CALLS"
if [[ "${1:-}" == "scripts/scan_batch_health.py" ]]; then
    echo '{"detail": ""}'
fi
exit 0
""",
    )
    _write_executable(bin_dir / "caffeinate", "#!/bin/bash\nexit 0\n")

    ticker_file = tmp_path / "tickers.txt"
    ticker_file.write_text("TOT.TO\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "COOLDOWN_SECONDS": "0",
            "FAKE_PYTHON_CALLS": str(calls),
            "PATH": f"{bin_dir}:{env['PATH']}",
            "VIRTUAL_ENV": str(tmp_path / "venv"),
        }
    )
    return env, ticker_file, calls


def _run(
    tmp_path: Path,
    runner_env: tuple[dict[str, str], Path, Path],
    *args: str,
    quick: bool = False,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    env, ticker_file, calls_path = runner_env
    env = env.copy()
    if quick:
        env["QUICK_MODE"] = "1"
    completed = subprocess.run(
        ["/bin/bash", str(_SCRIPT), str(ticker_file), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls: list[list[str]] = []
    current: list[str] | None = None
    if calls_path.exists():
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            if line == "CALL":
                current = []
            elif line == "END":
                assert current is not None
                calls.append(current)
                current = None
            elif current is not None:
                assert line.startswith("ARG=")
                current.append(line.removeprefix("ARG="))
    return completed, calls


def _analysis_call(calls: list[list[str]]) -> list[str]:
    return next(call for call in calls if call[:2] == ["-m", "src.main"])


def test_forwards_analyzer_arguments_verbatim_and_keeps_defaults(
    tmp_path: Path, runner_env: tuple[dict[str, str], Path, Path]
) -> None:
    completed, calls = _run(
        tmp_path,
        runner_env,
        "--",
        "--debate-reasoning-handoffs",
        "--deep-model",
        "model with spaces",
    )

    assert completed.returncode == 0, completed.stderr
    analysis = _analysis_call(calls)
    assert "--quiet" in analysis
    assert "--brief" in analysis
    assert analysis[-3:] == [
        "--debate-reasoning-handoffs",
        "--deep-model",
        "model with spaces",
    ]
    assert "--debate-reasoning-handoffs" in analysis


@pytest.mark.parametrize("logging_arg", ["--debug", "--verbose", "--brief", "--quiet"])
def test_explicit_logging_mode_replaces_quiet_brief_defaults(
    tmp_path: Path,
    runner_env: tuple[dict[str, str], Path, Path],
    logging_arg: str,
) -> None:
    completed, calls = _run(tmp_path, runner_env, "--", logging_arg)

    assert completed.returncode == 0, completed.stderr
    analysis = _analysis_call(calls)
    assert analysis.count(logging_arg) == 1
    for other in {"--debug", "--verbose", "--brief", "--quiet"} - {logging_arg}:
        assert other not in analysis


@pytest.mark.parametrize(
    "owned_args",
    [
        ("--ticker", "OTHER"),
        ("--output=elsewhere.md",),
        ("--imagedir", "elsewhere"),
        ("--quick",),
    ],
)
def test_rejects_arguments_owned_by_the_runner_before_analysis(
    tmp_path: Path,
    runner_env: tuple[dict[str, str], Path, Path],
    owned_args: tuple[str, ...],
) -> None:
    completed, calls = _run(tmp_path, runner_env, "--", *owned_args)

    assert completed.returncode == 2
    assert "owned by this runner" in completed.stderr
    assert not calls


def test_quick_handoff_conflict_fails_before_analysis(
    tmp_path: Path, runner_env: tuple[dict[str, str], Path, Path]
) -> None:
    completed, calls = _run(
        tmp_path,
        runner_env,
        "--",
        "--debate-reasoning-handoffs",
        quick=True,
    )

    assert completed.returncode == 2
    assert "requires a full two-round run" in completed.stderr
    assert not calls


def test_rejects_conflicting_logging_modes(
    tmp_path: Path, runner_env: tuple[dict[str, str], Path, Path]
) -> None:
    completed, calls = _run(tmp_path, runner_env, "--", "--debug", "--brief")

    assert completed.returncode == 2
    assert "at most one logging mode" in completed.stderr
    assert not calls


def test_requires_separator_before_forwarded_arguments(
    tmp_path: Path, runner_env: tuple[dict[str, str], Path, Path]
) -> None:
    completed, calls = _run(tmp_path, runner_env, "--debug")

    assert completed.returncode == 2
    assert "put analyzer arguments after --" in completed.stderr
    assert not calls
