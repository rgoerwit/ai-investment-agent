"""Permanent boundary preventing provider construction from spreading."""

import ast
import subprocess
import sys
from pathlib import Path

_PROVIDER_CONSTRUCTORS = {
    "ChatAnthropic",
    "ChatGoogleGenerativeAI",
    "ChatOpenAI",
}

# The compatibility facade is removed after the legacy configuration window.
_TEMPORARY_ALLOWED_FILES = {
    Path("src/llms.py"),
}
_PERMANENT_ALLOWED_ROOT = Path("src/llm_runtime/adapters")


def _constructor_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        for alias in node.names:
            if alias.name in _PROVIDER_CONSTRUCTORS:
                found.append((node.lineno, alias.name))
    return found


def test_provider_constructors_are_confined() -> None:
    violations: list[str] = []
    for path in Path("src").rglob("*.py"):
        imports = _constructor_imports(path)
        if not imports:
            continue
        if path in _TEMPORARY_ALLOWED_FILES or path.is_relative_to(
            _PERMANENT_ALLOWED_ROOT
        ):
            continue
        violations.extend(f"{path}:{line}: {name}" for line, name in imports)
    assert violations == []


def test_temporary_constructor_allowlist_matches_current_debt() -> None:
    actual = {path for path in _TEMPORARY_ALLOWED_FILES if _constructor_imports(path)}
    assert actual == _TEMPORARY_ALLOWED_FILES


def test_provider_neutral_runtime_control_plane_does_not_import_google_sdk() -> None:
    script = (
        "import sys; import src.main; "
        "assert 'src.llms' not in sys.modules; "
        "assert 'langchain_google_genai' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        close_fds=False,
    )
    assert result.returncode == 0, result.stderr
