"""
Enforce that all src/ modules use structlog, not stdlib logging.

These static AST checks prevent future regressions where a developer
accidentally reverts to logging.getLogger() or uses f-strings as log
message arguments (which defeat lazy evaluation).
"""

import ast
from pathlib import Path

# Files that intentionally use logging.getLogger() for legitimate reasons:
#   src/config.py          — bootstrap: calls logging.basicConfig() + structlog.configure()
#   src/main.py            — silences noisy third-party libs (aiohttp, httpx, etc.)
#   src/health_check.py    — silences third-party libs; standalone script run before structlog
#   src/report_generator.py — silences third-party libs in quiet-mode output function
STDLIB_LOGGING_ALLOWED = {
    "src/config.py",
    "src/main.py",
    "src/health_check.py",
    "src/report_generator.py",
}

# Files converted to structlog in the logging-consistency migration (Mar 2026).
# The f-string check is scoped to these files; pre-existing violations in the
# broader codebase are tracked as tech-debt to clean up in future PRs.
STRUCTLOG_CONVERTED = {
    "src/data/fmp_fetcher.py",
    "src/data/eodhd_fetcher.py",
    "src/llms.py",
    "src/report_generator.py",
    "src/health_check.py",
}


def test_no_stdlib_logger_in_src():
    """No module outside the whitelist should call logging.getLogger()."""
    violations = []
    for py_file in Path("src").rglob("*.py"):
        rel = str(py_file.as_posix())
        if rel in STDLIB_LOGGING_ALLOWED:
            continue
        source = py_file.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "getLogger"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logging"
            ):
                violations.append(f"{rel}:{node.lineno}")
    assert not violations, f"stdlib logging.getLogger() found in: {violations}"


def test_no_fstrings_in_log_calls():
    """Converted structlog modules must not use f-strings as the first positional arg.

    Scoped to STRUCTLOG_CONVERTED files; pre-existing violations in the broader
    codebase are tech-debt to be cleaned up in future PRs.
    """
    log_methods = {"debug", "info", "warning", "error", "critical", "exception"}
    violations = []
    for py_file in Path("src").rglob("*.py"):
        rel = str(py_file.as_posix())
        if rel not in STRUCTLOG_CONVERTED:
            continue
        source = py_file.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in log_methods
                and node.args
                and isinstance(node.args[0], ast.JoinedStr)  # f-string
            ):
                violations.append(f"{py_file.as_posix()}:{node.lineno}")
    assert not violations, f"f-string as first arg to log call found in: {violations}"
