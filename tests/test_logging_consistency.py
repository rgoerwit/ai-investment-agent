"""
Enforce that all src/ modules use structlog with consistent conventions.

Static AST checks prevent regressions where a developer:
- reverts to logging.getLogger() (stdlib),
- uses f-strings as log messages (defeats lazy evaluation / structure),
- logs raw exception text at operator-visible levels (secret-leak risk;
  use **summarize_exception(exc, operation=...) from src/error_safety.py),
- logs unredacted content previews at warning/error level
  (wrap in redact_sensitive_text), or
- uses prose sentences instead of snake_case event names.
"""

import ast
import re
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

LOG_METHODS = {"debug", "info", "warning", "error", "critical", "exception"}
OPERATOR_LEVELS = {"warning", "error", "critical"}
EVENT_NAME_PATTERN = re.compile(r"^[a-z0-9_.]+$")
PREVIEW_KWARG_PATTERN = re.compile(r"preview|prefix|snippet|excerpt")


def _iter_logger_calls(skip: set[str] | None = None):
    """Yield (relpath, call_node) for every logger.<level>() call in src/."""
    skip = skip or set()
    for py_file in sorted(Path("src").rglob("*.py")):
        rel = str(py_file.as_posix())
        if rel in skip:
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in LOG_METHODS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                yield rel, node


def test_no_stdlib_logger_in_src():
    """No module outside the whitelist should call logging.getLogger()."""
    violations = []
    for py_file in Path("src").rglob("*.py"):
        rel = str(py_file.as_posix())
        if rel in STDLIB_LOGGING_ALLOWED:
            continue
        tree = ast.parse(py_file.read_text())
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
    """No structlog call in src/ may use an f-string as the message."""
    violations = [
        f"{rel}:{node.lineno}"
        for rel, node in _iter_logger_calls(skip=STDLIB_LOGGING_ALLOWED)
        if node.args and isinstance(node.args[0], ast.JoinedStr)
    ]
    assert not violations, f"f-string as first arg to log call found in: {violations}"


def test_no_raw_error_kwarg_at_operator_levels():
    """logger.warning/error must not pass an `error=` keyword.

    Operator-visible exception context comes from
    **summarize_exception(exc, operation=...) (sanitized error_type /
    message_preview / failure_kind fields). Plain-string context belongs
    under `reason=`. Raw error=str(exc) leaks unredacted exception text.
    debug-level calls are exempt (not operator-facing).
    """
    violations = [
        f"{rel}:{node.lineno}"
        for rel, node in _iter_logger_calls()
        if node.func.attr in OPERATOR_LEVELS
        and any(kw.arg == "error" for kw in node.keywords)
    ]
    assert not violations, (
        "raw `error=` kwarg at operator-visible level (use "
        f"**summarize_exception(...) or reason=): {violations}"
    )


def test_preview_kwargs_redacted_at_operator_levels():
    """Content previews at warning/error must be wrapped in redact_sensitive_text."""

    def is_redacted(value: ast.expr) -> bool:
        if not isinstance(value, ast.Call):
            return False
        f = value.func
        name = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
        return "redact" in name

    violations = [
        f"{rel}:{node.lineno} ({kw.arg})"
        for rel, node in _iter_logger_calls()
        if node.func.attr in OPERATOR_LEVELS
        for kw in node.keywords
        if kw.arg and PREVIEW_KWARG_PATTERN.search(kw.arg) and not is_redacted(kw.value)
    ]
    assert (
        not violations
    ), f"unredacted preview kwarg at operator-visible level: {violations}"


def test_no_positional_args_after_event_name():
    """logger calls must use kwargs, not %-style positional args.

    structlog's PositionalArgumentsFormatter applies `event %= args`; an event
    name without % placeholders plus positional args raises TypeError at the
    call site (found live in valuation.py June 2026).
    """
    violations = [
        f"{rel}:{node.lineno}"
        for rel, node in _iter_logger_calls(skip=STDLIB_LOGGING_ALLOWED)
        if len(node.args) > 1
    ]
    assert not violations, f"positional args after log event (use kwargs): {violations}"


def test_log_event_names_are_snake_case():
    """First positional arg to logger calls must be a snake_case event name."""
    violations = []
    for rel, node in _iter_logger_calls(skip=STDLIB_LOGGING_ALLOWED):
        if not node.args:
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if not EVENT_NAME_PATTERN.match(arg.value):
                violations.append(f"{rel}:{node.lineno} {arg.value[:50]!r}")
    assert (
        not violations
    ), f"non-snake_case log event names (use event_name style): {violations}"
