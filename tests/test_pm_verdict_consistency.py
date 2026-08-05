"""Keep PM verdict normalization centralized."""

from __future__ import annotations

import ast
from pathlib import Path

VERDICT_CONSUMERS = {
    "src/report_generator.py",
    "src/reporting/memo.py",
    "src/charts/extractors/pm_block.py",
}


def test_verdict_consumers_use_shared_canonicalizer() -> None:
    missing_imports = []
    for rel in sorted(VERDICT_CONSUMERS):
        source = Path(rel).read_text()
        tree = ast.parse(source)
        imports_canonicalizer = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "src.pm_decision_parser"
            and any(alias.name == "canonicalize_pm_verdict" for alias in node.names)
            for node in ast.walk(tree)
        )
        if not imports_canonicalizer:
            missing_imports.append(rel)

    assert not missing_imports, (
        "PM verdict consumers must import canonicalize_pm_verdict from the neutral "
        f"src.pm_decision_parser (agents-free, breaks the pm_block cycle): {missing_imports}"
    )


def test_no_local_pm_verdict_alias_maps_or_normalizers() -> None:
    violations = []
    for rel in sorted(VERDICT_CONSUMERS):
        source = Path(rel).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_normalize_verdict":
                violations.append(f"{rel}:{node.lineno} defines _normalize_verdict")
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name)
                and target.id in {"VERDICT_NORMALIZATION", "_VERDICT_NORMALIZATION"}
                for target in node.targets
            ):
                violations.append(f"{rel}:{node.lineno} defines verdict alias map")

    assert (
        not violations
    ), f"Use src.pm_decision_parser.canonicalize_pm_verdict instead: {violations}"
