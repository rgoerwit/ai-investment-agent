from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import check_claude_metadata

ROOT = Path(__file__).resolve().parents[2]


def _skill(tmp_path: Path, body: str, *, name: str = "sample") -> Path:
    skill = tmp_path / ".claude" / "skills" / "sample" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        f"---\nname: {name}\ndescription: Sample skill.\n---\n{body}\n",
        encoding="utf-8",
    )
    return skill


def _rule(tmp_path: Path, body: str) -> Path:
    rule = tmp_path / ".claude" / "rules" / "sample.md"
    rule.parent.mkdir(parents=True)
    rule.write_text(body, encoding="utf-8")
    return rule


def test_public_claude_metadata_current_surface_passes() -> None:
    completed = subprocess.run(
        ["git", "ls-files", "--", ".claude/rules", ".claude/skills"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [ROOT / value for value in completed.stdout.splitlines()]
    errors = check_claude_metadata.validate_surface_inventory()
    errors.extend(
        error
        for path in paths
        if path.is_file() or path.is_symlink()
        for error in check_claude_metadata.validate_file(path)
    )
    assert errors == []


def test_claude_metadata_rejects_cross_tool_reference(tmp_path, monkeypatch) -> None:
    skill = _skill(tmp_path, "Read `.agents/skills/example/SKILL.md`.")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(skill)
    assert any("another coding tool" in error for error in errors)


def test_claude_metadata_rejects_machine_local_path(tmp_path, monkeypatch) -> None:
    machine_path = "/" + "Users/example/private-tool"
    rule = _rule(tmp_path, f"Run `{machine_path}`.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("machine-local" in error for error in errors)


def test_claude_metadata_rejects_environment_assignment(tmp_path, monkeypatch) -> None:
    rule = _rule(tmp_path, "SECRET_TOKEN=not-a-real-secret\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("environment-style assignment" in error for error in errors)


def test_claude_metadata_rejects_environment_mapping(tmp_path, monkeypatch) -> None:
    rule = _rule(tmp_path, '"SECRET_TOKEN": "not-a-real-secret"\n')
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("environment-style assignment" in error for error in errors)


def test_claude_metadata_allows_uppercase_prose_label(tmp_path, monkeypatch) -> None:
    rule = _rule(tmp_path, "NOTE: ordinary prose is not configuration.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    assert check_claude_metadata.validate_file(rule) == []


def test_claude_metadata_allows_symbolic_slash_prose(tmp_path, monkeypatch) -> None:
    rule = _rule(tmp_path, "Use `N/A` when the criterion does not apply.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    assert check_claude_metadata.validate_file(rule) == []


def test_claude_metadata_allows_public_environment_filenames(
    tmp_path, monkeypatch
) -> None:
    rule = _rule(tmp_path, "Copy `.env.example` to `.env`.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    assert check_claude_metadata.validate_file(rule) == []


def test_claude_metadata_rejects_private_environment_filename(
    tmp_path, monkeypatch
) -> None:
    private_name = ".env" + ".private"
    rule = _rule(tmp_path, f"Do not cite `{private_name}`.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("private environment-file variant" in error for error in errors)


def test_claude_metadata_rejects_missing_path(tmp_path, monkeypatch) -> None:
    rule = _rule(tmp_path, "Read `docs/MISSING.md`.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("missing repository path" in error for error in errors)


def test_claude_metadata_rejects_missing_path_inside_command(
    tmp_path, monkeypatch
) -> None:
    rule = _rule(tmp_path, "Run `poetry run pytest tests/MISSING.py -v`.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("missing repository path" in error for error in errors)


def test_claude_metadata_allows_tracked_relative_import(tmp_path, monkeypatch) -> None:
    rule = _rule(tmp_path, "Read @guide.md before editing.\n")
    guide = rule.parent / "guide.md"
    guide.write_text("Public guidance.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", ".claude"], cwd=tmp_path, check=True)
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    assert check_claude_metadata.validate_file(rule) == []


def test_claude_metadata_rejects_missing_active_import(tmp_path, monkeypatch) -> None:
    rule = _rule(tmp_path, "Read @missing.md before editing.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("imports a missing" in error for error in errors)


def test_claude_metadata_rejects_import_outside_repository(
    tmp_path, monkeypatch
) -> None:
    rule = _rule(tmp_path, "Read @../../../../outside.md before editing.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("outside the repository" in error for error in errors)


def test_claude_metadata_ignores_imports_in_code(tmp_path, monkeypatch) -> None:
    rule = _rule(
        tmp_path,
        "Use `@missing.md` as a literal.\n```md\n@also-missing.md\n```\n",
    )
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    assert check_claude_metadata.validate_file(rule) == []


def test_indexed_claude_metadata_rejects_untracked_reference(
    tmp_path, monkeypatch
) -> None:
    rule = _rule(tmp_path, "Read `docs/local.md`.\n")
    target = tmp_path / "docs" / "local.md"
    target.parent.mkdir()
    target.write_text("Local only.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", ".claude"], cwd=tmp_path, check=True)
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)

    errors = check_claude_metadata.validate_file(rule)
    assert any("untracked repository path" in error for error in errors)

    subprocess.run(["git", "add", "docs/local.md"], cwd=tmp_path, check=True)
    assert check_claude_metadata.validate_file(rule) == []


def test_public_claude_metadata_rejects_symlink(tmp_path, monkeypatch) -> None:
    target = tmp_path / "real.md"
    target.write_text("Public text.\n", encoding="utf-8")
    skill = tmp_path / ".claude" / "skills" / "sample" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.symlink_to(target)
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(skill)
    assert any("must not be a symlink" in error for error in errors)


def test_public_claude_metadata_rejects_non_utf8(tmp_path, monkeypatch) -> None:
    rule = tmp_path / ".claude" / "rules" / "sample.md"
    rule.parent.mkdir(parents=True)
    rule.write_bytes(b"\xff")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("must be UTF-8" in error for error in errors)


def test_claude_skill_name_must_match_directory(tmp_path, monkeypatch) -> None:
    skill = _skill(tmp_path, "Do the work.", name="different")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(skill)
    assert any("skill name must equal" in error for error in errors)


def test_claude_skill_requires_description(tmp_path, monkeypatch) -> None:
    skill = tmp_path / ".claude" / "skills" / "sample" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: sample\n---\nDo the work.\n", encoding="utf-8")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(skill)
    assert any("description is required" in error for error in errors)


def test_claude_rule_paths_must_be_string_list(tmp_path, monkeypatch) -> None:
    rule = _rule(tmp_path, "---\npaths: '*.py'\n---\nDo the work.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("rule paths must be" in error for error in errors)


def test_claude_metadata_rejects_invalid_yaml(tmp_path, monkeypatch) -> None:
    rule = _rule(tmp_path, "---\npaths: [\n---\nDo the work.\n")
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_file(rule)
    assert any("invalid YAML frontmatter" in error for error in errors)


def test_claude_inventory_rejects_project_settings(tmp_path, monkeypatch) -> None:
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "add", "-f", ".claude/settings.json"], cwd=tmp_path, check=True
    )
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_surface_inventory()
    assert any("unexpected tracked file" in error for error in errors)


def test_claude_guard_does_not_read_unexpected_settings(tmp_path, monkeypatch) -> None:
    settings = tmp_path / ".claude" / "settings.json"
    settings.parent.mkdir(parents=True)
    settings.write_bytes(b"\xff")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "add", "-f", ".claude/settings.json"], cwd=tmp_path, check=True
    )
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    assert check_claude_metadata.main(["check_claude_metadata.py", str(settings)]) == 1


def test_claude_inventory_rejects_synced_skill(tmp_path, monkeypatch) -> None:
    synced = tmp_path / ".claude" / "skills" / "synced" / "sample" / "SKILL.md"
    synced.parent.mkdir(parents=True)
    synced.write_text("---\nname: sample\ndescription: x\n---\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-f", ".claude"], cwd=tmp_path, check=True)
    monkeypatch.setattr(check_claude_metadata, "ROOT", tmp_path)
    errors = check_claude_metadata.validate_surface_inventory()
    assert any("unexpected tracked file" in error for error in errors)
