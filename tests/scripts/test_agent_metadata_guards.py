from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import check_agent_metadata

ROOT = Path(__file__).resolve().parents[2]


def test_public_metadata_current_surface_passes() -> None:
    paths = [ROOT / "AGENTS.md", *sorted((ROOT / ".agents").rglob("*"))]
    errors = [
        error
        for path in paths
        if path.is_file() or path.is_symlink()
        for error in check_agent_metadata.validate_file(path)
    ]
    assert errors == []


def test_skill_rejects_cross_tool_metadata(tmp_path, monkeypatch) -> None:
    skill = tmp_path / ".agents" / "skills" / "sample" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: sample\ndescription: Sample skill.\n---\nRead `.claude/rules/x.md`.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(skill)
    assert any("another tool's metadata" in error for error in errors)


def test_metadata_rejects_machine_local_path(tmp_path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    machine_path = "/" + "Users/example/private-tool"
    agents.write_text(f"Run `{machine_path}`.\n", encoding="utf-8")
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(agents)
    assert any("machine-local" in error for error in errors)


def test_metadata_rejects_environment_assignment(tmp_path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("SECRET_TOKEN=not-a-real-secret\n", encoding="utf-8")
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(agents)
    assert any("environment-style assignment" in error for error in errors)


def test_metadata_rejects_environment_mapping_value(tmp_path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text('"SECRET_TOKEN": "not-a-real-secret"\n', encoding="utf-8")
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(agents)
    assert any("environment-style assignment" in error for error in errors)


def test_metadata_allows_uppercase_prose_label(tmp_path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("NOTE: ordinary prose is not configuration.\n", encoding="utf-8")
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    assert check_agent_metadata.validate_file(agents) == []


def test_metadata_rejects_exported_environment_assignment(
    tmp_path, monkeypatch
) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("export SECRET_TOKEN=not-a-real-secret\n", encoding="utf-8")
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(agents)
    assert any("environment-style assignment" in error for error in errors)


def test_tracked_deps_allows_public_environment_filenames(tmp_path) -> None:
    metadata = tmp_path / "metadata.md"
    metadata.write_text("Copy `.env.example` to `.env`.\n", encoding="utf-8")
    completed = subprocess.run(
        ["bash", "scripts/check_tracked_deps.sh", str(metadata)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0


def test_tracked_deps_rejects_private_environment_filename(tmp_path) -> None:
    metadata = tmp_path / "metadata.md"
    private_name = ".env" + ".private"
    metadata.write_text(f"Do not cite `{private_name}`.\n", encoding="utf-8")
    completed = subprocess.run(
        ["bash", "scripts/check_tracked_deps.sh", str(metadata)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "unobtainable path" in completed.stderr


def test_tracked_deps_requires_example_sibling_to_be_tracked(tmp_path) -> None:
    metadata = tmp_path / "metadata.md"
    example = tmp_path / "private.example.txt"
    metadata.write_text("Read `private.txt`.\n", encoding="utf-8")
    example.write_text("Public template.\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("private.txt\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    command = ["bash", str(ROOT / "scripts/check_tracked_deps.sh"), str(metadata)]
    untracked = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert untracked.returncode == 1
    assert "unobtainable path" in untracked.stderr

    subprocess.run(
        ["git", "add", ".gitignore", "metadata.md", "private.example.txt"],
        cwd=tmp_path,
        check=True,
    )
    tracked = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert tracked.returncode == 0


def test_metadata_rejects_private_environment_filename(tmp_path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    private_name = ".env" + ".private"
    agents.write_text(f"Do not cite `{private_name}`.\n", encoding="utf-8")
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(agents)
    assert any("private environment-file variant" in error for error in errors)


def test_metadata_rejects_missing_backticked_repo_path(tmp_path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("Read `docs/MISSING.md`.\n", encoding="utf-8")
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(agents)
    assert any("missing repository path" in error for error in errors)


def test_metadata_rejects_missing_path_inside_command(tmp_path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text(
        "Run `poetry run pytest tests/MISSING.py -v`.\n", encoding="utf-8"
    )
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(agents)
    assert any("missing repository path" in error for error in errors)


def test_metadata_rejects_missing_path_inside_fenced_command(
    tmp_path, monkeypatch
) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text(
        "```bash\npoetry run pytest tests/MISSING.py -v\n```\n", encoding="utf-8"
    )
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(agents)
    assert any("missing repository path" in error for error in errors)


def test_indexed_metadata_rejects_untracked_reference(tmp_path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    target = tmp_path / "docs" / "local.md"
    target.parent.mkdir()
    agents.write_text("Read `docs/local.md`.\n", encoding="utf-8")
    target.write_text("Local only.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "AGENTS.md"], cwd=tmp_path, check=True)
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)

    errors = check_agent_metadata.validate_file(agents)
    assert any("untracked repository path" in error for error in errors)

    subprocess.run(["git", "add", "docs/local.md"], cwd=tmp_path, check=True)
    assert check_agent_metadata.validate_file(agents) == []


def test_root_guidance_rejects_ignored_tool_instruction_file(
    tmp_path, monkeypatch
) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("Never edit `CLAUDE.md`.\n", encoding="utf-8")
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(agents)
    assert any("ignored tool-specific instruction file" in error for error in errors)


def test_public_metadata_rejects_symlink(tmp_path, monkeypatch) -> None:
    skill = tmp_path / ".agents" / "skills" / "sample" / "SKILL.md"
    target = tmp_path / "real.md"
    skill.parent.mkdir(parents=True)
    target.write_text("Public text.\n", encoding="utf-8")
    skill.symlink_to(target)
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(skill)
    assert any("must not be a symlink" in error for error in errors)


def test_public_metadata_rejects_non_utf8_text(tmp_path, monkeypatch) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_bytes(b"\xff")
    monkeypatch.setattr(check_agent_metadata, "ROOT", tmp_path)
    errors = check_agent_metadata.validate_file(agents)
    assert any("must be UTF-8" in error for error in errors)


def test_doc_layering_skips_only_repository_root_agents_file(tmp_path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "docs" / "AGENTS.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("Read `.agents/skills/example/SKILL.md`.\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    completed = subprocess.run(
        ["bash", str(ROOT / "scripts/check_doc_layering.sh"), str(nested)],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "documentation cites agent metadata" in completed.stderr


def test_agents_size_guard_enforces_policy(tmp_path) -> None:
    agents = tmp_path / "AGENTS.md"
    agents.write_text("x" * 13, encoding="utf-8")
    completed = subprocess.run(
        ["bash", "scripts/check_agents_size.sh", str(agents), "12"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "repository policy caps" in completed.stderr


def test_agents_size_guard_requires_file(tmp_path) -> None:
    missing = tmp_path / "AGENTS.md"
    completed = subprocess.run(
        ["bash", "scripts/check_agents_size.sh", str(missing), "12"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "public Codex contract is required" in completed.stderr


def test_root_agents_ignore_matrix() -> None:
    public = subprocess.run(
        ["git", "check-ignore", "-q", "AGENTS.md"], cwd=ROOT, check=False
    )
    override = subprocess.run(
        ["git", "check-ignore", "-q", "AGENTS.override.md"],
        cwd=ROOT,
        check=False,
    )
    archived = subprocess.run(
        ["git", "check-ignore", "-q", "AGENTS-2000.md"],
        cwd=ROOT,
        check=False,
    )
    assert public.returncode == 1
    assert override.returncode == 0
    assert archived.returncode == 0
