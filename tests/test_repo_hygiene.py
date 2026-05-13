from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_only_canonical_pre_commit_config_exists():
    assert (ROOT / ".pre-commit-config.yaml").is_file()
    assert not (ROOT / "pre-commit-config.yaml").exists()


def test_generated_analysis_examples_are_not_in_repo_root():
    allowed = {"README-EXAMPLE-ANALYSES.md"}
    generated_readmes = {
        path.name for path in ROOT.glob("README-*.md") if path.name not in allowed
    }
    assert generated_readmes == set()


def test_tracked_analysis_images_live_under_docs_examples():
    root_images = [
        path
        for path in (ROOT / "images").glob("*")
        if path.is_file()
        and (
            path.name.endswith("_radar.png")
            or path.name.endswith("_football_field.png")
        )
    ]
    assert root_images == []
