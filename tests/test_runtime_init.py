from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.runtime_init import initialize_runtime_environment


def _settings_for_paths(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        results_dir=tmp_path / "results",
        data_cache_dir=tmp_path / "data-cache",
        chroma_persist_directory=str(tmp_path / "chroma"),
        images_dir=tmp_path / "images",
        mcp_usage_db_path=tmp_path / "runtime" / "mcp_usage.db",
    )


def test_settings_construction_does_not_create_runtime_directories(tmp_path):
    settings = _settings_for_paths(tmp_path)

    assert not Path(settings.results_dir).exists()
    assert not Path(settings.data_cache_dir).exists()
    assert not Path(settings.chroma_persist_directory).exists()
    assert not Path(settings.images_dir).exists()
    assert not Path(settings.mcp_usage_db_path).parent.exists()


def test_initialize_runtime_environment_creates_runtime_directories(tmp_path):
    settings = _settings_for_paths(tmp_path)

    initialize_runtime_environment(settings)

    assert Path(settings.results_dir).is_dir()
    assert Path(settings.data_cache_dir).is_dir()
    assert Path(settings.chroma_persist_directory).is_dir()
    assert Path(settings.images_dir).is_dir()
    assert Path(settings.mcp_usage_db_path).parent.is_dir()
