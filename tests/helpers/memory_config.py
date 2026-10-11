"""A mocked memory config must carry the real embedding identity."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.config import Settings

_IDENTITY_FIELDS = (
    "embedding_provider",
    "embedding_model",
    "embedding_dimension",
    "embedding_schema_version",
)


def pin_embedding_identity(mock_config: MagicMock) -> MagicMock:
    """Copy the shipped embedding identity onto a mock config.

    ``resolve_embedding_binding`` reads these fields plainly (it used to fall
    back to literals for anything that was not a str or int, which hid a mock
    here), so a ``MagicMock`` config must be given them.
    """
    real = Settings(_env_file=None)
    for field in _IDENTITY_FIELDS:
        setattr(mock_config, field, getattr(real, field))
    return mock_config
