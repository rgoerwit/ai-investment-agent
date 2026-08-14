from unittest.mock import MagicMock

from scripts.embedding_collections import initialize_current, inspect_collections


def test_inspect_collection_has_no_mutating_calls() -> None:
    collection = MagicMock(name="collection")
    collection.name = "legacy"
    collection.count.return_value = 3
    collection.metadata = {"embedding_model": "old"}
    client = MagicMock()
    client.list_collections.return_value = [collection]
    assert inspect_collections(client)[0]["count"] == 3
    client.delete_collection.assert_not_called()
    client.get_or_create_collection.assert_not_called()


def test_initialize_current_creates_only_fingerprinted_target(monkeypatch) -> None:
    from scripts import embedding_collections as module
    from src.config import Settings

    monkeypatch.setattr(
        module,
        "config",
        Settings(_env_file=None, google_api_key="g"),
    )
    client = MagicMock()
    name = initialize_current(client, "lessons_learned")
    assert name.startswith("lessons_learned__emb_")
    client.delete_collection.assert_not_called()
    assert client.get_or_create_collection.call_args.kwargs["name"] == name
