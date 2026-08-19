import sys
from unittest.mock import MagicMock, patch

from src.config import Settings
from src.embeddings import (
    EmbeddingBinding,
    fingerprinted_collection_name,
    resolve_embedding_binding,
)


def test_embedding_identity_fingerprints_provider_model_dimension_and_schema() -> None:
    base = EmbeddingBinding("google", "gemini-embedding-001", 768)
    assert fingerprinted_collection_name("lessons_learned", base) != (
        fingerprinted_collection_name(
            "lessons_learned", EmbeddingBinding("openai", "text-embedding-3-small", 768)
        )
    )
    assert (
        base.fingerprint()
        != EmbeddingBinding("google", "gemini-embedding-001", 1536).fingerprint()
    )


def test_openai_binding_resolution_does_not_import_google_sdk() -> None:
    sys.modules.pop("langchain_google_genai", None)
    binding = resolve_embedding_binding(
        Settings(
            _env_file=None,
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimension=768,
        )
    )
    assert binding.provider == "openai"
    assert "langchain_google_genai" not in sys.modules


def test_memory_model_switch_never_deletes_legacy_collection(monkeypatch) -> None:
    import src.memory as memory_module

    fake_embeddings = MagicMock()
    fake_embeddings.embed_query.return_value = [0.1] * 3
    fake_client = MagicMock()
    fake_collection = MagicMock()
    fake_collection.count.return_value = 4
    fake_client.get_or_create_collection.return_value = fake_collection
    settings = Settings(
        _env_file=None,
        google_api_key="g",
        embedding_provider="google",
        embedding_model="a-new-embedding-model",
        embedding_dimension=768,
    )
    monkeypatch.setattr(memory_module, "config", settings)
    monkeypatch.setattr(
        memory_module, "GoogleGenerativeAIEmbeddings", lambda **kwargs: fake_embeddings
    )
    monkeypatch.setattr(
        memory_module.FinancialSituationMemory,
        "_get_shared_chroma_client",
        classmethod(lambda cls, name: fake_client),
    )
    memory_module.FinancialSituationMemory._reset_shared_state_for_tests()

    memory = memory_module.FinancialSituationMemory("lessons_learned")

    assert memory.available is True
    fake_client.delete_collection.assert_not_called()
    created_name = fake_client.get_or_create_collection.call_args.kwargs["name"]
    assert created_name.startswith("lessons_learned__emb_")
    assert created_name != "lessons_learned"
    memory_module.FinancialSituationMemory._reset_shared_state_for_tests()
