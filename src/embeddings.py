"""Provider-selectable embedding construction and non-destructive identity."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit

from src.config import Settings, config


@dataclass(frozen=True)
class EmbeddingBinding:
    provider: str
    model: str
    dimension: int | None
    schema_version: int = 1

    def fingerprint(self) -> str:
        material = (
            f"{self.provider}:{self.model}:{self.dimension}:{self.schema_version}"
        )
        return sha256(material.encode()).hexdigest()[:12]

    def metadata(self) -> dict[str, str | int]:
        return {
            key: value
            for key, value in asdict(self).items()
            if isinstance(value, (str, int))
        }


def resolve_embedding_binding(settings: Settings = config) -> EmbeddingBinding:
    provider_value = getattr(settings, "embedding_provider", "google")
    model_value = getattr(settings, "embedding_model", "gemini-embedding-001")
    dimension_value = getattr(settings, "embedding_dimension", 768)
    schema_value = getattr(settings, "embedding_schema_version", 1)
    provider = provider_value if isinstance(provider_value, str) else "google"
    model = model_value if isinstance(model_value, str) else "gemini-embedding-001"
    dimension = dimension_value if isinstance(dimension_value, int) else 768
    schema_version = schema_value if isinstance(schema_value, int) else 1
    return EmbeddingBinding(
        provider=provider.strip().lower(),
        model=model.strip(),
        dimension=dimension,
        schema_version=schema_version,
    )


def fingerprinted_collection_name(base: str, binding: EmbeddingBinding) -> str:
    suffix = f"__emb_{binding.fingerprint()}"
    # Chroma collection names are bounded; keep the stable identity suffix.
    return f"{base[: max(1, 63 - len(suffix))]}{suffix}"


def build_embeddings(
    binding: EmbeddingBinding,
    settings: Settings = config,
) -> Any:
    """Build an embedding client with lazy provider imports."""

    if binding.provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        google_kwargs: dict[str, Any] = {
            "model": f"models/{binding.model}",
            "google_api_key": settings.get_google_api_key(),
            "task_type": "retrieval_document",
            "output_dimensionality": binding.dimension,
        }
        return GoogleGenerativeAIEmbeddings(**google_kwargs)
    if binding.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {
            "model": binding.model,
            "api_key": settings.openai_api_key,
        }
        if binding.dimension is not None:
            kwargs["dimensions"] = binding.dimension
        base_url = settings.get_openai_api_base()
        # OPENAI_API_BASE selects the *review chat plane's* compatible vendor
        # (Moonshot/Kimi and friends). Those vendors serve no embeddings API, so
        # inheriting the base here silently pointed every memory read/write at an
        # endpoint that cannot answer — and memory failures degrade quietly. Only
        # an explicit OpenAI base is honored; a compatible host is ignored.
        if base_url and urlsplit(base_url).hostname == "api.openai.com":
            kwargs["base_url"] = base_url
        return OpenAIEmbeddings(**kwargs)
    raise ValueError(f"unsupported embedding provider: {binding.provider!r}")


def embedding_credential(settings: Settings, provider: str) -> str:
    if provider == "google":
        return settings.get_google_api_key()
    if provider == "openai":
        return settings.get_openai_api_key()
    return ""
