"""Provider-selectable embedding construction and non-destructive identity."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit

from src.config import Settings, config

_OPENAI_EMBEDDINGS_HOST = "api.openai.com"


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


def GoogleGenerativeAIEmbeddings(**kwargs: Any) -> Any:
    """The Google embeddings client; the SDK import stays lazy. Tests patch here."""

    from langchain_google_genai import GoogleGenerativeAIEmbeddings as implementation

    return implementation(**kwargs)


def embedding_endpoint_host(
    binding: EmbeddingBinding, settings: Settings = config
) -> str | None:
    """The host an embedding request is sent to, when it is not the SDK default.

    Keys the rate limiter (``limiter_for_binding``) and decides the OpenAI base
    URL, so the two cannot disagree. ``OPENAI_API_BASE`` selects the *review
    chat plane's* compatible vendor (Moonshot/Kimi and friends). Those vendors
    serve no embeddings API, so inheriting the base silently pointed every
    memory read and write at an endpoint that cannot answer, and memory
    failures degrade quietly. Only an explicit OpenAI base is honoured.
    """
    if binding.provider != "openai":
        return None
    host = urlsplit(settings.get_openai_api_base() or "").hostname
    return host if host == _OPENAI_EMBEDDINGS_HOST else None


def build_embeddings(
    binding: EmbeddingBinding,
    settings: Settings = config,
) -> Any:
    """Build an embedding client with lazy provider imports."""

    if binding.provider == "google":
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
            # One SDK attempt per request: the memory embedding retry owns
            # retries, so the SDK's default two would multiply its attempts.
            "max_retries": 0,
        }
        if binding.dimension is not None:
            kwargs["dimensions"] = binding.dimension
        if embedding_endpoint_host(binding, settings) is not None:
            kwargs["base_url"] = settings.get_openai_api_base()
        return OpenAIEmbeddings(**kwargs)
    raise ValueError(f"unsupported embedding provider: {binding.provider!r}")


def embedding_credential(settings: Settings, provider: str) -> str:
    if provider == "google":
        return settings.get_google_api_key()
    if provider == "openai":
        return settings.get_openai_api_key()
    return ""
