"""Commercial, lineage, adapter, and endpoint identity for model bindings."""

from dataclasses import dataclass, replace
from urllib.parse import urlsplit

KNOWN_ENDPOINT_VENDORS = {
    "api.openai.com": "openai",
    "api.deepseek.com": "deepseek",
    "api.z.ai": "zai",
    "api.moonshot.cn": "moonshot",
    "api.moonshot.ai": "moonshot",
    "api.x.ai": "xai",
}


@dataclass(frozen=True)
class ModelIdentity:
    vendor_id: str
    model_lineage: str
    adapter_kind: str
    endpoint_host: str | None = None

    def at_endpoint(self, base_url: str | None) -> "ModelIdentity":
        return replace(self, endpoint_host=sanitize_endpoint_host(base_url))


def sanitize_endpoint_host(base_url: str | None) -> str | None:
    """Return a lowercase hostname only, never a path, query, or credentials."""

    if not base_url:
        return None
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("provider base URL must be an absolute HTTP(S) URL")
    return parsed.hostname.lower().rstrip(".")


def vendor_for_endpoint_host(host: str | None) -> str | None:
    return KNOWN_ENDPOINT_VENDORS.get(host or "")
