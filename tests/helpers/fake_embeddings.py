"""A deterministic, offline embedding client for tests that need a working memory.

``FinancialSituationMemory`` is ``available`` only when its embedding probe
succeeds. Unit tests refuse the provider SDK, so a test that must exercise real
Chroma storage supplies this client in place of ``build_embeddings``: the same
text always maps to the same vector, and different texts to different vectors.
"""

from __future__ import annotations

import hashlib


class DeterministicEmbeddings:
    """Hash-seeded unit-interval vectors of a fixed dimension."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        vector: list[float] = []
        counter = 0
        while len(vector) < self.dimension:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            vector.extend(byte / 255.0 for byte in digest)
            counter += 1
        return vector[: self.dimension]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)
