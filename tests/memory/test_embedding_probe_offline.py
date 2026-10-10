"""A unit test that builds a memory must not reach the embeddings provider.

``FinancialSituationMemory`` probes its embeddings client at construction. The
session's dummy ``GOOGLE_API_KEY`` is enough to build a real client, so without
``_embedding_provider_unreachable`` in ``tests/conftest.py`` the probe sent a real
``embed_content`` request to Google from every unit test that built a memory
unpatched. Proven by planting: with that fixture removed, the first test below
records a network attempt.
"""

from __future__ import annotations

import pytest

from src.memory import FinancialSituationMemory
from tests.conftest import EmbeddingProviderUnreachable
from tests.helpers.offline_network import install_offline_network


@pytest.fixture
def fresh_shared_embeddings():
    FinancialSituationMemory._reset_shared_state_for_tests()
    yield
    FinancialSituationMemory._reset_shared_state_for_tests()


def test_constructing_a_memory_sends_no_probe_request(
    monkeypatch, fresh_shared_embeddings
):
    attempts = install_offline_network(monkeypatch)

    memory = FinancialSituationMemory("probe_offline_guard")

    assert attempts == []
    # The probe ran against the real client and failed as an unreachable
    # provider would: the client is kept, marked unavailable.
    assert memory.embeddings is not None
    assert memory.embeddings_available is False
    assert memory.available is False


def test_every_embedding_sdk_request_is_refused_before_sending(monkeypatch):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_openai import OpenAIEmbeddings

    attempts = install_offline_network(monkeypatch)
    google = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key="test-dummy"
    )
    openai = OpenAIEmbeddings(model="text-embedding-3-small", api_key="sk-dummy")

    for client in (google, openai):
        with pytest.raises(EmbeddingProviderUnreachable):
            client.embed_query("x")
        with pytest.raises(EmbeddingProviderUnreachable):
            client.embed_documents(["x"])
    assert attempts == []


@pytest.mark.asyncio
async def test_async_embedding_sdk_requests_are_refused_too():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    client = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key="test-dummy"
    )
    with pytest.raises(EmbeddingProviderUnreachable):
        await client.aembed_query("x")
    with pytest.raises(EmbeddingProviderUnreachable):
        await client.aembed_documents(["x"])
