import pytest
from llm_reliability.client import LLMClient
from llm_reliability.models import CompletionRequest
from llm_reliability.errors import PermanentError


@pytest.mark.asyncio
async def test_openai_provider():
    """Test that OpenAI provider works through the neutral client"""
    client = LLMClient(provider="openai")

    request = CompletionRequest(
        prompt="Say hello in one short sentence.",
        max_tokens=40,
        temperature=0.0
    )

    response = await client.complete(request)

    assert response.content is not None
    assert len(response.content) > 0
    assert response.provider == "openai"
    print("\n[OpenAI] →", response.content)


@pytest.mark.asyncio
async def test_anthropic_provider():
    """Test that Anthropic provider works through the neutral client"""
    client = LLMClient(provider="anthropic")

    request = CompletionRequest(
        prompt="Say hello in one short sentence.",
        max_tokens=40,
        temperature=0.0
    )

    response = await client.complete(request)

    assert response.content is not None
    assert len(response.content) > 0
    assert response.provider == "anthropic"
    print("\n[Anthropic] →", response.content)


@pytest.mark.asyncio
async def test_unsupported_provider():
    """Should raise a clean PermanentError for unknown provider"""
    with pytest.raises(PermanentError):
        LLMClient(provider="unknown-provider")
