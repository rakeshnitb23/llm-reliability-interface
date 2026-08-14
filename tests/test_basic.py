import pytest
from llm_reliability.client import LLMClient
from llm_reliability.models import CompletionRequest
from llm_reliability.errors import PermanentError


@pytest.mark.asyncio
async def test_basic_completion():
    """
    Slice 1 test:
    - Can we successfully call OpenAI?
    - Do we get back a clean CompletionResponse?
    """
    client = LLMClient()  # needs OPENAI_API_KEY in environment

    request = CompletionRequest(
        prompt="Say hello in one short sentence.",
        max_tokens=30,
        temperature=0.0
    )

    response = await client.complete(request)

    # Basic assertions
    assert response.content is not None
    assert len(response.content) > 0
    assert response.provider == "openai"
    assert response.model is not None
    print("\nResponse content:", response.content)


@pytest.mark.asyncio
async def test_missing_api_key():
    """Check that we raise a clean PermanentError when API key is missing"""
    with pytest.raises(PermanentError):
        client = LLMClient(api_key="")  # force empty key
