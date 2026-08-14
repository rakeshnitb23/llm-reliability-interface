"""
Contract tests for the reliability layer.

These tests do NOT call a real provider. They mock at the OpenAI SDK
boundary (`AsyncOpenAI`) and verify the promises our own code makes:

  1. TransientError -> retried, and eventually succeeds/fails cleanly.
  2. Malformed structured output -> SchemaError, never a bad object, never retried.
  3. A stream can be cancelled mid-flight without an unhandled exception,
     and the underlying provider stream is actually closed (no leaked connection).

We are testing OUR client's behavior under provider failure, not whether
OpenAI itself works.
"""

import contextlib
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from pydantic import BaseModel

from llm_reliability.client import LLMClient
from llm_reliability.errors import SchemaError, TransientError
from llm_reliability.models import CompletionRequest


def _rate_limit_error() -> openai.RateLimitError:
    """Build a real openai.RateLimitError (status_code=429) to raise from the mock."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(429, request=request, json={"error": {"message": "rate limited"}})
    return openai.RateLimitError("rate limited", response=response, body=None)


def _make_client() -> LLMClient:
    # api_key passed explicitly so no real OPENAI_API_KEY is needed.
    return LLMClient(provider="openai", api_key="test-key", max_retries=3, base_delay=0.01)


class _FakeChatCompletion:
    """Minimal stand-in for openai.types.chat.ChatCompletion."""

    def __init__(self, content: str):
        self.model = "gpt-4o-mini"
        self.usage = None
        self.choices = [type("Choice", (), {"message": type("Msg", (), {"content": content})()})]


# ---------------------------------------------------------------------------
# 1. Throttling / rate limit -> TransientError -> retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limit_is_retried_and_eventually_succeeds():
    """
    Contract: a rate-limit response from the provider must be treated as
    TransientError and retried by the client, not surfaced immediately.
    """
    client = _make_client()

    mock_create = AsyncMock(
        side_effect=[
            _rate_limit_error(),
            _rate_limit_error(),
            _FakeChatCompletion("hello there"),
        ]
    )

    with patch.object(client._provider._client.chat.completions, "create", mock_create), \
         patch("llm_reliability.client.asyncio.sleep", new=AsyncMock()) as mock_sleep:

        response = await client.complete(
            CompletionRequest(prompt="hi", max_tokens=10)
        )

    assert response.content == "hello there"
    assert mock_create.call_count == 3          # 2 failures + 1 success
    assert mock_sleep.call_count == 2            # backoff before retry #2 and #3


@pytest.mark.asyncio
async def test_rate_limit_gives_up_after_max_retries():
    """
    Contract: retries are bounded. After max_retries is exhausted, the
    client must surface TransientError rather than retrying forever.
    """
    client = _make_client()  # max_retries=3

    mock_create = AsyncMock(side_effect=lambda **_: (_ for _ in ()).throw(_rate_limit_error()))

    with patch.object(client._provider._client.chat.completions, "create", mock_create), \
         patch("llm_reliability.client.asyncio.sleep", new=AsyncMock()):

        with pytest.raises(TransientError):
            await client.complete(CompletionRequest(prompt="hi", max_tokens=10))

    # 1 initial attempt + 3 retries = 4 calls
    assert mock_create.call_count == 4


# ---------------------------------------------------------------------------
# 2. Malformed structured output -> SchemaError
# ---------------------------------------------------------------------------

class _Person(BaseModel):
    name: str
    age: int


@pytest.mark.asyncio
async def test_malformed_structured_output_raises_schema_error_and_no_data_returned():
    """
    Contract: if the model doesn't return a value matching the requested
    Pydantic schema, we must raise SchemaError and never hand back a
    partially-valid or bad object.
    """
    client = _make_client()

    # Simulate the SDK failing to parse the model's output into our schema.
    fake_completion = type(
        "FakeCompletion",
        (),
        {"choices": [type("Choice", (), {"message": type("Msg", (), {"parsed": None})()})]},
    )()
    mock_parse = AsyncMock(return_value=fake_completion)

    with patch.object(client._provider._client.chat.completions, "parse", mock_parse):
        with pytest.raises(SchemaError):
            await client.complete_structured(
                CompletionRequest(prompt="give me a person", max_tokens=50),
                _Person,
            )

    # SchemaError is not a TransientError -> must NOT be retried.
    assert mock_parse.call_count == 1


# ---------------------------------------------------------------------------
# 3. Stream cancellation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_can_be_cancelled_early_without_crashing():
    """
    Contract: a caller can stop consuming `client.stream(...)` partway
    through (e.g. user cancels) without an unhandled exception, and the
    underlying provider stream is actually torn down (no leaked connection).
    """
    client = _make_client()

    closed = {"value": False}

    class _FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            self._i = getattr(self, "_i", 0) + 1
            chunk = type(
                "Chunk", (), {"choices": [type("Choice", (), {"delta": type("Delta", (), {"content": f"tok{self._i}"})()})]}
            )
            return chunk

        async def close(self):
            closed["value"] = True

    mock_create = AsyncMock(return_value=_FakeStream())

    tokens = []
    with patch.object(client._provider._client.chat.completions, "create", mock_create):
        async with contextlib.aclosing(
            client.stream(CompletionRequest(prompt="hi", max_tokens=100))
        ) as agen:
            async for token in agen:
                tokens.append(token)
                if len(tokens) >= 3:
                    break  # simulate caller cancelling early

    assert tokens == ["tok1", "tok2", "tok3"]
    assert closed["value"] is True  # provider-side stream was cleanly closed, not leaked
