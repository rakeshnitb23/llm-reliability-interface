# src/llm_reliability/providers/anthropic_provider.py

import os
from typing import AsyncGenerator, Optional, Type, TypeVar

from anthropic import AsyncAnthropic, APIError
from pydantic import BaseModel, ValidationError

from llm_reliability.models import CompletionRequest, CompletionResponse, ResponseModel
from llm_reliability.errors import TransientError, PermanentError, LLMReliabilityError, SchemaError
from .base import BaseProvider

T = TypeVar("T", bound=BaseModel)


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-5", timeout: float = 30.0):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise PermanentError("ANTHROPIC_API_KEY is not set")

        self.model = model
        self._client = AsyncAnthropic(api_key=self.api_key, timeout=timeout)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            response = await self._client.messages.create(
                model=request.model or self.model,
                max_tokens=request.max_tokens or 1024,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}],
            )

            content = response.content[0].text if response.content else None
            if not content:
                raise PermanentError("Model returned empty content")

            return CompletionResponse(
                content=content,
                model=response.model,
                provider="anthropic",
                usage={
                    "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                    "completion_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
                },
            )

        except APIError as e:
            status = getattr(e, "status_code", None)
            if status in {429, 500, 502, 503, 504}:
                raise TransientError(
                    f"Transient Anthropic error: {str(e)}") from e
            else:
                raise PermanentError(
                    f"Permanent Anthropic error: {str(e)}") from e

        except Exception as e:
            raise LLMReliabilityError(f"Unexpected error: {str(e)}") from e

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        try:
            async with self._client.messages.stream(
                model=request.model or self.model,
                max_tokens=request.max_tokens or 1024,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield text

        except APIError as e:
            status = getattr(e, "status_code", None)
            if status in {429, 500, 502, 503, 504}:
                raise TransientError(
                    f"Transient Anthropic error: {str(e)}") from e
            else:
                raise PermanentError(
                    f"Permanent Anthropic error: {str(e)}") from e

        except Exception as e:
            raise LLMReliabilityError(f"Unexpected error: {str(e)}") from e

    async def complete_structured(
        self, request: CompletionRequest, response_model: Type[T]
    ) -> T:
        try:
            response = await self._client.messages.create(
                model=request.model or self.model,
                max_tokens=request.max_tokens or 1024,
                temperature=request.temperature,
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": response_model.model_json_schema(),
                    }
                },
                messages=[{"role": "user", "content": request.prompt}],
            )

            content = response.content[0].text if response.content else None
            if not content:
                raise SchemaError("Model returned empty content")

            try:
                return response_model.model_validate_json(content)
            except ValidationError as e:
                raise SchemaError(
                    f"Structured output validation failed: {e}") from e

        except SchemaError:
            raise

        except APIError as e:
            status = getattr(e, "status_code", None)
            if status in {429, 500, 502, 503, 504}:
                raise TransientError(
                    f"Transient Anthropic error: {str(e)}") from e
            else:
                raise PermanentError(
                    f"Permanent Anthropic error: {str(e)}") from e

        except Exception as e:
            raise LLMReliabilityError(f"Unexpected error: {str(e)}") from e
