# src/llm_reliability/providers/openai_provider.py

from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar
import os
from typing import AsyncGenerator, Optional

from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletion

from llm_reliability.models import CompletionRequest, CompletionResponse
from llm_reliability.errors import SchemaError, TransientError, PermanentError, LLMReliabilityError
from .base import BaseProvider
from typing import Type, TypeVar
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", timeout: float = 30.0):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise PermanentError("OPENAI_API_KEY is not set")

        self.model = model
        self._client = AsyncOpenAI(api_key=self.api_key)

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        try:
            response: ChatCompletion = await self._client.chat.completions.create(
                model=request.model or self.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )

            content = response.choices[0].message.content
            if content is None:
                raise PermanentError("Model returned empty content")

            return CompletionResponse(
                content=content,
                model=response.model,
                provider="openai",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            )

        except OpenAIError as e:
            status = getattr(e, "status_code", None)
            if status in {429, 500, 502, 503, 504}:
                raise TransientError(
                    f"Transient OpenAI error: {str(e)}") from e
            else:
                raise PermanentError(
                    f"Permanent OpenAI error: {str(e)}") from e

        except Exception as e:
            raise LLMReliabilityError(f"Unexpected error: {str(e)}") from e

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        try:
            stream = await self._client.chat.completions.create(
                model=request.model or self.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=True,
            )

            try:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            finally:
                await stream.close()

        except OpenAIError as e:
            status = getattr(e, "status_code", None)
            if status in {429, 500, 502, 503, 504}:
                raise TransientError(f"Transient OpenAI error: {str(e)}") from e
            else:
                raise PermanentError(f"Permanent OpenAI error: {str(e)}") from e

        except Exception as e:
            raise LLMReliabilityError(f"Unexpected error: {str(e)}") from e

    async def complete_structured(
        self, request: CompletionRequest, response_model: Type[T]
    ) -> T:
        try:
            completion = await self._client.chat.completions.parse(
                model=request.model or self.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format=response_model,   # ← native structured output
            )

            parsed = completion.choices[0].message.parsed

            if parsed is None:
                raise SchemaError("Model returned no parsed structured output")

            return parsed

        except SchemaError:
            raise

        except ValidationError as e:
            raise SchemaError(f"Structured output validation failed: {e}") from e

        except OpenAIError as e:
            status = getattr(e, "status_code", None)
            if status in {429, 500, 502, 503, 504}:
                raise TransientError(f"Transient OpenAI error: {str(e)}") from e
            else:
                raise PermanentError(f"Permanent OpenAI error: {str(e)}") from e

        except Exception as e:
            raise LLMReliabilityError(f"Unexpected error: {str(e)}") from e
