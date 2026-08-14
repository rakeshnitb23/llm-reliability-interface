# src/llm_reliability/client.py

import asyncio
import contextlib
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional, Literal
from llm_reliability.models import CompletionRequest, CompletionResponse
from llm_reliability.providers.openai_provider import OpenAIProvider
from llm_reliability.providers.anthropic_provider import AnthropicProvider
from llm_reliability.providers.base import BaseProvider
from llm_reliability.errors import PermanentError, TransientError
from typing import Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(
        self,
        provider: Literal["openai", "anthropic"] = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30,
        max_retries: int = 3,
        base_delay: float = 1.0,

    ):
        self.provider_name = provider
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay

        if provider == "openai":
            self._provider: BaseProvider = OpenAIProvider(
                api_key=api_key, model=model or "gpt-4o-mini", timeout=timeout,)
        elif provider == "anthropic":
            self._provider: BaseProvider = AnthropicProvider(
                api_key=api_key, model=model or "claude-3-5-sonnet-20241022", timeout=timeout,)
        else:
            raise PermanentError(f"Unsupported provider: {provider}")

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Public method – works the same for every provider."""
        return await self._with_retries(self._provider.complete, request)

    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        # Streaming is not retried: once tokens have been yielded to the
        # caller, a retry could not be spliced back in without duplicating
        # or losing output.
        #
        # aclosing() ensures that if the caller stops consuming this
        # generator early (cancellation), the provider's underlying stream
        # is deterministically closed rather than left for GC to clean up.
        async with contextlib.aclosing(self._provider.stream(request)) as agen:
            async for token in agen:
                yield token

    async def complete_structured(
        self, request: CompletionRequest, response_model: Type[T]
    ) -> T:
        return await self._with_retries(
            self._provider.complete_structured, request, response_model
        )

    async def _with_retries(self, func: Callable[..., Awaitable[Any]], *args: Any) -> Any:
        attempt = 0
        while True:
            try:
                return await func(*args)
            except TransientError:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                await asyncio.sleep(self.base_delay * (2 ** (attempt - 1)))
