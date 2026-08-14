from abc import ABC, abstractmethod
from typing import AsyncGenerator, Type, TypeVar
from llm_reliability.models import CompletionRequest, CompletionResponse, ResponseModel
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseProvider(ABC):
    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        pass

    @abstractmethod
    async def stream(self, request: CompletionRequest) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    async def complete_structured(self, request: CompletionRequest, response_model: Type[T]) -> T:
        pass
