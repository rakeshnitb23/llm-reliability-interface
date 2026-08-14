# src/llm_reliability/models.py

from typing import Optional
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The user prompt")
    model: str = "gpt-4o-mini"
    max_tokens: Optional[int] = Field(default=None, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class CompletionResponse(BaseModel):
    content: str
    model: str
    provider: str = "openai"          # useful later when we have multiple providers
    usage: Optional[dict] = None


class ResponseModel(BaseModel):
    content: str
    model: str
    provider: str = "openai"          # useful later when we have multiple providers
    usage: Optional[dict] = None
