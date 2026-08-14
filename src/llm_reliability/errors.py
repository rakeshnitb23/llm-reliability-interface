# src/llm_reliability/errors.py

class LLMReliabilityError(Exception):
    """Base exception for the entire reliability layer."""
    pass


class TransientError(LLMReliabilityError):
    """
    Temporary failure (rate limit, 500, 502, 503, network blip).
    Safe to retry.
    """
    pass


class PermanentError(LLMReliabilityError):
    """
    Permanent failure (invalid request, auth error, bad schema, etc.).
    Should not be retried.
    """
    pass


class SchemaError(LLMReliabilityError):
    """
    Raised when the model returns data that does not match the expected schema.
    (We will use this more in later slices)
    """
    pass
