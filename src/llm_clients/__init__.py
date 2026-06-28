"""LLM client implementations.

All clients implement the same :class:`BaseLLMClient` interface so they are
fully interchangeable in the pipeline.
"""

from .base import BaseLLMClient
from .mock_client import MockLLMClient

__all__ = ["BaseLLMClient", "MockLLMClient"]
