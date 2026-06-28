"""Abstract base class shared by every LLM client.

Keeping a tiny, uniform interface means the pipeline never cares *which*
provider it is talking to. Swapping GPT for the mock client (or vice versa)
requires no changes anywhere else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Interface implemented by all concrete clients."""

    def __init__(self, name: str, model_name: str) -> None:
        # Logical key, e.g. "gpt" / "claude" / "gemini" / "grok".
        self.name = name
        # Concrete model identifier sent to the provider, e.g. "gpt-4o-mini".
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """Return the raw text completion for ``prompt``.

        Implementations should raise a clear exception (not return junk) if
        they are misconfigured, e.g. a missing API key.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model_name!r})"
