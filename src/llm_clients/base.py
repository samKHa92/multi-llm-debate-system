"""Abstract base class shared by every LLM client."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    def __init__(self, name: str, model_name: str) -> None:
        self.name = name  # logical key: gpt | claude | gemini | grok
        self.model_name = model_name  # provider model id, e.g. "gpt-4o-mini"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.model_name!r})"
