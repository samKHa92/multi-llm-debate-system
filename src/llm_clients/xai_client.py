"""xAI (Grok) client wrapper, OpenAI-compatible. Requires XAI_API_KEY."""

from __future__ import annotations

from ..config import get_env
from .base import BaseLLMClient
from .openai_compat import chat_completion


class XAIClient(BaseLLMClient):
    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(name="grok", model_name=model_name or get_env("XAI_MODEL", "grok-3"))
        api_key = get_env("XAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "XAI_API_KEY is not set. Add it to your .env file or use --mode mock."
            )
        base_url = get_env("XAI_BASE_URL", "https://api.x.ai/v1")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The 'openai' package is required for XAIClient. Run: pip install openai") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        return chat_completion(
            self._client,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
