"""Anthropic (Claude) client wrapper.

Requires the ``anthropic`` package and an ``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from ..config import get_env
from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(
            name="claude",
            model_name=model_name or get_env("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        )
        api_key = get_env("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file or use --mode mock."
            )
        try:
            import anthropic  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("The 'anthropic' package is required for AnthropicClient. Run: pip install anthropic") from exc
        self._client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        resp = self._client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Claude returns a list of content blocks; concatenate any text blocks.
        parts = [block.text for block in resp.content if getattr(block, "type", "") == "text"]
        return "".join(parts)
