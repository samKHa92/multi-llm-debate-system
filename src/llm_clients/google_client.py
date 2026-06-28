"""Google (Gemini) client wrapper.

Requires the ``google-generativeai`` package and a ``GOOGLE_API_KEY``.
"""

from __future__ import annotations

from ..config import get_env
from .base import BaseLLMClient


class GoogleClient(BaseLLMClient):
    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(name="gemini", model_name=model_name or get_env("GOOGLE_MODEL", "gemini-1.5-flash"))
        api_key = get_env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Add it to your .env file or use --mode mock."
            )
        try:
            import google.generativeai as genai  # lazy import
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'google-generativeai' package is required for GoogleClient. Run: pip install google-generativeai"
            ) from exc
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        resp = self._model.generate_content(
            prompt,
            generation_config=self._genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return resp.text or ""
