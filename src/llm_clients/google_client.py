"""Google (Gemini) client wrapper: API key or ADC (enterprise) auth."""

from __future__ import annotations

from ..config import get_env
from .base import BaseLLMClient


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def _enterprise_enabled() -> bool:
    return _truthy(get_env("GOOGLE_GENAI_USE_ENTERPRISE")) or _truthy(
        get_env("GOOGLE_GENAI_USE_VERTEXAI")
    )


def _use_adc() -> bool:
    if _enterprise_enabled():
        return True
    return bool(get_env("GOOGLE_CLOUD_PROJECT")) and not get_env("GOOGLE_API_KEY")


def _normalize_enterprise_model(model_name: str) -> str:
    aliases = {
        "gemini-2.0-flash-001": "gemini-2.0-flash",
        "gemini-2.0-flash-lite-001": "gemini-2.0-flash-lite",
    }
    return aliases.get(model_name.strip(), model_name.strip())


class GoogleClient(BaseLLMClient):
    def __init__(self, model_name: str | None = None) -> None:
        self._adc = _use_adc()
        default_model = "gemini-2.5-flash" if self._adc else "gemini-2.0-flash-001"
        raw_model = model_name or get_env("GOOGLE_MODEL", default_model)
        resolved = _normalize_enterprise_model(raw_model) if self._adc else raw_model
        super().__init__(name="gemini", model_name=resolved)
        self._location = get_env("GOOGLE_CLOUD_LOCATION", "global") if self._adc else ""
        self._project = get_env("GOOGLE_CLOUD_PROJECT", "") if self._adc else ""
        self._client = self._build_client()

    def _build_client(self):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as exc:
            raise RuntimeError(
                "The 'google-genai' package is required for GoogleClient. Run: pip install google-genai"
            ) from exc

        self._genai_types = genai_types

        if self._adc:
            project = get_env("GOOGLE_CLOUD_PROJECT")
            if not project:
                raise RuntimeError(
                    "Gemini Enterprise / ADC mode requires GOOGLE_CLOUD_PROJECT in .env. "
                    "Set GOOGLE_GENAI_USE_ENTERPRISE=true, then run:\n"
                    "  gcloud auth application-default login\n"
                    "  gcloud auth application-default set-quota-project YOUR_PROJECT"
                )
            location = get_env("GOOGLE_CLOUD_LOCATION", "global")
            return genai.Client(
                enterprise=True,
                project=project,
                location=location,
                http_options=genai_types.HttpOptions(api_version="v1"),
            )

        api_key = get_env("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Google auth not configured. Either set GOOGLE_API_KEY, or for ADC set "
                "GOOGLE_GENAI_USE_ENTERPRISE=true and GOOGLE_CLOUD_PROJECT, then run "
                "gcloud auth application-default login. Or use --mode mock."
            )
        return genai.Client(api_key=api_key)

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        try:
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self._genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return resp.text or ""
        except Exception as exc:
            err = str(exc)
            if "404" in err and "NOT_FOUND" in err:
                raise RuntimeError(
                    f"Google model not found: {self.model_name!r} "
                    f"(project={self._project!r}, location={self._location!r}).\n"
                    "For Gemini Enterprise Agent Platform, try in .env:\n"
                    "  GOOGLE_GENAI_USE_ENTERPRISE=true\n"
                    "  GOOGLE_CLOUD_PROJECT=multi-llm-debate-system\n"
                    "  GOOGLE_CLOUD_LOCATION=global\n"
                    "  GOOGLE_MODEL=gemini-2.5-flash\n"
                    "Also verify in GCP Console:\n"
                    "  1) Agent Platform API is enabled\n"
                    "  2) Your user has roles/aiplatform.user on the project\n"
                    "  3) ADC is fresh: gcloud auth application-default login"
                ) from exc
            raise
