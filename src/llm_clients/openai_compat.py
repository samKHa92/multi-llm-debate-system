"""Shared chat-completion helper for OpenAI-compatible APIs.

Newer OpenAI models (e.g. gpt-5.x) require ``max_completion_tokens`` instead of
``max_tokens``. xAI and older OpenAI models still use ``max_tokens``. We try
both so each client works without hard-coding model names.
"""

from __future__ import annotations

from typing import Any


def chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    from openai import BadRequestError

    base = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    last_error: BadRequestError | None = None

    # Prefer max_completion_tokens first (newer OpenAI), then max_tokens (xAI / legacy).
    for param in ("max_completion_tokens", "max_tokens"):
        try:
            resp = client.chat.completions.create(**base, **{param: max_tokens})
            return resp.choices[0].message.content or ""
        except BadRequestError as exc:
            err = str(exc).lower()
            if "unsupported parameter" in err and param.replace("_", " ") in err.replace("_", " "):
                last_error = exc
                continue
            if "unsupported parameter" in err and param in err:
                last_error = exc
                continue
            raise

    if last_error is not None:
        raise last_error
    return ""
