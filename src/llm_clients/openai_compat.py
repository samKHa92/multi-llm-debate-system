"""Shared chat-completion helper for OpenAI-compatible APIs."""

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

    # Newer OpenAI models want max_completion_tokens; xAI/legacy use max_tokens.
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
