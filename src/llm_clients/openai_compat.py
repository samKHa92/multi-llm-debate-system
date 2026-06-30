"""Shared chat-completion helper for OpenAI-compatible APIs."""

from __future__ import annotations

import sys
from typing import Any


def _is_content_policy_error(message: str) -> bool:
    """True if the error is a content-moderation / policy flag (not a config bug)."""
    m = message.lower()
    return (
        "invalid_prompt" in m
        or "usage policy" in m
        or "flagged" in m
        or "content_policy" in m
        or "content management policy" in m  # Azure phrasing
        or "responsibleai" in m
    )


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
            # A prompt flagged by content moderation is deterministic and would
            # crash a long run. Degrade gracefully: this single call returns ""
            # and the pipeline falls back to defaults for that stage.
            if _is_content_policy_error(err):
                print(
                    f"[warn] {model}: prompt flagged by content policy; "
                    "returning empty response for this call.",
                    file=sys.stderr,
                )
                return ""
            # The "wrong token param" case: try the other parameter name.
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
