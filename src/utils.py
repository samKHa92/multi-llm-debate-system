"""Small, dependency-light helpers shared across the project."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from .models import Problem

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_problems(path: str | Path) -> list[Problem]:
    return [Problem(**row) for row in load_jsonl(path)]


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _loads(candidate: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _first_brace_block(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> dict[str, Any]:
    """Best-effort JSON object extraction from an LLM reply; {} on failure."""
    if not text:
        return {}

    candidates: list[str] = [text]
    fence = _FENCE_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    block = _first_brace_block(text)
    if block:
        candidates.append(block)

    for candidate in candidates:
        parsed = _loads(candidate)
        if parsed is not None:
            return parsed
    return {}


def coerce_float(value: Any, default: float = 0.5) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f > 1.0:  # tolerate percentages like 85 -> 0.85
        f /= 100.0
    return max(0.0, min(1.0, f))


def majority_vote(
    answers: Iterable[str],
    normalizer: Callable[[Any], str] | None = None,
) -> tuple[str, str]:
    """Return (winning_answer, consensus_type in {full, partial, none})."""
    norm = normalizer or (lambda x: str(x).strip().lower())
    cleaned = [a for a in answers if a is not None and str(a).strip()]
    if not cleaned:
        return "", "none"

    counts = Counter(norm(a) for a in cleaned)
    best_key, best_count = counts.most_common(1)[0]
    total = len(cleaned)

    if best_count == total:
        consensus = "full"
    elif best_count > total / 2:
        consensus = "partial"
    else:
        return "", "none"

    winner = next(a for a in cleaned if norm(a) == best_key)
    return winner, consensus
