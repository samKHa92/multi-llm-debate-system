"""Small, dependency-light helpers shared across the project."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .models import Problem


# ---------------------------------------------------------------------------
# JSON / file helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a .jsonl file into a list of dicts (ignores blank lines)."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_problems(path: str | Path) -> list[Problem]:
    """Load and validate the problem dataset."""
    return [Problem(**row) for row in load_jsonl(path)]


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Robust JSON extraction from LLM text
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from a model's text reply.

    Real LLMs sometimes wrap JSON in markdown fences or add stray prose. We
    try, in order: direct parse, fenced code block, first balanced ``{...}``.
    Returns an empty dict if nothing parses (callers fall back to defaults).
    """
    if not text:
        return {}

    # 1) Direct parse.
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) ```json ... ``` fenced block.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3) First balanced curly-brace block.
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return {}


def coerce_float(value: Any, default: float = 0.5) -> float:
    """Parse a confidence-like value into a float in [0, 1]."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    # Some models answer with percentages (e.g. 85 instead of 0.85).
    if f > 1.0:
        f = f / 100.0
    return max(0.0, min(1.0, f))


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------
def majority_vote(answers: Iterable[str], normalizer=None) -> tuple[str, str]:
    """Return (winning_answer, consensus_type) from a list of answers.

    consensus_type is one of:
      * "full"    - all answers agree
      * "partial" - a strict majority agrees (but not all)
      * "none"    - no majority (all different / tie)
    The first occurrence of a winning value is returned in its original form.
    """
    answers = [a for a in answers if a is not None and str(a).strip() != ""]
    if not answers:
        return "", "none"

    norm = normalizer or (lambda x: str(x).strip().lower())
    counts: dict[str, int] = {}
    first_seen: dict[str, str] = {}
    for a in answers:
        key = norm(a)
        counts[key] = counts.get(key, 0) + 1
        first_seen.setdefault(key, a)

    best_key = max(counts, key=lambda k: counts[k])
    best_count = counts[best_key]
    total = len(answers)

    if best_count == total:
        consensus = "full"
    elif best_count > total / 2:
        consensus = "partial"
    else:
        # No strict majority (e.g. 1/1/1 split or a 1/1 tie).
        return "", "none"

    return first_seen[best_key], consensus
