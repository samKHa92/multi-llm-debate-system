"""Answer normalization and automatic correctness checking.

The dataset uses four ``answer_type`` values and we check each appropriately:
  * integer        -> compare as integers
  * float          -> compare numerically, within ``tolerance`` if provided
  * multiple_choice-> compare the normalized option letter
  * short_text     -> compare normalized text against accepted answers
"""

from __future__ import annotations

import re

from ..models import Problem

# Characters/words we strip so that "153." == "153" and "$1,000" == "1000".
_PUNCT_RE = re.compile(r"[\s,$%]+")


def normalize_answer(answer: str) -> str:
    """Lowercase, trim, and strip common formatting noise."""
    if answer is None:
        return ""
    text = str(answer).strip().lower()
    # Remove trailing period and surrounding quotes/brackets.
    text = text.strip(" .\"'()[]")
    # Collapse whitespace and remove thousands separators / currency symbols.
    text = _PUNCT_RE.sub("", text)
    return text


def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matches_one(predicted: str, target: str, problem: Problem) -> bool:
    atype = problem.answer_type
    p_norm = normalize_answer(predicted)
    t_norm = normalize_answer(target)

    if atype == "integer":
        pf, tf = _try_float(p_norm), _try_float(t_norm)
        if pf is None or tf is None:
            return p_norm == t_norm
        return int(round(pf)) == int(round(tf))

    if atype == "float":
        pf, tf = _try_float(p_norm), _try_float(t_norm)
        if pf is None or tf is None:
            return p_norm == t_norm
        tol = problem.tolerance if problem.tolerance is not None else 1e-6
        return abs(pf - tf) <= tol

    if atype == "multiple_choice":
        # Compare just the leading option letter, e.g. "B) foo" -> "b".
        return (p_norm[:1] if p_norm else "") == (t_norm[:1] if t_norm else "")

    # short_text (default): exact normalized match.
    return p_norm == t_norm


def is_correct(predicted: str, problem: Problem) -> bool:
    """True if ``predicted`` matches the correct or any accepted answer."""
    if predicted is None or str(predicted).strip() == "":
        return False
    targets = [problem.correct_answer, *problem.accepted_answers]
    return any(_matches_one(predicted, t, problem) for t in targets)
