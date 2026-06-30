"""Answer normalization and automatic correctness checking."""

from __future__ import annotations

import re

from ..models import Problem

_PUNCT_RE = re.compile(r"[\s,$%]+")
_DEFAULT_FLOAT_TOL = 1e-6


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    text = str(answer).strip().lower().strip(" .\"'()[]")
    return _PUNCT_RE.sub("", text)


def _try_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _match_integer(p: str, t: str, _: Problem) -> bool:
    pf, tf = _try_float(p), _try_float(t)
    if pf is None or tf is None:
        return p == t
    return round(pf) == round(tf)


def _match_float(p: str, t: str, problem: Problem) -> bool:
    pf, tf = _try_float(p), _try_float(t)
    if pf is None or tf is None:
        return p == t
    tol = problem.tolerance if problem.tolerance is not None else _DEFAULT_FLOAT_TOL
    return abs(pf - tf) <= tol


def _match_multiple_choice(p: str, t: str, _: Problem) -> bool:
    return p[:1] == t[:1]


def _match_short_text(p: str, t: str, _: Problem) -> bool:
    return p == t


_MATCHERS = {
    "integer": _match_integer,
    "float": _match_float,
    "multiple_choice": _match_multiple_choice,
    "short_text": _match_short_text,
}


def _matches_one(predicted: str, target: str, problem: Problem) -> bool:
    matcher = _MATCHERS.get(problem.answer_type, _match_short_text)
    return matcher(normalize_answer(predicted), normalize_answer(target), problem)


def is_correct(predicted: str, problem: Problem) -> bool:
    if predicted is None or not str(predicted).strip():
        return False
    targets = [problem.correct_answer, *problem.accepted_answers]
    return any(_matches_one(predicted, t, problem) for t in targets)
