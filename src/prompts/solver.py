"""Stage 1 prompt: a solver solves the problem (single=True for baseline)."""

from __future__ import annotations

from ..models import Problem

_ANSWER_FORMAT_HINTS = {
    "integer": "an integer (digits only, e.g. 153)",
    "float": "a decimal number (e.g. 3.14)",
    "multiple_choice": "a single option letter (e.g. B)",
    "short_text": "a short phrase or word",
}


def _answer_hint(problem: Problem) -> str:
    return _ANSWER_FORMAT_HINTS.get(problem.answer_type, "a short, precise answer")


def build_solver_prompt(problem: Problem, solver_id: str, single: bool = False) -> str:
    stage = "single" if single else "solver"
    slot_line = "" if single else f"Your solver slot: {solver_id}\n"
    intro = (
        "You are answering a hard problem on your own (baseline, no collaboration)."
        if single
        else "You are an independent Solver. Solve the problem on your own; you cannot see other solvers."
    )
    return f"""STAGE: {stage}
Problem ID: {problem.id}
{slot_line}{intro}

Problem category: {problem.category}
Problem:
{problem.question}

The final answer must be {_answer_hint(problem)}.

Respond with STRICT JSON only (no markdown, no extra text) in this schema:
{{
  "reasoning": "<your step-by-step reasoning>",
  "final_answer": "<your final answer, formatted as instructed>",
  "confidence": <float 0..1>
}}
"""
