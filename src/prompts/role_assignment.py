"""Stage 0 prompt: each model self-assesses Solver vs Judge fitness."""

from __future__ import annotations

from ..models import Problem


def build_role_assignment_prompt(problem: Problem) -> str:
    return f"""STAGE: role_assignment
Problem ID: {problem.id}

You are one of four LLMs that will collaborate on a hard reasoning problem.
First, decide which role suits you best for THIS specific problem:
  - Solver: produce a full step-by-step solution.
  - Judge: evaluate other models' solutions and pick the best.

Problem category: {problem.category}
Problem:
{problem.question}

Respond with STRICT JSON only (no markdown, no extra text) in this schema:
{{
  "solver_confidence": <float 0..1>,
  "judge_confidence": <float 0..1>,
  "reasoning": "<one or two sentences explaining your self-assessment>"
}}
"""
