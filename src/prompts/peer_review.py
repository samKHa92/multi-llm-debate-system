"""Stage 2 prompt: a solver reviews one peer's solution."""

from __future__ import annotations

from ..models import Problem, SolverSolution


def build_peer_review_prompt(
    problem: Problem,
    reviewer_id: str,
    target_solution: SolverSolution,
) -> str:
    return f"""STAGE: peer_review
Problem ID: {problem.id}
Reviewer: {reviewer_id}
Target: {target_solution.solver_id}

You are Solver {reviewer_id}. Critically review the solution below written by a
peer ({target_solution.solver_id}). Be rigorous and specific.

Problem:
{problem.question}

Peer's reasoning:
{target_solution.reasoning}

Peer's final answer: {target_solution.final_answer}

Respond with STRICT JSON only (no markdown, no extra text) in this schema:
{{
  "strengths": ["..."],
  "weaknesses": ["..."],
  "errors": [
    {{"location": "<where>", "error_type": "<type>", "description": "<what>", "severity": "minor|major|critical"}}
  ],
  "suggested_changes": ["..."],
  "overall_assessment": "<short verdict, e.g. solid / promising_but_flawed / incorrect>"
}}
"""
