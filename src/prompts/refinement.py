"""Stage 3 prompt: a solver refines its solution given two peer reviews."""

from __future__ import annotations

from ..models import PeerReview, Problem, SolverSolution


def _format_review(review: PeerReview, idx: int) -> str:
    errors = "; ".join(f"[{e.severity}] {e.location}: {e.description}" for e in review.errors) or "none"
    return (
        f"Review {idx} (from {review.reviewer_id}):\n"
        f"  strengths: {review.strengths}\n"
        f"  weaknesses: {review.weaknesses}\n"
        f"  errors: {errors}\n"
        f"  suggested_changes: {review.suggested_changes}\n"
        f"  overall: {review.overall_assessment}"
    )


def build_refinement_prompt(
    problem: Problem,
    original_solution: SolverSolution,
    peer_reviews: list[PeerReview],
) -> str:
    reviews_block = "\n\n".join(_format_review(r, i + 1) for i, r in enumerate(peer_reviews))
    return f"""STAGE: refinement
Problem ID: {problem.id}
Your solver slot: {original_solution.solver_id}

You are Solver {original_solution.solver_id}. Two peers reviewed your solution.
Address EACH critique explicitly: accept valid ones and fix your solution, or
reject invalid ones with a clear explanation. Then produce your refined answer.

Problem:
{problem.question}

Your original reasoning:
{original_solution.reasoning}

Your original final answer: {original_solution.final_answer}

Peer reviews:
{reviews_block}

Respond with STRICT JSON only (no markdown, no extra text) in this schema:
{{
  "changes_made": [
    {{"critique": "<the critique>", "response": "<how you addressed it>", "accepted": true}}
  ],
  "refined_solution": "<your updated reasoning>",
  "refined_answer": "<your final answer, formatted as instructed>",
  "confidence": <float 0..1>
}}
"""
