"""Stage 4 prompt: the judge picks the best refined solution."""

from __future__ import annotations

from ..models import PeerReview, Problem, RefinementResult, SolverSolution


def build_judge_prompt(
    problem: Problem,
    original_solutions: list[SolverSolution],
    peer_reviews: list[PeerReview],
    refined_solutions: list[RefinementResult],
) -> str:
    originals_block = "\n\n".join(
        f"{s.solver_id} ({s.model_key}) original answer: {s.final_answer}\n"
        f"  reasoning: {s.reasoning}"
        for s in original_solutions
    )
    reviews_block = "\n".join(
        f"{r.reviewer_id} -> {r.target_id}: {r.overall_assessment} "
        f"(weaknesses: {r.weaknesses})"
        for r in peer_reviews
    )
    # "<solver_x> refined answer: <value>" is parsed by the mock judge; keep stable.
    refined_block = "\n\n".join(
        f"{r.solver_id} refined answer: {r.refined_answer}\n"
        f"  refined solution: {r.refined_solution}\n"
        f"  confidence: {r.confidence}"
        for r in refined_solutions
    )
    return f"""STAGE: judge
Problem ID: {problem.id}

You are the Final Judge. Review all three solvers' original solutions, the peer
reviews, and the refined solutions. Decide which refined solution is best and
most likely correct. You must pick exactly one winner.

Problem:
{problem.question}

=== Original solutions ===
{originals_block}

=== Peer reviews ===
{reviews_block}

=== Refined solutions ===
{refined_block}

Respond with STRICT JSON only (no markdown, no extra text) in this schema:
{{
  "winner": "solver_1 | solver_2 | solver_3",
  "confidence": <float 0..1>,
  "reasoning": "<why this solution is the strongest>"
}}
"""
