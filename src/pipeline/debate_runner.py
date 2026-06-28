"""The orchestrator: runs the full 4-stage debate for a single problem.

It is intentionally provider-agnostic - it only talks to ``BaseLLMClient``
objects, so the same code path runs in mock mode and real mode.
"""

from __future__ import annotations

from pathlib import Path

from ..config import (
    JUDGE_TEMPERATURE,
    MODEL_KEYS,
    RUNS_DIR,
    SOLVER_TEMPERATURE,
)
from ..llm_clients.base import BaseLLMClient
from ..llm_clients.mock_client import MockLLMClient
from ..models import (
    CritiqueResponse,
    JudgeDecision,
    PeerReview,
    Problem,
    ProblemRunResult,
    RefinementResult,
    ReviewError,
    SolverSolution,
)
from ..prompts import (
    build_judge_prompt,
    build_peer_review_prompt,
    build_refinement_prompt,
    build_solver_prompt,
)
from ..utils import coerce_float, extract_json, save_json
from .role_assignment import assign_roles, collect_self_assessments


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------
# Per-model skill knobs for mock mode. Tuned so that GPT > Claude > Gemini >
# Grok, baselines land around ~0.55-0.7, and the debate system beats both
# baselines after refinement + judging.
_MOCK_SKILL = {"gpt": 0.72, "claude": 0.68, "gemini": 0.6, "grok": 0.55}


def build_clients(mode: str, problems: list[Problem] | None = None) -> dict[str, BaseLLMClient]:
    """Construct the four logical clients keyed by model name.

    mode="mock" -> four MockLLMClients (no keys needed).
    mode="real" -> the real provider wrappers (require API keys in .env).
    """
    if mode == "mock":
        answer_key = {
            p.id: {"correct": p.correct_answer, "answer_type": p.answer_type}
            for p in (problems or [])
        }
        return {
            key: MockLLMClient(
                name=key,
                model_name=f"mock-{key}",
                answer_key=answer_key,
                skill=_MOCK_SKILL.get(key, 0.6),
            )
            for key in MODEL_KEYS
        }

    if mode == "real":
        # Import lazily so mock mode never imports optional SDK wrappers.
        from ..llm_clients.openai_client import OpenAIClient
        from ..llm_clients.anthropic_client import AnthropicClient
        from ..llm_clients.google_client import GoogleClient
        from ..llm_clients.xai_client import XAIClient

        return {
            "gpt": OpenAIClient(),
            "claude": AnthropicClient(),
            "gemini": GoogleClient(),
            "grok": XAIClient(),
        }

    raise ValueError(f"Unknown mode: {mode!r} (expected 'mock' or 'real').")


# ---------------------------------------------------------------------------
# Parsing helpers (turn raw JSON-ish text into validated models)
# ---------------------------------------------------------------------------
def _parse_solver(raw: str, solver_id: str, model_key: str) -> SolverSolution:
    data = extract_json(raw)
    return SolverSolution(
        solver_id=solver_id,
        model_key=model_key,
        reasoning=str(data.get("reasoning", "")),
        final_answer=str(data.get("final_answer", "")).strip(),
        confidence=coerce_float(data.get("confidence"), 0.5),
    )


def _parse_review(raw: str, reviewer_id: str, target_id: str) -> PeerReview:
    data = extract_json(raw)
    errors = []
    for e in data.get("errors", []) or []:
        if isinstance(e, dict):
            errors.append(
                ReviewError(
                    location=str(e.get("location", "")),
                    error_type=str(e.get("error_type", "")),
                    description=str(e.get("description", "")),
                    severity=str(e.get("severity", "minor")),
                )
            )
    return PeerReview(
        reviewer_id=reviewer_id,
        target_id=target_id,
        strengths=[str(x) for x in data.get("strengths", []) or []],
        weaknesses=[str(x) for x in data.get("weaknesses", []) or []],
        errors=errors,
        suggested_changes=[str(x) for x in data.get("suggested_changes", []) or []],
        overall_assessment=str(data.get("overall_assessment", "")),
    )


def _parse_refinement(raw: str, solver_id: str, model_key: str) -> RefinementResult:
    data = extract_json(raw)
    changes = []
    for c in data.get("changes_made", []) or []:
        if isinstance(c, dict):
            changes.append(
                CritiqueResponse(
                    critique=str(c.get("critique", "")),
                    response=str(c.get("response", "")),
                    accepted=bool(c.get("accepted", False)),
                )
            )
    return RefinementResult(
        solver_id=solver_id,
        model_key=model_key,
        changes_made=changes,
        refined_solution=str(data.get("refined_solution", "")),
        refined_answer=str(data.get("refined_answer", "")).strip(),
        confidence=coerce_float(data.get("confidence"), 0.5),
    )


def _parse_judge(raw: str, judge_key: str, valid_slots: list[str]) -> JudgeDecision:
    data = extract_json(raw)
    winner = str(data.get("winner", "")).strip()
    if winner not in valid_slots:
        winner = valid_slots[0]  # safe fallback
    return JudgeDecision(
        judge_model_key=judge_key,
        winner=winner,
        confidence=coerce_float(data.get("confidence"), 0.5),
        reasoning=str(data.get("reasoning", "")),
    )


# ---------------------------------------------------------------------------
# Single-problem orchestration
# ---------------------------------------------------------------------------
def run_problem(
    clients: dict[str, BaseLLMClient],
    problem: Problem,
    run_id: str,
    save: bool = True,
) -> ProblemRunResult:
    """Run all stages for one problem and (optionally) cache the trace."""
    # --- Stage 0 + 0.5: role self-assessment and deterministic assignment ---
    assessments = collect_self_assessments(clients, problem)
    roles = assign_roles(assessments)
    judge_client = clients[roles.judge]

    # --- Stage 1: independent solver solutions ---
    solutions: list[SolverSolution] = []
    for slot, model_key in roles.solver_slots.items():
        raw = clients[model_key].generate(
            build_solver_prompt(problem, slot), temperature=SOLVER_TEMPERATURE
        )
        solutions.append(_parse_solver(raw, slot, model_key))
    sol_by_slot = {s.solver_id: s for s in solutions}

    # --- Stage 2: peer review (each solver reviews the other two) ---
    reviews: list[PeerReview] = []
    for reviewer in solutions:
        for target in solutions:
            if target.solver_id == reviewer.solver_id:
                continue
            raw = clients[reviewer.model_key].generate(
                build_peer_review_prompt(problem, reviewer.solver_id, target),
                temperature=SOLVER_TEMPERATURE,
            )
            reviews.append(_parse_review(raw, reviewer.solver_id, target.solver_id))

    # --- Stage 3: refinement (each solver sees the 2 reviews about itself) ---
    refinements: list[RefinementResult] = []
    for sol in solutions:
        own_reviews = [r for r in reviews if r.target_id == sol.solver_id]
        raw = clients[sol.model_key].generate(
            build_refinement_prompt(problem, sol, own_reviews),
            temperature=SOLVER_TEMPERATURE,
        )
        refinements.append(_parse_refinement(raw, sol.solver_id, sol.model_key))
    refine_by_slot = {r.solver_id: r for r in refinements}

    # --- Stage 4: final judgment ---
    valid_slots = [s.solver_id for s in solutions]
    raw = judge_client.generate(
        build_judge_prompt(problem, solutions, reviews, refinements),
        temperature=JUDGE_TEMPERATURE,
    )
    judge_decision = _parse_judge(raw, roles.judge, valid_slots)

    # Copy the winning solver's refined answer as the final debate answer.
    debate_final_answer = refine_by_slot[judge_decision.winner].refined_answer

    # --- Baseline: single-model answers (one shot, no debate) ---
    single_model_answers: dict[str, str] = {}
    for key, client in clients.items():
        raw = client.generate(
            build_solver_prompt(problem, "solver_1", single=True),
            temperature=SOLVER_TEMPERATURE,
        )
        single_model_answers[key] = str(extract_json(raw).get("final_answer", "")).strip()

    result = ProblemRunResult(
        run_id=run_id,
        problem=problem,
        assigned_roles=roles,
        initial_solutions=solutions,
        peer_reviews=reviews,
        refinements=refinements,
        judge_decision=judge_decision,
        debate_final_answer=debate_final_answer,
        single_model_answers=single_model_answers,
    )

    if save:
        out_path = RUNS_DIR / run_id / f"{problem.id}.json"
        save_json(out_path, result.model_dump())

    return result


def run_path(run_id: str, problem_id: str) -> Path:
    """Where a given problem's trace is cached."""
    return RUNS_DIR / run_id / f"{problem_id}.json"
