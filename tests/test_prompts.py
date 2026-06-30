"""Tests for the per-stage prompt builders."""

import json

from src.llm_clients.mock_client import MockLLMClient
from src.models import PeerReview, Problem, RefinementResult, SolverSolution
from src.prompts import (
    build_judge_prompt,
    build_peer_review_prompt,
    build_refinement_prompt,
    build_role_assignment_prompt,
    build_solver_prompt,
)


def _problem(answer_type="integer"):
    return Problem(
        id="p1",
        category="mathematical_reasoning",
        question="What is 6 * 7?",
        correct_answer="42",
        answer_type=answer_type,
    )


def _solution(slot="solver_1", model_key="gpt", answer="42"):
    return SolverSolution(solver_id=slot, model_key=model_key, reasoning="r", final_answer=answer)


def test_role_assignment_prompt_markers():
    text = build_role_assignment_prompt(_problem())
    assert "STAGE: role_assignment" in text
    assert "STRICT JSON" in text
    assert "p1" in text


def test_solver_prompt_includes_slot_and_stage():
    text = build_solver_prompt(_problem(), "solver_2")
    assert "STAGE: solver" in text
    assert "Your solver slot: solver_2" in text


def test_single_baseline_prompt_has_no_slot():
    text = build_solver_prompt(_problem(), "solver_1", single=True)
    assert "STAGE: single" in text
    assert "Your solver slot" not in text


def test_solver_prompt_answer_hint_varies_by_type():
    assert "integer" in build_solver_prompt(_problem("integer"), "solver_1")
    assert "decimal" in build_solver_prompt(_problem("float"), "solver_1")
    assert "option letter" in build_solver_prompt(_problem("multiple_choice"), "solver_1")


def test_peer_review_prompt_names_reviewer_and_target():
    text = build_peer_review_prompt(_problem(), "solver_1", _solution("solver_2"))
    assert "STAGE: peer_review" in text
    assert "Reviewer: solver_1" in text
    assert "Target: solver_2" in text


def test_refinement_prompt_includes_reviews():
    review = PeerReview(reviewer_id="solver_2", target_id="solver_1", weaknesses=["check arithmetic"])
    text = build_refinement_prompt(_problem(), _solution("solver_1"), [review])
    assert "STAGE: refinement" in text
    assert "check arithmetic" in text


def test_judge_prompt_format_is_parseable_by_mock_judge():
    """The judge prompt must expose refined answers in the exact format the
    mock judge parses, so the two stay in sync."""
    refinements = [
        RefinementResult(solver_id="solver_1", model_key="gpt", refined_answer="99"),
        RefinementResult(solver_id="solver_2", model_key="claude", refined_answer="42"),
        RefinementResult(solver_id="solver_3", model_key="grok", refined_answer="7"),
    ]
    prompt = build_judge_prompt(_problem(), [_solution()], [], refinements)
    assert "STAGE: judge" in prompt

    client = MockLLMClient(
        name="gpt",
        answer_key={"p1": {"correct": "42", "answer_type": "integer"}},
        judge_skill=1.0,
    )
    decision = json.loads(client.generate(prompt))
    assert decision["winner"] == "solver_2"
