"""Tests for the pydantic schemas in src/models.py."""

from src.models import EvaluationRow, Problem


def test_problem_defaults():
    p = Problem(id="x", category="c", question="q", correct_answer="a")
    assert p.answer_type == "short_text"
    assert p.difficulty == "hard"
    assert p.accepted_answers == []
    assert p.tolerance is None


def test_evaluation_row_flattens_single_model_correct():
    row = EvaluationRow(
        problem_id="p1",
        category="logic",
        correct_answer="yes",
        single_model_correct={"gpt": True, "claude": False},
    )
    flat = row.to_flat_dict()

    # The nested dict is expanded into per-model columns and removed.
    assert "single_model_correct" not in flat
    assert flat["single_gpt_correct"] is True
    assert flat["single_claude_correct"] is False


def test_evaluation_row_flat_dict_is_csv_friendly():
    row = EvaluationRow(problem_id="p1", category="c", correct_answer="a")
    flat = row.to_flat_dict()
    # Every value should be a scalar (no nested dicts/lists) for a clean CSV.
    assert all(not isinstance(v, (dict, list)) for v in flat.values())


def test_evaluation_row_defaults():
    row = EvaluationRow(problem_id="p1", category="c", correct_answer="a")
    assert row.debate_correct is False
    assert row.voting_correct is False
    assert row.judge_correct_when_disagreement is None
    assert row.consensus_type == ""
