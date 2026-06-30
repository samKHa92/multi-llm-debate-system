"""Tests for the deterministic offline MockLLMClient."""

import json

from src.llm_clients.mock_client import MockLLMClient

_ANSWER_KEY = {"p1": {"correct": "42", "answer_type": "integer"}}


def _client(name="gpt", **kwargs):
    return MockLLMClient(name=name, answer_key=_ANSWER_KEY, **kwargs)


def test_unknown_stage_returns_valid_json_fallback():
    out = _client().generate("no stage marker here")
    assert json.loads(out) == {"answer": "mock"}


def test_role_assignment_stage_shape():
    out = _client().generate("STAGE: role_assignment\nProblem ID: p1")
    data = json.loads(out)
    assert set(data) == {"solver_confidence", "judge_confidence", "reasoning"}
    assert 0.0 <= data["solver_confidence"] <= 1.0
    assert 0.0 <= data["judge_confidence"] <= 1.0


def test_solver_stage_shape_and_fields():
    out = _client().generate("STAGE: solver\nProblem ID: p1\nYour solver slot: solver_1")
    data = json.loads(out)
    assert "final_answer" in data
    assert "confidence" in data
    assert "reasoning" in data


def test_single_stage_only_returns_final_answer():
    out = _client().generate("STAGE: single\nProblem ID: p1")
    data = json.loads(out)
    assert set(data) == {"final_answer"}


def test_high_skill_solver_is_correct():
    out = _client(skill=1.0).generate("STAGE: solver\nProblem ID: p1\nYour solver slot: solver_1")
    assert json.loads(out)["final_answer"] == "42"


def test_zero_skill_solver_is_wrong_but_typed():
    out = _client(skill=0.0).generate("STAGE: solver\nProblem ID: p1\nYour solver slot: solver_1")
    answer = json.loads(out)["final_answer"]
    assert answer != "42"
    assert answer.isdigit()  # integer answer_type -> still a number


def test_unknown_problem_id_yields_mock_answer():
    out = _client().generate("STAGE: solver\nProblem ID: unknown\nYour solver slot: solver_1")
    assert json.loads(out)["final_answer"] == "mock_answer"


def test_generate_is_deterministic():
    prompt = "STAGE: solver\nProblem ID: p1\nYour solver slot: solver_1"
    assert _client().generate(prompt) == _client().generate(prompt)


def test_different_models_can_differ():
    prompt = "STAGE: role_assignment\nProblem ID: p1"
    gpt = _client("gpt").generate(prompt)
    grok = _client("grok").generate(prompt)
    assert gpt != grok  # confidences are salted by model name


def test_judge_prefers_a_correct_refined_answer():
    prompt = (
        "STAGE: judge\nProblem ID: p1\n"
        "solver_1 refined answer: 99\n"
        "solver_2 refined answer: 42\n"
        "solver_3 refined answer: 7\n"
    )
    out = _client(judge_skill=1.0).generate(prompt)
    assert json.loads(out)["winner"] == "solver_2"


def test_judge_winner_is_always_a_valid_slot():
    prompt = (
        "STAGE: judge\nProblem ID: p1\n"
        "solver_1 refined answer: 1\n"
        "solver_2 refined answer: 2\n"
        "solver_3 refined answer: 3\n"
    )
    winner = json.loads(_client().generate(prompt))["winner"]
    assert winner in {"solver_1", "solver_2", "solver_3"}


def test_peer_review_stage_shape():
    prompt = "STAGE: peer_review\nProblem ID: p1\nReviewer: solver_1\nTarget: solver_2"
    data = json.loads(_client().generate(prompt))
    assert {"strengths", "weaknesses", "errors", "suggested_changes", "overall_assessment"} <= set(data)
