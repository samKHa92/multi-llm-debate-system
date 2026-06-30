"""Additional unit tests for evaluation logic and aggregation edge cases."""

import json

import pytest

from src.models import (
    AssignedRoles,
    JudgeDecision,
    Problem,
    ProblemRunResult,
    RefinementResult,
    SolverSolution,
)
from src.pipeline.evaluation import (
    build_evaluation_row,
    compute_metrics,
    evaluate_run,
    load_run_results,
)


def _result(
    *,
    initial,
    refined,
    debate_answer,
    winner="solver_1",
    single=None,
    answer_type="integer",
    correct="42",
    category="math",
):
    problem = Problem(
        id="p1",
        category=category,
        question="q",
        correct_answer=correct,
        answer_type=answer_type,
    )
    slots = ["solver_1", "solver_2", "solver_3"]
    return ProblemRunResult(
        run_id="t",
        problem=problem,
        assigned_roles=AssignedRoles(judge="grok", solvers=["gpt", "claude", "gemini"]),
        initial_solutions=[
            SolverSolution(solver_id=s, model_key="gpt", final_answer=a)
            for s, a in zip(slots, initial)
        ],
        refinements=[
            RefinementResult(solver_id=s, model_key="gpt", refined_answer=a)
            for s, a in zip(slots, refined)
        ],
        judge_decision=JudgeDecision(judge_model_key="grok", winner=winner),
        debate_final_answer=debate_answer,
        single_model_answers=single or {},
    )


def test_build_evaluation_row_voting_and_consensus():
    row = build_evaluation_row(
        _result(initial=["42", "42", "99"], refined=["42", "42", "42"], debate_answer="42")
    )
    assert row.debate_correct is True
    assert row.voting_answer == "42"
    assert row.voting_correct is True
    assert row.consensus_type == "partial"


def test_build_evaluation_row_detects_improvement():
    # 1 correct initially, 3 correct after refinement.
    row = build_evaluation_row(
        _result(initial=["42", "1", "2"], refined=["42", "42", "42"], debate_answer="42")
    )
    assert row.improved_after_refinement is True


def test_build_evaluation_row_no_improvement_when_stable():
    row = build_evaluation_row(
        _result(initial=["42", "42", "42"], refined=["42", "42", "42"], debate_answer="42")
    )
    assert row.improved_after_refinement is False


def test_judge_correctness_only_scored_on_disagreement():
    # Refined answers all agree -> not a disagreement problem.
    agree = build_evaluation_row(
        _result(initial=["42", "42", "42"], refined=["42", "42", "42"], debate_answer="42")
    )
    assert agree.judge_correct_when_disagreement is None

    # Refined answers differ -> judged, and the winner was correct.
    disagree = build_evaluation_row(
        _result(initial=["1", "2", "3"], refined=["42", "7", "9"], debate_answer="42")
    )
    assert disagree.judge_correct_when_disagreement is True


def test_compute_metrics_empty_rows():
    assert compute_metrics([]) == {}


def test_compute_metrics_values():
    row = build_evaluation_row(
        _result(
            initial=["42", "42", "99"],
            refined=["42", "42", "42"],
            debate_answer="42",
            single={"gpt": "42", "claude": "1", "gemini": "2", "grok": "3"},
        )
    )
    metrics = compute_metrics([row])
    assert metrics["num_problems"] == 1
    assert metrics["full_debate_system_accuracy"] == 1.0
    assert metrics["simple_voting_baseline_accuracy"] == 1.0
    assert metrics["single_llm_baseline_accuracy"] == 0.25  # 1 of 4 models correct
    assert metrics["debate_accuracy_by_category"] == {"math": 1.0}


def test_load_run_results_missing_dir_returns_empty():
    assert load_run_results("definitely_not_a_real_run_id") == []


def test_evaluate_run_raises_without_results():
    with pytest.raises(RuntimeError):
        evaluate_run("missing_run", results=[])


def test_evaluate_run_writes_csv_and_metrics(tmp_path):
    result = _result(
        initial=["42", "42", "99"],
        refined=["42", "42", "42"],
        debate_answer="42",
        single={"gpt": "42", "claude": "1", "gemini": "2", "grok": "3"},
    )
    csv_path = tmp_path / "results.csv"
    df, metrics = evaluate_run("t", results=[result], csv_path=csv_path)

    assert csv_path.exists()
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists()
    assert json.loads(metrics_path.read_text())["num_problems"] == 1
    assert len(df) == 1
