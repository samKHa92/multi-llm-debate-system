"""End-to-end tests of the mock pipeline (no API keys required)."""

from src.models import Problem
from src.pipeline.debate_runner import build_clients, run_problem
from src.pipeline.evaluation import build_evaluation_row, compute_metrics, evaluate_run


def _sample_problem():
    return Problem(
        id="math_test_001",
        category="mathematical_reasoning",
        question="What is 2 + 2?",
        correct_answer="4",
        accepted_answers=["4"],
        answer_type="integer",
    )


def test_mock_pipeline_runs_one_problem_end_to_end():
    problem = _sample_problem()
    clients = build_clients("mock", [problem])
    result = run_problem(clients, problem, run_id="pytest_run", save=False)

    # Roles: exactly one judge, three solver slots.
    assert result.assigned_roles.judge in {"gpt", "claude", "gemini", "grok"}
    assert len(result.assigned_roles.solver_slots) == 3
    assert result.assigned_roles.judge not in result.assigned_roles.solvers

    # Stage outputs are all present and well-formed.
    assert len(result.initial_solutions) == 3
    assert len(result.peer_reviews) == 6  # 3 solvers x 2 peers
    assert len(result.refinements) == 3
    assert result.judge_decision.winner in {"solver_1", "solver_2", "solver_3"}

    # The debate final answer is the winner's refined answer.
    winner = result.judge_decision.winner
    winner_refined = next(r.refined_answer for r in result.refinements if r.solver_id == winner)
    assert result.debate_final_answer == winner_refined

    # Single-model baseline produced an answer for each of the 4 models.
    assert set(result.single_model_answers) == {"gpt", "claude", "gemini", "grok"}


def test_mock_pipeline_is_deterministic():
    problem = _sample_problem()
    clients = build_clients("mock", [problem])
    r1 = run_problem(clients, problem, run_id="d1", save=False)
    r2 = run_problem(clients, problem, run_id="d2", save=False)
    assert r1.debate_final_answer == r2.debate_final_answer
    assert r1.judge_decision.winner == r2.judge_decision.winner


def test_evaluation_row_contains_expected_fields():
    problem = _sample_problem()
    clients = build_clients("mock", [problem])
    result = run_problem(clients, problem, run_id="pytest_run", save=False)
    row = build_evaluation_row(result)

    flat = row.to_flat_dict()
    expected_fields = {
        "problem_id",
        "category",
        "correct_answer",
        "solver_1_initial_answer",
        "solver_2_initial_answer",
        "solver_3_initial_answer",
        "solver_1_refined_answer",
        "solver_2_refined_answer",
        "solver_3_refined_answer",
        "judge_winner",
        "debate_final_answer",
        "debate_correct",
        "voting_answer",
        "voting_correct",
        "consensus_type",
        "improved_after_refinement",
        "judge_correct_when_disagreement",
    }
    assert expected_fields.issubset(flat.keys())
    assert row.consensus_type in {"full", "partial", "none"}


def test_metrics_have_baseline_and_debate_keys():
    problems = [_sample_problem()]
    clients = build_clients("mock", problems)
    results = [run_problem(clients, p, run_id="m", save=False) for p in problems]
    rows = [build_evaluation_row(r) for r in results]
    metrics = compute_metrics(rows)
    for key in [
        "overall_accuracy",
        "improvement_rate",
        "consensus_rate",
        "single_llm_baseline_accuracy",
        "simple_voting_baseline_accuracy",
        "full_debate_system_accuracy",
    ]:
        assert key in metrics


def test_full_run_writes_csv(tmp_path):
    problem = _sample_problem()
    clients = build_clients("mock", [problem])
    result = run_problem(clients, problem, run_id="csvtest", save=False)
    csv_path = tmp_path / "results.csv"
    df, metrics = evaluate_run("csvtest", results=[result], csv_path=csv_path)
    assert csv_path.exists()
    assert len(df) == 1
