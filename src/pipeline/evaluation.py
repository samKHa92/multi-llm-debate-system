"""Turn raw run traces into results.csv and summary metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import pandas as pd

from ..config import MODEL_KEYS, RESULTS_CSV, RUNS_DIR
from ..models import EvaluationRow, ProblemRunResult
from ..utils import majority_vote
from .answer_extraction import is_correct, normalize_answer


def load_run_results(run_id: str) -> list[ProblemRunResult]:
    run_dir = RUNS_DIR / run_id
    results: list[ProblemRunResult] = []
    if not run_dir.exists():
        return results
    for path in sorted(run_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as fh:
            results.append(ProblemRunResult(**json.load(fh)))
    return results


def build_evaluation_row(result: ProblemRunResult) -> EvaluationRow:
    problem = result.problem
    initial = {s.solver_id: s.final_answer for s in result.initial_solutions}
    refined = {r.solver_id: r.refined_answer for r in result.refinements}

    slots = ["solver_1", "solver_2", "solver_3"]
    initial_list = [initial.get(s, "") for s in slots]
    refined_list = [refined.get(s, "") for s in slots]

    debate_answer = result.debate_final_answer
    debate_ok = is_correct(debate_answer, problem)

    # Voting baseline = majority of the 3 initial solver answers.
    voting_answer, consensus_type = majority_vote(initial_list, normalizer=normalize_answer)
    voting_ok = is_correct(voting_answer, problem) if voting_answer else False

    n_correct_initial = sum(is_correct(a, problem) for a in initial_list)
    n_correct_refined = sum(is_correct(a, problem) for a in refined_list)
    improved = n_correct_refined > n_correct_initial

    refined_norm = {normalize_answer(a) for a in refined_list if str(a).strip()}
    solvers_disagree = len(refined_norm) > 1
    judge_correct_disagree = debate_ok if solvers_disagree else None

    single_correct = {
        key: is_correct(result.single_model_answers.get(key, ""), problem)
        for key in MODEL_KEYS
        if key in result.single_model_answers
    }

    return EvaluationRow(
        problem_id=problem.id,
        category=problem.category,
        correct_answer=problem.correct_answer,
        solver_1_initial_answer=initial_list[0],
        solver_2_initial_answer=initial_list[1],
        solver_3_initial_answer=initial_list[2],
        solver_1_refined_answer=refined_list[0],
        solver_2_refined_answer=refined_list[1],
        solver_3_refined_answer=refined_list[2],
        judge_winner=result.judge_decision.winner,
        debate_final_answer=debate_answer,
        debate_correct=debate_ok,
        voting_answer=voting_answer,
        voting_correct=voting_ok,
        consensus_type=consensus_type,
        improved_after_refinement=improved,
        judge_correct_when_disagreement=judge_correct_disagree,
        single_model_correct=single_correct,
    )


def compute_metrics(rows: list[EvaluationRow]) -> dict:
    n = len(rows)
    if n == 0:
        return {}

    def rate(predicate) -> float:
        return sum(1 for r in rows if predicate(r)) / n

    disagreement_rows = [r for r in rows if r.judge_correct_when_disagreement is not None]
    judge_acc = (
        fmean(bool(r.judge_correct_when_disagreement) for r in disagreement_rows)
        if disagreement_rows
        else None
    )

    single_acc: dict[str, float] = {}
    for key in MODEL_KEYS:
        vals = [r.single_model_correct[key] for r in rows if key in r.single_model_correct]
        if vals:
            single_acc[key] = fmean(vals)
    single_avg = fmean(single_acc.values()) if single_acc else 0.0

    cat_correct: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        cat_correct[r.category].append(r.debate_correct)
    by_category = {cat: fmean(cat_correct[cat]) for cat in sorted(cat_correct)}

    return {
        "num_problems": n,
        "overall_accuracy": rate(lambda r: r.debate_correct),
        "improvement_rate": rate(lambda r: r.improved_after_refinement),
        "consensus_rate": rate(lambda r: r.consensus_type == "full"),
        "judge_accuracy_when_disagreement": judge_acc,
        "num_disagreement_problems": len(disagreement_rows),
        "single_llm_baseline_accuracy": single_avg,
        "single_llm_accuracy_by_model": single_acc,
        "simple_voting_baseline_accuracy": rate(lambda r: r.voting_correct),
        "full_debate_system_accuracy": rate(lambda r: r.debate_correct),
        "debate_accuracy_by_category": by_category,
    }


def evaluate_run(
    run_id: str,
    results: list[ProblemRunResult] | None = None,
    csv_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    if results is None:
        results = load_run_results(run_id)
    if not results:
        raise RuntimeError(f"No run results found for run_id={run_id!r}. Run the pipeline first.")

    rows = [build_evaluation_row(r) for r in results]
    metrics = compute_metrics(rows)

    df = pd.DataFrame([r.to_flat_dict() for r in rows])
    csv_path = csv_path or RESULTS_CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    metrics_path = csv_path.parent / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return df, metrics
