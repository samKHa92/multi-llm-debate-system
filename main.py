"""CLI entry point for the Multi-LLM Collaborative Debate System.

Examples:
    python main.py --mode mock --limit 3
    python main.py --mode real --problems data/problems.jsonl
    python main.py --mode real --run-id final_run_01 --skip-existing true
"""

from __future__ import annotations

import argparse
import time

from tqdm import tqdm

from src.config import (
    DEFAULT_PROBLEMS_FILE,
    PLOTS_DIR,
    RESULTS_CSV,
    ensure_dirs,
    plots_dir_for_run,
)
from src.llm_clients.base import BaseLLMClient
from src.models import Problem
from src.pipeline.debate_runner import build_clients, run_path, run_problem
from src.pipeline.evaluation import evaluate_run
from src.plotting import generate_all_plots
from src.utils import load_problems

_SUMMARY_FIELDS = [
    ("Single-LLM baseline accuracy", "single_llm_baseline_accuracy"),
    ("Simple voting baseline", "simple_voting_baseline_accuracy"),
    ("Full debate system accuracy", "full_debate_system_accuracy"),
    ("Improvement rate", "improvement_rate"),
    ("Consensus rate", "consensus_rate"),
]


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-LLM Collaborative Debate System")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock", help="Which clients to use.")
    parser.add_argument("--problems", default=str(DEFAULT_PROBLEMS_FILE), help="Path to a .jsonl dataset.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N problems.")
    parser.add_argument("--run-id", default=None, help="Run identifier (folder name under outputs/runs/).")
    parser.add_argument(
        "--skip-existing",
        type=_str2bool,
        default=False,
        help="If true, do not rerun problems that already have a cached trace.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def _debate_all(
    clients: dict[str, BaseLLMClient],
    problems: list[Problem],
    run_id: str,
    skip_existing: bool,
) -> None:
    for problem in tqdm(problems, desc="Debating", unit="problem"):
        if skip_existing and run_path(run_id, problem.id).exists():
            tqdm.write(f"  skip {problem.id} (cached)")
            continue
        run_problem(clients, problem, run_id=run_id, save=True)


def _print_summary(metrics: dict) -> None:
    print("\n=== Summary metrics ===")
    for label, key in _SUMMARY_FIELDS:
        print(f"  {label:<29}: {metrics[key]:.1%}")
    judge_acc = metrics.get("judge_accuracy_when_disagreement")
    judge_str = "n/a" if judge_acc is None else f"{judge_acc:.1%}"
    print(f"  {'Judge accuracy (disagreement)':<29}: {judge_str}")


def _emit_plots(metrics: dict, run_id: str) -> None:
    print("\nGenerating plots...")
    # Per-run copy preserves history; top-level dir holds the latest.
    run_plots_dir = plots_dir_for_run(run_id)
    for path in generate_all_plots(metrics, run_plots_dir):
        print(f"  saved {path}")
    print(f"Run plots saved to {run_plots_dir}")

    generate_all_plots(metrics, PLOTS_DIR)
    print(f"Latest plots refreshed in {PLOTS_DIR}")


def main() -> None:
    args = parse_args()
    ensure_dirs()

    run_id = args.run_id or f"{args.mode}_run_{time.strftime('%Y%m%d_%H%M%S')}"
    problems = load_problems(args.problems)
    if args.limit is not None:
        problems = problems[: args.limit]

    print(f"Mode: {args.mode} | Run ID: {run_id} | Problems: {len(problems)}")
    clients = build_clients(args.mode, problems)
    print("Clients:", ", ".join(f"{k}={v.model_name}" for k, v in clients.items()))

    _debate_all(clients, problems, run_id, args.skip_existing)

    print("\nEvaluating run...")
    df, metrics = evaluate_run(run_id)
    print(f"Wrote {RESULTS_CSV} ({len(df)} rows)")

    _print_summary(metrics)

    if not args.no_plots:
        _emit_plots(metrics, run_id)


if __name__ == "__main__":
    main()
