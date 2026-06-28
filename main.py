"""Command-line entry point for the Multi-LLM Collaborative Debate System.

Examples
--------
Run the full pipeline offline (no API keys) on the first 3 problems:
    python main.py --mode mock --limit 3

Run on the whole dataset with real APIs (requires keys in .env):
    python main.py --mode real --problems data/problems.jsonl

Resume a named run, skipping problems already cached:
    python main.py --mode real --run-id final_run_01 --skip-existing true
"""

from __future__ import annotations

import argparse
import time

from tqdm import tqdm

from src.config import DEFAULT_PROBLEMS_FILE, PLOTS_DIR, RESULTS_CSV, ensure_dirs
from src.pipeline.debate_runner import build_clients, run_path, run_problem
from src.pipeline.evaluation import evaluate_run
from src.plotting import generate_all_plots
from src.utils import load_problems


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

    # --- Run the debate for each problem (with caching / skip-existing) ---
    for problem in tqdm(problems, desc="Debating", unit="problem"):
        if args.skip_existing and run_path(run_id, problem.id).exists():
            tqdm.write(f"  skip {problem.id} (cached)")
            continue
        run_problem(clients, problem, run_id=run_id, save=True)

    # --- Evaluate and write results.csv (+ metrics.json) ---
    print("\nEvaluating run...")
    df, metrics = evaluate_run(run_id)
    print(f"Wrote {RESULTS_CSV} ({len(df)} rows)")

    print("\n=== Summary metrics ===")
    print(f"  Single-LLM baseline accuracy : {metrics['single_llm_baseline_accuracy']:.1%}")
    print(f"  Simple voting baseline       : {metrics['simple_voting_baseline_accuracy']:.1%}")
    print(f"  Full debate system accuracy  : {metrics['full_debate_system_accuracy']:.1%}")
    print(f"  Improvement rate             : {metrics['improvement_rate']:.1%}")
    print(f"  Consensus rate               : {metrics['consensus_rate']:.1%}")
    judge_acc = metrics.get("judge_accuracy_when_disagreement")
    judge_str = "n/a" if judge_acc is None else f"{judge_acc:.1%}"
    print(f"  Judge accuracy (disagreement): {judge_str}")

    # --- Plots ---
    if not args.no_plots:
        print("\nGenerating plots...")
        paths = generate_all_plots(metrics)
        for p in paths:
            print(f"  saved {p}")
        print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
