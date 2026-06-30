"""Central configuration: paths, model registry, and defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_CSV = OUTPUTS_DIR / "results.csv"
DEFAULT_PROBLEMS_FILE = DATA_DIR / "problems.jsonl"


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plots_dir_for_run(run_id: str) -> Path:
    return PLOTS_DIR / run_id


# Order doubles as the tie-break priority for role assignment.
MODEL_KEYS = ["gpt", "claude", "gemini", "grok"]

MODEL_DISPLAY = {
    "gpt": "GPT",
    "claude": "Claude",
    "gemini": "Gemini",
    "grok": "Grok",
}

SOLVER_TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.1
ROLE_ASSESS_TEMPERATURE = 0.2

DEFAULT_MAX_TOKENS = 1500


def get_env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)
