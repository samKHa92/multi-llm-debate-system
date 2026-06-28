"""Central configuration: paths, model registry, and defaults.

Loads environment variables from a local ``.env`` file (if present) using
python-dotenv. No API keys are ever hardcoded here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root if it exists. This is a no-op when missing,
# which keeps mock mode working with zero configuration.
load_dotenv()

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RUNS_DIR = OUTPUTS_DIR / "runs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_CSV = OUTPUTS_DIR / "results.csv"
DEFAULT_PROBLEMS_FILE = DATA_DIR / "problems.jsonl"


def ensure_dirs() -> None:
    """Create output directories if they do not exist yet."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# The four logical models. The "key" is used everywhere in the pipeline; the
# tie-break priority for role assignment follows this exact order.
MODEL_KEYS = ["gpt", "claude", "gemini", "grok"]

# Human-friendly display names (used in plots / logs).
MODEL_DISPLAY = {
    "gpt": "GPT",
    "claude": "Claude",
    "gemini": "Gemini",
    "grok": "Grok",
}

# Default temperatures. Solvers are slightly more creative than the judge.
SOLVER_TEMPERATURE = 0.3
JUDGE_TEMPERATURE = 0.1
ROLE_ASSESS_TEMPERATURE = 0.2

DEFAULT_MAX_TOKENS = 1500


def get_env(key: str, default: str | None = None) -> str | None:
    """Thin wrapper around os.getenv (single place to mock/patch in tests)."""
    return os.getenv(key, default)
