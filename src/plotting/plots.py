"""Generate the five evaluation figures with matplotlib."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402

from ..config import MODEL_DISPLAY, PLOTS_DIR  # noqa: E402


def _annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _save(fig, out_dir: Path, name: str) -> Path:
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_accuracy_comparison(metrics: dict, out_dir: Path) -> Path:
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    for key, acc in metrics.get("single_llm_accuracy_by_model", {}).items():
        labels.append(MODEL_DISPLAY.get(key, key))
        values.append(acc)
        colors.append("#9ecae1")

    labels.append("Voting")
    values.append(metrics.get("simple_voting_baseline_accuracy", 0.0))
    colors.append("#fdae6b")

    labels.append("Debate\n(ours)")
    values.append(metrics.get("full_debate_system_accuracy", 0.0))
    colors.append("#31a354")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors)
    _annotate_bars(ax, bars)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy: Single-LLM vs Voting vs Debate System")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _save(fig, out_dir, "accuracy_comparison.png")


def plot_accuracy_by_category(metrics: dict, out_dir: Path) -> Path:
    by_cat = metrics.get("debate_accuracy_by_category", {})
    cats = list(by_cat.keys())
    vals = [by_cat[c] for c in cats]
    pretty = [c.replace("_", "\n") for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(pretty, vals, color="#756bb1")
    _annotate_bars(ax, bars)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Debate accuracy")
    ax.set_title("Debate System Accuracy by Category")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _save(fig, out_dir, "accuracy_by_category.png")


def plot_refinement_improvement(metrics: dict, out_dir: Path) -> Path:
    labels = ["Single-LLM", "Voting", "Debate"]
    values = [
        metrics.get("single_llm_baseline_accuracy", 0.0),
        metrics.get("simple_voting_baseline_accuracy", 0.0),
        metrics.get("full_debate_system_accuracy", 0.0),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    bars1 = ax1.bar(labels, values, color=["#9ecae1", "#fdae6b", "#31a354"])
    _annotate_bars(ax1, bars1)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("System Accuracy")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    imp = metrics.get("improvement_rate", 0.0)
    bars2 = ax2.bar(["Improvement rate"], [imp], color="#e6550d", width=0.5)
    _annotate_bars(ax2, bars2)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Fraction of problems")
    ax2.set_title("Problems Improved After Refinement")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Effect of Peer Review + Refinement")
    return _save(fig, out_dir, "refinement_improvement.png")


def plot_consensus_rate(metrics: dict, out_dir: Path) -> Path:
    consensus = metrics.get("consensus_rate", 0.0)
    fig, ax = plt.subplots(figsize=(5, 5))
    sizes = [consensus, max(0.0, 1 - consensus)]
    ax.pie(
        sizes,
        labels=["All 3 solvers agree", "Solvers disagree"],
        autopct="%1.0f%%",
        colors=["#31a354", "#de2d26"],
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    ax.set_title("Solver Consensus Rate (initial answers)")
    return _save(fig, out_dir, "consensus_rate.png")


def plot_judge_accuracy_disagreement(metrics: dict, out_dir: Path) -> Path:
    judge_acc = metrics.get("judge_accuracy_when_disagreement")
    judge_acc = 0.0 if judge_acc is None else judge_acc
    n_disagree = metrics.get("num_disagreement_problems", 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Judge correct\nwhen solvers disagree"], [judge_acc], color="#3182bd", width=0.5)
    _annotate_bars(ax, bars)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Judge Accuracy on Disagreement\n(n = {n_disagree} problems)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    return _save(fig, out_dir, "judge_accuracy_disagreement.png")


_PLOTTERS = (
    plot_accuracy_comparison,
    plot_accuracy_by_category,
    plot_refinement_improvement,
    plot_consensus_rate,
    plot_judge_accuracy_disagreement,
)


def generate_all_plots(metrics: dict, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return [plotter(metrics, out_dir) for plotter in _PLOTTERS]
