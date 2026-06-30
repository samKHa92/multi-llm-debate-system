"""Tests for plot generation (rendered with the headless Agg backend)."""

from src.plotting import generate_all_plots

_METRICS = {
    "num_problems": 4,
    "overall_accuracy": 0.75,
    "improvement_rate": 0.5,
    "consensus_rate": 0.25,
    "judge_accuracy_when_disagreement": 0.66,
    "num_disagreement_problems": 3,
    "single_llm_baseline_accuracy": 0.6,
    "single_llm_accuracy_by_model": {"gpt": 0.7, "claude": 0.6, "gemini": 0.55, "grok": 0.5},
    "simple_voting_baseline_accuracy": 0.65,
    "full_debate_system_accuracy": 0.75,
    "debate_accuracy_by_category": {"math": 0.8, "logic": 0.7},
}

_EXPECTED = {
    "accuracy_comparison.png",
    "accuracy_by_category.png",
    "refinement_improvement.png",
    "consensus_rate.png",
    "judge_accuracy_disagreement.png",
}


def test_generate_all_plots_writes_five_files(tmp_path):
    paths = generate_all_plots(_METRICS, tmp_path)
    assert len(paths) == 5
    assert {p.name for p in paths} == _EXPECTED
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0


def test_generate_all_plots_creates_missing_dir(tmp_path):
    out_dir = tmp_path / "nested" / "run_42"
    paths = generate_all_plots(_METRICS, out_dir)
    assert out_dir.is_dir()
    assert all(p.parent == out_dir for p in paths)


def test_generate_all_plots_handles_missing_metric_keys(tmp_path):
    # Builders use metrics.get(...) with defaults, so a sparse dict must not crash.
    paths = generate_all_plots({}, tmp_path)
    assert len(paths) == 5
    assert all(p.exists() for p in paths)
