"""Tests for the deterministic Stage 0.5 role assignment algorithm."""

from src.models import RoleSelfAssessment
from src.pipeline.role_assignment import assign_roles


def _assess(model_key, judge_conf, solver_conf=0.5):
    return RoleSelfAssessment(
        model_key=model_key,
        solver_confidence=solver_conf,
        judge_confidence=judge_conf,
    )


def test_highest_judge_confidence_becomes_judge():
    assessments = [
        _assess("gpt", 0.40),
        _assess("claude", 0.90),  # clearly the best judge
        _assess("gemini", 0.50),
        _assess("grok", 0.60),
    ]
    roles = assign_roles(assessments)
    assert roles.judge == "claude"
    assert set(roles.solvers) == {"gpt", "gemini", "grok"}
    assert len(roles.solver_slots) == 3


def test_tie_break_uses_fixed_priority_order():
    # All tied on judge confidence -> priority gpt > claude > gemini > grok.
    assessments = [
        _assess("grok", 0.70),
        _assess("gemini", 0.70),
        _assess("claude", 0.70),
        _assess("gpt", 0.70),
    ]
    roles = assign_roles(assessments)
    assert roles.judge == "gpt"


def test_solver_slots_follow_priority_order():
    assessments = [
        _assess("gpt", 0.95),  # gpt is judge
        _assess("grok", 0.30),
        _assess("claude", 0.40),
        _assess("gemini", 0.20),
    ]
    roles = assign_roles(assessments)
    assert roles.judge == "gpt"
    # Remaining solvers ordered by priority: claude, gemini, grok.
    assert roles.solver_slots == {
        "solver_1": "claude",
        "solver_2": "gemini",
        "solver_3": "grok",
    }


def test_deterministic_repeatable():
    assessments = [
        _assess("gpt", 0.5),
        _assess("claude", 0.8),
        _assess("gemini", 0.8),  # tie with claude -> claude wins by priority
        _assess("grok", 0.1),
    ]
    first = assign_roles(assessments)
    second = assign_roles(assessments)
    assert first.judge == second.judge == "claude"
