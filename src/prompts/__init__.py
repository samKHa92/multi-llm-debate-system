"""Prompt builders for each stage of the debate."""

from .role_assignment import build_role_assignment_prompt
from .solver import build_solver_prompt
from .peer_review import build_peer_review_prompt
from .refinement import build_refinement_prompt
from .judge import build_judge_prompt

__all__ = [
    "build_role_assignment_prompt",
    "build_solver_prompt",
    "build_peer_review_prompt",
    "build_refinement_prompt",
    "build_judge_prompt",
]
