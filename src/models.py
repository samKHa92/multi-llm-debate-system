"""Pydantic schemas used across the whole pipeline.

These models give us:
  * validation of LLM output (so malformed JSON is caught early),
  * a single source of truth for the data shapes,
  * easy (de)serialization to/from JSON for caching runs.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Problem(BaseModel):
    """A single challenge problem from the dataset."""

    id: str
    category: str
    question: str
    correct_answer: str
    # All answers that should be treated as correct (synonyms / equivalent forms).
    accepted_answers: list[str] = Field(default_factory=list)
    # One of: integer | float | multiple_choice | short_text
    answer_type: str = "short_text"
    difficulty: str = "hard"
    # Optional numeric tolerance used when answer_type == "float".
    tolerance: Optional[float] = None
    source_or_notes: str = ""


# ---------------------------------------------------------------------------
# Stage 0: role self-assessment
# ---------------------------------------------------------------------------
class RoleSelfAssessment(BaseModel):
    """How a single model rates itself for this problem."""

    model_key: str  # gpt | claude | gemini | grok
    solver_confidence: float = 0.5
    judge_confidence: float = 0.5
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Stage 0.5: deterministic role assignment
# ---------------------------------------------------------------------------
class AssignedRoles(BaseModel):
    """Result of the deterministic role assignment algorithm."""

    judge: str  # model_key chosen as judge
    solvers: list[str]  # the three model_keys acting as solvers
    # solver slot id (solver_1/2/3) -> model_key, kept for traceability
    solver_slots: dict[str, str] = Field(default_factory=dict)
    assessments: list[RoleSelfAssessment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 1: independent solver solutions
# ---------------------------------------------------------------------------
class SolverSolution(BaseModel):
    solver_id: str  # solver_1 | solver_2 | solver_3
    model_key: str  # which model produced it
    reasoning: str = ""
    final_answer: str = ""
    confidence: float = 0.5


# ---------------------------------------------------------------------------
# Stage 2: peer review
# ---------------------------------------------------------------------------
class ReviewError(BaseModel):
    location: str = ""
    error_type: str = ""
    description: str = ""
    severity: str = "minor"  # minor | major | critical


class PeerReview(BaseModel):
    reviewer_id: str  # the solver writing the review
    target_id: str  # the solver being reviewed
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    errors: list[ReviewError] = Field(default_factory=list)
    suggested_changes: list[str] = Field(default_factory=list)
    overall_assessment: str = ""


# ---------------------------------------------------------------------------
# Stage 3: refinement
# ---------------------------------------------------------------------------
class CritiqueResponse(BaseModel):
    critique: str = ""
    response: str = ""
    accepted: bool = False


class RefinementResult(BaseModel):
    solver_id: str
    model_key: str
    changes_made: list[CritiqueResponse] = Field(default_factory=list)
    refined_solution: str = ""
    refined_answer: str = ""
    confidence: float = 0.5


# ---------------------------------------------------------------------------
# Stage 4: final judgment
# ---------------------------------------------------------------------------
class JudgeDecision(BaseModel):
    judge_model_key: str
    winner: str  # solver_1 | solver_2 | solver_3
    confidence: float = 0.5
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Full per-problem trace
# ---------------------------------------------------------------------------
class ProblemRunResult(BaseModel):
    """Everything produced for one problem, saved to outputs/runs/."""

    run_id: str
    problem: Problem
    assigned_roles: AssignedRoles
    initial_solutions: list[SolverSolution] = Field(default_factory=list)
    peer_reviews: list[PeerReview] = Field(default_factory=list)
    refinements: list[RefinementResult] = Field(default_factory=list)
    judge_decision: JudgeDecision
    debate_final_answer: str = ""
    # Single-model baseline answers, one per model_key.
    single_model_answers: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class EvaluationRow(BaseModel):
    """One row of outputs/results.csv."""

    problem_id: str
    category: str
    correct_answer: str
    solver_1_initial_answer: str = ""
    solver_2_initial_answer: str = ""
    solver_3_initial_answer: str = ""
    solver_1_refined_answer: str = ""
    solver_2_refined_answer: str = ""
    solver_3_refined_answer: str = ""
    judge_winner: str = ""
    debate_final_answer: str = ""
    debate_correct: bool = False
    voting_answer: str = ""
    voting_correct: bool = False
    consensus_type: str = ""  # full | partial | none
    improved_after_refinement: bool = False
    judge_correct_when_disagreement: Optional[bool] = None
    # Single-model baseline correctness (best-effort, one column per model).
    single_model_correct: dict[str, bool] = Field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten nested fields so pandas writes a clean CSV."""
        data = self.model_dump()
        single = data.pop("single_model_correct", {}) or {}
        for model_key, ok in single.items():
            data[f"single_{model_key}_correct"] = ok
        return data
