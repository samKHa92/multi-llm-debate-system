"""Pydantic schemas shared across the pipeline."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class Problem(BaseModel):
    id: str
    category: str
    question: str
    correct_answer: str
    accepted_answers: list[str] = Field(default_factory=list)
    answer_type: str = "short_text"  # integer | float | multiple_choice | short_text
    difficulty: str = "hard"
    tolerance: Optional[float] = None  # used when answer_type == "float"
    source_or_notes: str = ""


class RoleSelfAssessment(BaseModel):
    model_key: str
    solver_confidence: float = 0.5
    judge_confidence: float = 0.5
    reasoning: str = ""


class AssignedRoles(BaseModel):
    judge: str
    solvers: list[str]
    solver_slots: dict[str, str] = Field(default_factory=dict)
    assessments: list[RoleSelfAssessment] = Field(default_factory=list)


class SolverSolution(BaseModel):
    solver_id: str
    model_key: str
    reasoning: str = ""
    final_answer: str = ""
    confidence: float = 0.5


class ReviewError(BaseModel):
    location: str = ""
    error_type: str = ""
    description: str = ""
    severity: str = "minor"  # minor | major | critical


class PeerReview(BaseModel):
    reviewer_id: str
    target_id: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    errors: list[ReviewError] = Field(default_factory=list)
    suggested_changes: list[str] = Field(default_factory=list)
    overall_assessment: str = ""


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


class JudgeDecision(BaseModel):
    judge_model_key: str
    winner: str
    confidence: float = 0.5
    reasoning: str = ""


class ProblemRunResult(BaseModel):
    run_id: str
    problem: Problem
    assigned_roles: AssignedRoles
    initial_solutions: list[SolverSolution] = Field(default_factory=list)
    peer_reviews: list[PeerReview] = Field(default_factory=list)
    refinements: list[RefinementResult] = Field(default_factory=list)
    judge_decision: JudgeDecision
    debate_final_answer: str = ""
    single_model_answers: dict[str, str] = Field(default_factory=dict)


class EvaluationRow(BaseModel):
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
    single_model_correct: dict[str, bool] = Field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        single = data.pop("single_model_correct", {}) or {}
        for model_key, ok in single.items():
            data[f"single_{model_key}_correct"] = ok
        return data
