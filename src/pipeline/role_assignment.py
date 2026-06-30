"""Stage 0 + 0.5: collect self-assessments and assign roles deterministically."""

from __future__ import annotations

from ..config import MODEL_KEYS, ROLE_ASSESS_TEMPERATURE
from ..llm_clients.base import BaseLLMClient
from ..models import AssignedRoles, Problem, RoleSelfAssessment
from ..prompts import build_role_assignment_prompt
from ..utils import coerce_float, extract_json

_PRIORITY = {key: i for i, key in enumerate(MODEL_KEYS)}


def collect_self_assessments(
    clients: dict[str, BaseLLMClient],
    problem: Problem,
) -> list[RoleSelfAssessment]:
    assessments: list[RoleSelfAssessment] = []
    prompt = build_role_assignment_prompt(problem)
    for key, client in clients.items():
        raw = client.generate(prompt, temperature=ROLE_ASSESS_TEMPERATURE)
        data = extract_json(raw)
        assessments.append(
            RoleSelfAssessment(
                model_key=key,
                solver_confidence=coerce_float(data.get("solver_confidence"), 0.5),
                judge_confidence=coerce_float(data.get("judge_confidence"), 0.5),
                reasoning=str(data.get("reasoning", "")),
            )
        )
    return assessments


def assign_roles(assessments: list[RoleSelfAssessment]) -> AssignedRoles:
    if not assessments:
        raise ValueError("Cannot assign roles without any self-assessments.")

    # Judge = highest judge_confidence, ties broken by fixed priority.
    def judge_rank(a: RoleSelfAssessment) -> tuple[float, int]:
        return (-a.judge_confidence, _PRIORITY.get(a.model_key, 99))

    judge_key = min(assessments, key=judge_rank).model_key

    solver_keys = sorted(
        (a.model_key for a in assessments if a.model_key != judge_key),
        key=lambda k: _PRIORITY.get(k, 99),
    )
    solver_slots = {f"solver_{i + 1}": key for i, key in enumerate(solver_keys)}

    return AssignedRoles(
        judge=judge_key,
        solvers=solver_keys,
        solver_slots=solver_slots,
        assessments=assessments,
    )
