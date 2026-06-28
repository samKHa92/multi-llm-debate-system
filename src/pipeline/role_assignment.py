"""Stage 0 + 0.5: collect self-assessments and assign roles deterministically.

Algorithm (Stage 0.5):
  1. Ask all 4 models for Solver/Judge confidence (Stage 0).
  2. The model with the highest *judge* confidence becomes the Final Judge.
  3. The other three become Solvers (slots solver_1/2/3).
  4. Ties are broken by the fixed priority order: gpt > claude > gemini > grok.
"""

from __future__ import annotations

from ..config import MODEL_KEYS, ROLE_ASSESS_TEMPERATURE
from ..llm_clients.base import BaseLLMClient
from ..models import AssignedRoles, Problem, RoleSelfAssessment
from ..prompts import build_role_assignment_prompt
from ..utils import coerce_float, extract_json

# Lower index = higher priority for tie-breaks.
_PRIORITY = {key: i for i, key in enumerate(MODEL_KEYS)}


def collect_self_assessments(
    clients: dict[str, BaseLLMClient],
    problem: Problem,
) -> list[RoleSelfAssessment]:
    """Run Stage 0 for every client and parse the results."""
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
    """Pure, deterministic Stage 0.5 assignment (easy to unit-test)."""
    if not assessments:
        raise ValueError("Cannot assign roles without any self-assessments.")

    # Pick the judge: highest judge_confidence, tie-break by fixed priority.
    def judge_sort_key(a: RoleSelfAssessment) -> tuple[float, int]:
        # Negative confidence so that higher confidence sorts first; lower
        # priority index also sorts first.
        return (-a.judge_confidence, _PRIORITY.get(a.model_key, 99))

    ordered = sorted(assessments, key=judge_sort_key)
    judge_key = ordered[0].model_key

    # Remaining models become solvers, kept in the fixed priority order so the
    # solver slots are stable and reproducible.
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
