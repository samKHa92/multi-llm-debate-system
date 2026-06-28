"""A deterministic, offline mock LLM client.

Why this exists
---------------
The whole point of the assignment is the *pipeline*. We must be able to run
the entire 4-stage debate end-to-end with **no API keys and no cost**, both
for unit tests and for generating example plots.

This mock does three things:
  1. Detects which stage it is being asked about (a ``STAGE:`` line that every
     prompt builder includes) and returns valid JSON for that stage.
  2. Is fully deterministic: identical inputs always yield identical output
     (driven by SHA-256 hashing), so tests are stable.
  3. *Simulates* believable behaviour. Each mock "model" has a skill level and
     an improvement-after-feedback tendency, so the debate system measurably
     beats the single-model and voting baselines - exactly the story the plots
     are meant to tell. To do this honestly the mock is given the answer key
     (``answer_key``); a real client never sees correct answers.

Real provider clients deliberately do NOT inherit any of this behaviour.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

from .base import BaseLLMClient


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


class MockLLMClient(BaseLLMClient):
    def __init__(
        self,
        name: str,
        model_name: str = "mock-model",
        answer_key: Optional[dict[str, dict[str, Any]]] = None,
        skill: float = 0.6,
        improve_skill: float = 0.5,
        judge_skill: float = 0.8,
    ) -> None:
        super().__init__(name=name, model_name=model_name)
        # problem_id -> {"correct": str, "answer_type": str}
        self.answer_key = answer_key or {}
        self.skill = skill  # P(correct) for a fresh solve
        self.improve_skill = improve_skill  # P(wrong -> correct) after feedback
        self.judge_skill = judge_skill  # P(judge picks a correct solver)

    # ------------------------------------------------------------------
    # Deterministic pseudo-randomness
    # ------------------------------------------------------------------
    def _rand(self, *parts: Any) -> float:
        """Stable float in [0, 1) derived from the inputs."""
        raw = "|".join(str(p) for p in parts)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def _bool(self, prob: float, *parts: Any) -> bool:
        return self._rand(*parts) < prob

    # ------------------------------------------------------------------
    # Prompt parsing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _field(prompt: str, label: str) -> str:
        m = re.search(rf"{re.escape(label)}\s*:\s*(.+)", prompt)
        return m.group(1).strip() if m else ""

    def _stage(self, prompt: str) -> str:
        return self._field(prompt, "STAGE").lower()

    def _problem_id(self, prompt: str) -> str:
        return self._field(prompt, "Problem ID")

    def _correct_answer(self, problem_id: str) -> str:
        return str(self.answer_key.get(problem_id, {}).get("correct", ""))

    def _answer_type(self, problem_id: str) -> str:
        return str(self.answer_key.get(problem_id, {}).get("answer_type", "short_text"))

    # ------------------------------------------------------------------
    # Wrong-answer fabrication (varied but deterministic)
    # ------------------------------------------------------------------
    def _wrong_answer(self, correct: str, answer_type: str, salt: str) -> str:
        offset = 1 + int(self._rand("offset", salt) * 4)  # 1..4
        if answer_type == "integer":
            try:
                return str(int(float(correct)) + offset)
            except ValueError:
                return f"{correct}_x"
        if answer_type == "float":
            try:
                return str(round(float(correct) + offset * 0.5, 3))
            except ValueError:
                return f"{correct}_x"
        if answer_type == "multiple_choice":
            letters = ["A", "B", "C", "D"]
            cur = _normalize(correct).upper()[:1] or "A"
            choices = [l for l in letters if l != cur]
            return choices[offset % len(choices)]
        return f"alternative_{offset}"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
        stage = self._stage(prompt)
        if stage == "role_assignment":
            return self._gen_role_assignment(prompt)
        if stage == "solver":
            return self._gen_solver(prompt)
        if stage == "single":
            return self._gen_single(prompt)
        if stage == "peer_review":
            return self._gen_peer_review(prompt)
        if stage == "refinement":
            return self._gen_refinement(prompt)
        if stage == "judge":
            return self._gen_judge(prompt)
        # Fallback: still valid JSON so callers never crash.
        return json.dumps({"answer": "mock"})

    # ------------------------------------------------------------------
    # Per-stage generators
    # ------------------------------------------------------------------
    def _gen_role_assignment(self, prompt: str) -> str:
        pid = self._problem_id(prompt)
        # Confidences vary by model so the judge choice (and tie-breaks) differ
        # across problems, but are deterministic.
        solver_conf = round(0.5 + 0.45 * self._rand("solver_conf", pid, self.name), 2)
        judge_conf = round(0.4 + 0.55 * self._rand("judge_conf", pid, self.name), 2)
        return json.dumps(
            {
                "solver_confidence": solver_conf,
                "judge_confidence": judge_conf,
                "reasoning": (
                    f"As {self.name}, I weighed my strengths for problem {pid}. "
                    f"Solver confidence {solver_conf}, judge confidence {judge_conf}."
                ),
            }
        )

    def _solve_answer(self, pid: str, salt: str, prob: float) -> tuple[str, bool]:
        """Return (answer, is_correct) for a solving-type stage."""
        correct = self._correct_answer(pid)
        if not correct:
            # No answer key (e.g. an unknown problem): emit something stable.
            return ("mock_answer", False)
        is_ok = self._bool(prob, salt, pid, self.name)
        if is_ok:
            return (correct, True)
        return (self._wrong_answer(correct, self._answer_type(pid), f"{salt}{self.name}"), False)

    def _gen_solver(self, prompt: str) -> str:
        pid = self._problem_id(prompt)
        slot = self._field(prompt, "Your solver slot") or "solver_1"
        answer, ok = self._solve_answer(pid, "solve", self.skill)
        conf = round(0.6 + 0.3 * self._rand("solve_conf", pid, self.name), 2)
        return json.dumps(
            {
                "reasoning": (
                    f"Step 1: parse the problem. Step 2: apply the relevant method. "
                    f"Step 3: compute and sanity-check. ({self.name}/{slot})"
                ),
                "final_answer": answer,
                "confidence": conf,
                "_mock_correct": ok,  # harmless extra field; ignored by parser
            }
        )

    def _gen_single(self, prompt: str) -> str:
        # Single-model baseline: one shot, no debate. Slightly lower skill to
        # reflect "no second chance".
        pid = self._problem_id(prompt)
        answer, _ = self._solve_answer(pid, "single", max(0.0, self.skill - 0.05))
        return json.dumps({"final_answer": answer})

    def _gen_peer_review(self, prompt: str) -> str:
        reviewer = self._field(prompt, "Reviewer") or "solver_1"
        target = self._field(prompt, "Target") or "solver_2"
        pid = self._problem_id(prompt)
        flag_error = self._bool(0.5, "review_err", pid, reviewer, target)
        errors = (
            [
                {
                    "location": "Step 3",
                    "error_type": "logical_error",
                    "description": "A transition between steps is not fully justified.",
                    "severity": "major",
                }
            ]
            if flag_error
            else []
        )
        return json.dumps(
            {
                "strengths": ["Clear setup", "Reasonable method"],
                "weaknesses": ["Could double-check arithmetic"]
                + (["Unjustified leap in the middle"] if flag_error else []),
                "errors": errors,
                "suggested_changes": ["Verify the final computation", "Consider edge cases"],
                "overall_assessment": "promising_but_flawed" if flag_error else "solid",
            }
        )

    def _gen_refinement(self, prompt: str) -> str:
        pid = self._problem_id(prompt)
        slot = self._field(prompt, "Your solver slot") or "solver_1"
        # Recompute our own initial correctness deterministically.
        _, was_correct = self._solve_answer(pid, "solve", self.skill)
        if was_correct:
            # Usually keep the correct answer (high stability).
            stays = self._bool(0.92, "refine_keep", pid, self.name)
            refined_answer, ok = (
                self._solve_answer(pid, "solve", self.skill)
                if stays
                else (self._wrong_answer(self._correct_answer(pid), self._answer_type(pid), f"flip{self.name}"), False)
            )
        else:
            # Feedback gives a chance to fix a wrong initial answer.
            fixed = self._bool(self.improve_skill, "refine_fix", pid, self.name)
            if fixed:
                refined_answer, ok = (self._correct_answer(pid), True)
            else:
                refined_answer, ok = self._solve_answer(pid, "solve", self.skill)
        conf = round(0.65 + 0.3 * self._rand("refine_conf", pid, self.name), 2)
        return json.dumps(
            {
                "changes_made": [
                    {"critique": "Verify the final computation", "response": "Re-derived and confirmed.", "accepted": True},
                    {"critique": "Consider edge cases", "response": "Edge case does not apply here.", "accepted": False},
                ],
                "refined_solution": f"Refined reasoning for {slot} ({self.name}).",
                "refined_answer": refined_answer,
                "confidence": conf,
                "_mock_correct": ok,
            }
        )

    def _gen_judge(self, prompt: str) -> str:
        pid = self._problem_id(prompt)
        correct = _normalize(self._correct_answer(pid))
        # Parse the refined answers the judge was shown.
        refined: dict[str, str] = {}
        for m in re.finditer(r"(solver_\d)\s+refined answer\s*:\s*(.+)", prompt):
            refined[m.group(1)] = m.group(2).strip()
        slots = sorted(refined) or ["solver_1", "solver_2", "solver_3"]

        correct_slots = [s for s in slots if correct and _normalize(refined.get(s, "")) == correct]

        if correct_slots and self._bool(self.judge_skill, "judge_pick", pid, self.name):
            # Judge correctly identifies a right answer.
            winner = correct_slots[0]
        else:
            # Judge errs (or there is no correct answer to find): pick deterministically.
            idx = int(self._rand("judge_else", pid, self.name) * len(slots)) % len(slots)
            winner = slots[idx]

        conf = round(0.6 + 0.35 * self._rand("judge_conf", pid, self.name), 2)
        return json.dumps(
            {
                "winner": winner,
                "confidence": conf,
                "reasoning": f"{winner} presents the most rigorous and well-verified solution.",
            }
        )
