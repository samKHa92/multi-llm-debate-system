"""Tests for answer normalization and automatic correctness checking."""

from src.models import Problem
from src.pipeline.answer_extraction import is_correct, normalize_answer


def test_normalize_strips_noise():
    assert normalize_answer("  153. ") == "153"
    assert normalize_answer("$1,000") == "1000"
    assert normalize_answer("Knave") == "knave"
    assert normalize_answer(None) == ""


def test_integer_checking():
    p = Problem(id="i", category="c", question="q", correct_answer="153", answer_type="integer")
    assert is_correct("153", p)
    assert is_correct("153.0", p)
    assert is_correct(" 153 ", p)
    assert not is_correct("154", p)
    assert not is_correct("", p)


def test_float_checking_with_tolerance():
    p = Problem(
        id="f", category="c", question="q", correct_answer="0.289",
        answer_type="float", tolerance=0.005,
    )
    assert is_correct("0.289", p)
    assert is_correct("0.288", p)
    assert not is_correct("0.30", p)


def test_multiple_choice_checking():
    p = Problem(
        id="mc", category="c", question="q", correct_answer="B",
        answer_type="multiple_choice", accepted_answers=["B"],
    )
    assert is_correct("B", p)
    assert is_correct("b) bid your true value", p)
    assert not is_correct("A", p)


def test_short_text_with_accepted_answers():
    p = Problem(
        id="st", category="c", question="q", correct_answer="defect",
        answer_type="short_text", accepted_answers=["defect", "defection", "betray"],
    )
    assert is_correct("Defect", p)
    assert is_correct("betray", p)
    assert not is_correct("cooperate", p)
