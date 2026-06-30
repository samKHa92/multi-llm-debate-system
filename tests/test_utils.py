"""Tests for the shared helpers in src/utils.py."""

import json

from src.utils import (
    coerce_float,
    extract_json,
    load_json,
    load_jsonl,
    load_problems,
    majority_vote,
    save_json,
)


def test_extract_json_direct_object():
    assert extract_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_extract_json_from_markdown_fence():
    text = 'Here you go:\n```json\n{"winner": "solver_2"}\n```\nThanks!'
    assert extract_json(text) == {"winner": "solver_2"}


def test_extract_json_from_surrounding_prose():
    text = 'The answer is {"final_answer": "42"} for sure.'
    assert extract_json(text) == {"final_answer": "42"}


def test_extract_json_handles_nested_braces():
    text = 'noise {"a": {"b": 1}} trailing'
    assert extract_json(text) == {"a": {"b": 1}}


def test_extract_json_returns_empty_on_garbage():
    assert extract_json("") == {}
    assert extract_json("no json here") == {}
    assert extract_json("{not valid json}") == {}


def test_extract_json_ignores_non_object_json():
    # A bare list or scalar is valid JSON but not a dict -> {}.
    assert extract_json("[1, 2, 3]") == {}
    assert extract_json("5") == {}


def test_coerce_float_passthrough_and_clamp():
    assert coerce_float("0.8") == 0.8
    assert coerce_float(1.0) == 1.0
    assert coerce_float(150) == 1.0  # clamped to [0, 1]
    assert coerce_float(-3) == 0.0


def test_coerce_float_percentage_rescale():
    assert coerce_float(85) == 0.85
    assert coerce_float("90") == 0.90


def test_coerce_float_uses_default_on_bad_input():
    assert coerce_float(None) == 0.5
    assert coerce_float("abc") == 0.5
    assert coerce_float("abc", default=0.1) == 0.1


def test_majority_vote_full_consensus():
    assert majority_vote(["4", "4", "4"]) == ("4", "full")


def test_majority_vote_partial_consensus():
    assert majority_vote(["4", "4", "5"]) == ("4", "partial")


def test_majority_vote_no_majority():
    assert majority_vote(["4", "5", "6"]) == ("", "none")
    assert majority_vote(["a", "b"]) == ("", "none")


def test_majority_vote_empty_input():
    assert majority_vote([]) == ("", "none")
    assert majority_vote(["", None, "  "]) == ("", "none")


def test_majority_vote_filters_blanks_then_decides():
    assert majority_vote(["4", "", None, "4"]) == ("4", "full")


def test_majority_vote_returns_first_original_form():
    answer, consensus = majority_vote(["Yes", "yes", "no"])
    assert answer == "Yes"  # original casing of the first winning occurrence
    assert consensus == "partial"


def test_majority_vote_custom_normalizer():
    # Normalize on first character only -> "apple" and "avocado" collide.
    answer, consensus = majority_vote(
        ["apple", "avocado", "banana"], normalizer=lambda x: str(x)[0]
    )
    assert answer == "apple"
    assert consensus == "partial"


def test_json_file_round_trip(tmp_path):
    payload = {"id": "p1", "values": [1, 2, 3], "unicode": "café"}
    path = tmp_path / "nested" / "data.json"
    save_json(path, payload)
    assert path.exists()
    assert load_json(path) == payload


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert load_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_load_problems_parses_into_models(tmp_path):
    path = tmp_path / "problems.jsonl"
    path.write_text(
        '{"id": "p1", "category": "logic", "question": "q?", '
        '"correct_answer": "yes", "answer_type": "short_text"}\n',
        encoding="utf-8",
    )
    problems = load_problems(path)
    assert len(problems) == 1
    assert problems[0].id == "p1"
    assert problems[0].correct_answer == "yes"
