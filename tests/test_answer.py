"""Answer extraction, normalisation and equivalence."""

from __future__ import annotations

import pytest

from cot_faithcheck.answer import answers_equivalent, extract_answer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The answer is 42.", "42"),
        ("Final answer: B", "B"),
        ("... so Answer: yes", "yes"),
        (r"We compute \boxed{17}.", "17"),
        ("Blah blah.\n(C)", "C"),
        ("I think the option B is correct.", "B"),  # "option X" resolves to the letter
        ("", ""),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_extract_prefers_last_marker():
    text = "The answer is 3. Wait, let me redo it. The answer is 5."
    assert extract_answer(text) == "5"


@pytest.mark.parametrize(
    "a,b,equal",
    [
        ("42", "42.0", True),
        ("$42", "42", True),
        ("B", "b", True),
        ("B", "option B", True),
        ("yes", "no", False),
        ("42", "43", False),
        ("", "x", False),
    ],
)
def test_answers_equivalent(a, b, equal):
    assert answers_equivalent(a, b) is equal


def test_numeric_tolerance():
    assert answers_equivalent("1.0000001", "1.0", numeric_tol=1e-3)
    assert not answers_equivalent("1.1", "1.0", numeric_tol=1e-3)
