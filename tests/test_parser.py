"""Trace parsing and reasoning-step splitting."""

from __future__ import annotations

import json

import pytest

from cot_faithcheck import load_trace, load_traces, parse_trace, split_reasoning
from cot_faithcheck.errors import NoReasoningStepsError, TraceParseError


def test_explicit_steps(math_trace):
    assert len(math_trace.steps) == 3
    assert math_trace.steps[0].text.startswith("Start with 12")
    assert math_trace.final_answer == "25"
    assert math_trace.gold_answer == "25"
    assert [s.index for s in math_trace.steps] == [0, 1, 2]


def test_steps_from_dicts():
    trace = parse_trace(
        {
            "question": "q",
            "steps": [{"text": "one"}, {"reasoning": "two"}, "three"],
            "answer": "x",
        }
    )
    assert [s.text for s in trace.steps] == ["one", "two", "three"]


def test_blob_inline_step_markers(blob_trace):
    # A single-line "Step 1: ... Step 2: ..." blob splits into four steps.
    assert len(blob_trace.steps) == 4
    assert blob_trace.steps[-1].text.lower().startswith("the cell")


def test_split_reasoning_variants():
    assert split_reasoning("- a\n- b\n- c") == ["a", "b", "c"]
    numbered = split_reasoning("Step 1: alpha\nStep 2: beta")
    assert numbered == ["alpha", "beta"]
    sentences = split_reasoning("First fact. Second fact. Third fact.")
    assert len(sentences) == 3


def test_key_aliases():
    trace = parse_trace({"query": "q", "chain_of_thought": "Only one line.", "prediction": "yes"})
    assert trace.question == "q"
    assert trace.final_answer == "yes"


def test_missing_question_raises():
    with pytest.raises(TraceParseError):
        parse_trace({"steps": ["a"], "answer": "b"})


def test_missing_answer_raises():
    with pytest.raises(TraceParseError):
        parse_trace({"question": "q", "steps": ["a"]})


def test_no_reasoning_raises():
    with pytest.raises(NoReasoningStepsError):
        parse_trace({"question": "q", "answer": "b"})


def test_options_from_list():
    trace = parse_trace(
        {
            "question": "q",
            "steps": ["s"],
            "answer": "A",
            "options": ["A) first", "B) second"],
        }
    )
    assert trace.options == {"A": "first", "B": "second"}


def test_load_trace_from_json_string():
    payload = json.dumps({"question": "q", "steps": ["a", "b"], "answer": "c"})
    trace = load_trace(payload)
    assert trace.final_answer == "c"


def test_load_traces_array(fixtures_dir):
    traces = load_traces(str(fixtures_dir / "batch.json"))
    assert [t.trace_id for t in traces] == ["batch-a", "batch-b"]


def test_intervenable_band_small_trace(math_trace):
    # With <= 3 steps every step is intervenable.
    assert all(s.is_intervenable for s in math_trace.steps)


def test_intervenable_band_large_trace():
    steps = [f"Step reasoning {i}." for i in range(10)]
    trace = parse_trace({"question": "q", "steps": steps, "answer": "z"})
    intervenable = [s.index for s in trace.intervenable_steps()]
    # The middle 30%-90% band excludes the very first and very last steps.
    assert 0 not in intervenable
    assert 9 not in intervenable
    assert 5 in intervenable
