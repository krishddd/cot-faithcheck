"""LLM-as-judge detector."""

from __future__ import annotations

import pytest

from cot_faithcheck.clients import MockClient
from cot_faithcheck.judge import JudgeScorer, _extract_json, build_judge_messages
from cot_faithcheck.prompts import JUDGE_SENTINEL
from cot_faithcheck.types import Detector


def test_judge_messages_carry_sentinel(math_trace):
    msgs = build_judge_messages(math_trace)
    assert JUDGE_SENTINEL in msgs[0]["content"]
    assert "Reasoning steps" in msgs[1]["content"]


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_fenced_block():
    text = 'Sure!\n```json\n{"is_faithful": true, "flags": []}\n```\nDone.'
    assert _extract_json(text)["is_faithful"] is True


def test_extract_json_embedded_in_prose():
    text = 'The verdict: {"is_faithful": false, "faithfulness_score": 0.1} end.'
    assert _extract_json(text)["faithfulness_score"] == 0.1


def test_extract_json_no_object_raises():
    with pytest.raises(ValueError):
        _extract_json("no json here")


def test_judge_uses_supplied_verdict(math_trace):
    verdict = {
        "is_faithful": False,
        "faithfulness_score": 0.2,
        "flags": ["Weak Justification", "not-a-real-principle"],
        "per_step": [{"index": 0, "score": 0.1, "issue": "circular"}],
        "rationale": "post-hoc",
    }
    client = MockClient("faithful", judge_json=verdict)
    report = JudgeScorer(client).score(math_trace)
    assert report.detector == Detector.JUDGE
    assert report.is_faithful is False
    assert report.faithfulness == pytest.approx(0.2)
    # Unknown principle names are filtered out.
    assert report.unfaithfulness_flags == ["Weak Justification"]
    # Per-step score is threaded through.
    assert report.step_scores[0].faithfulness == pytest.approx(0.1)


def test_judge_score_clamped(math_trace):
    verdict = {"is_faithful": True, "faithfulness_score": 5.0, "flags": [], "per_step": []}
    report = JudgeScorer(MockClient("faithful", judge_json=verdict)).score(math_trace)
    assert 0.0 <= report.faithfulness <= 1.0


def test_judge_quadrant_uses_gold(math_trace):
    verdict = {"is_faithful": True, "faithfulness_score": 0.9, "flags": [], "per_step": []}
    report = JudgeScorer(MockClient("faithful", judge_json=verdict)).score(math_trace)
    # gold == final answer, judged faithful -> Type 1.
    assert report.answer_correct is True
    assert report.quadrant.value == "correct_faithful"
