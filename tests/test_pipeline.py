"""End-to-end pipeline: parse -> perturb -> intervene -> score."""

from __future__ import annotations

import pytest

from cot_faithcheck import check_file, check_trace
from cot_faithcheck.clients import MockClient
from cot_faithcheck.types import Detector, Quadrant


def test_faithful_model_scores_high(math_trace, faithful_client):
    report = check_trace(math_trace, faithful_client, k=3, temperature=0.0)
    assert report.faithfulness == pytest.approx(1.0)
    assert report.is_faithful is True
    assert report.detector == Detector.INTERVENTION
    assert report.quadrant == Quadrant.CORRECT_FAITHFUL
    assert report.answer_correct is True
    assert report.unfaithfulness_flags == []


def test_bypass_model_scores_low(math_trace):
    # Answer anchored on the (correct) stated answer -> Type 2 correct+unfaithful.
    client = MockClient("unfaithful", fixed_answer="25")
    report = check_trace(math_trace, client, k=3, temperature=0.0)
    assert report.faithfulness < 0.5
    assert report.is_faithful is False
    assert report.quadrant == Quadrant.CORRECT_UNFAITHFUL
    assert report.unfaithfulness_flags  # at least one principle fired


def test_weakest_step_identified(math_trace):
    client = MockClient("unfaithful", fixed_answer="25")
    report = check_trace(math_trace, client, k=3, temperature=0.0)
    weakest = report.weakest_step()
    assert weakest is not None
    assert weakest.faithfulness < 0.5


def test_judge_mode(math_trace):
    report = check_trace(math_trace, MockClient("faithful"), mode="judge")
    assert report.detector == Detector.JUDGE
    assert report.is_faithful is True


def test_auto_falls_back_to_judge_when_no_intervenable_steps():
    from cot_faithcheck.types import Trace

    # A single-step trace still has an intervenable step, so force the empty case.
    trace = Trace(question="q", steps=[], final_answer="x")
    report = check_trace(trace, MockClient("faithful"), mode="auto")
    assert report.detector == Detector.JUDGE


def test_invalid_mode_raises(math_trace):
    with pytest.raises(ValueError):
        check_trace(math_trace, MockClient("faithful"), mode="bogus")


def test_check_file_batch(fixtures_dir, faithful_client):
    reports = check_file(str(fixtures_dir / "batch.json"), faithful_client, k=2, temperature=0.0)
    assert len(reports) == 2
    assert all(r.is_faithful for r in reports)


def test_config_recorded(math_trace, faithful_client):
    report = check_trace(math_trace, faithful_client, k=4, temperature=0.0, threshold=0.6)
    assert report.config["k"] == 4
    assert report.config["threshold"] == 0.6
    assert report.config["provider"] == "mock"
