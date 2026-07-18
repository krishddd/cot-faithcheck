"""Step-criticality gating and confidence intervals, end to end."""

from __future__ import annotations

import pytest

from cot_faithcheck import check_trace, parse_trace
from cot_faithcheck.clients import MockClient


def _trace_with_peripheral_step():
    # Steps 1 and 3 carry numbers (load-bearing for the faithful mock's sum);
    # step 2 is a peripheral aside that changes nothing when removed.
    return parse_trace(
        {
            "question": "Add the numbers.",
            "steps": [
                "Start with 10.",
                "Add 20.",
                "Let us be careful and double-check.",
                "Add 30.",
            ],
            "answer": "60",
            "gold_answer": "60",
        }
    )


def test_peripheral_step_not_penalised_end_to_end():
    trace = _trace_with_peripheral_step()
    report = check_trace(trace, MockClient("faithful"), k=6, seed=1)
    # The peripheral aside is detected and excluded; the trace stays fully faithful.
    assert report.n_peripheral_steps >= 1
    assert report.n_critical_steps >= 1
    assert report.faithfulness == pytest.approx(1.0)
    assert report.is_faithful is True
    peripheral = [s for s in report.step_scores if not s.is_critical]
    assert peripheral and all("careful" in s.step_text for s in peripheral)


def test_bypass_pins_to_zero_with_flag():
    trace = _trace_with_peripheral_step()
    report = check_trace(trace, MockClient("unfaithful", fixed_answer="60"), k=6, seed=1)
    assert report.n_critical_steps == 0
    assert report.faithfulness == pytest.approx(0.0)
    assert "Causal Bypass" in report.unfaithfulness_flags


def test_report_has_confidence_intervals():
    trace = _trace_with_peripheral_step()
    report = check_trace(trace, MockClient("faithful"), k=6, seed=1)
    assert report.faithfulness_ci is not None
    lo, hi = report.faithfulness_ci.low, report.faithfulness_ci.high
    assert 0.0 <= lo <= hi <= 1.0
    # Every intervention carries its own CI over the k trials.
    for s in report.step_scores:
        for r in s.interventions:
            assert r.ci is not None
            assert r.n_trials == 6


def test_confidence_interval_serialises():
    trace = _trace_with_peripheral_step()
    d = check_trace(trace, MockClient("faithful"), k=4, seed=1).to_dict()
    assert "faithfulness_ci" in d
    assert set(d["faithfulness_ci"]) == {"low", "high", "level"}


def test_criticality_threshold_is_configurable():
    trace = _trace_with_peripheral_step()
    # With an impossibly high threshold, even load-bearing steps look peripheral,
    # which collapses the trace to a bypass verdict.
    strict = check_trace(trace, MockClient("faithful"), k=4, seed=1, criticality_threshold=1.01)
    assert strict.n_critical_steps == 0
    assert strict.faithfulness == pytest.approx(0.0)
