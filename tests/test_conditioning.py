"""Assistant-prefill conditioning and the logprob soft metric (P2)."""

from __future__ import annotations

import math

import pytest

from cot_faithcheck import check_trace, parse_trace
from cot_faithcheck.clients import MockClient
from cot_faithcheck.intervention import InterventionRunner, RunnerConfig
from cot_faithcheck.prompts import build_prefill_messages


def _trace():
    return parse_trace(
        {
            "question": "Add the numbers.",
            "steps": ["Take 10.", "Add 20.", "Add 30."],
            "answer": "60",
            "gold_answer": "60",
        }
    )


# -- prefill conditioning -----------------------------------------------------
def test_prefill_messages_put_reasoning_in_assistant_turn():
    trace = _trace()
    msgs = build_prefill_messages(trace.question, trace.steps[:2])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "Step 1" in msgs[-1]["content"] and "Step 2" in msgs[-1]["content"]
    # The user turn must not carry the reasoning (that is the whole point).
    assert "Step 1" not in msgs[1]["content"]


def test_auto_resolves_to_prefill_when_supported():
    trace = _trace()
    report = check_trace(trace, MockClient("faithful"), k=3, seed=1, conditioning="auto")
    assert report.config["conditioning"] == "prefill"


def test_conditioning_falls_back_to_template_without_capability():
    class NoPrefill(MockClient):
        supports_prefill = False

    trace = _trace()
    report = check_trace(trace, NoPrefill("faithful"), k=2, seed=1, conditioning="prefill")
    assert report.config["conditioning"] == "template"


def test_prefill_and_template_agree_on_mock():
    trace = _trace()
    pre = check_trace(trace, MockClient("faithful"), k=4, seed=1, conditioning="prefill")
    tpl = check_trace(trace, MockClient("faithful"), k=4, seed=1, conditioning="template")
    assert pre.faithfulness == pytest.approx(tpl.faithfulness)


def test_runner_exposes_resolved_mode():
    runner = InterventionRunner(MockClient("faithful"), RunnerConfig(k=2, conditioning="template"))
    assert runner.conditioning_mode == "template"
    assert runner.use_prefill is False


def test_invalid_conditioning_rejected():
    with pytest.raises(ValueError):
        RunnerConfig(conditioning="bogus")


# -- logprob soft metric ------------------------------------------------------
def test_logprob_soft_metric_selected_when_supported():
    trace = _trace()
    report = check_trace(trace, MockClient("faithful"), k=3, seed=1, use_logprobs=True)
    assert report.config["soft_metric"] == "logprob"


def test_logprob_falls_back_to_montecarlo_without_capability():
    class NoLogprobs(MockClient):
        supports_logprobs = False

    trace = _trace()
    report = check_trace(trace, NoLogprobs("faithful"), k=3, seed=1, use_logprobs=True)
    assert report.config["soft_metric"] == "montecarlo"


def test_logprob_soft_drops_on_load_bearing_step():
    trace = _trace()
    report = check_trace(trace, MockClient("faithful"), k=3, seed=1, use_logprobs=True)
    # Deleting a load-bearing step drains probability mass from the baseline answer.
    r = report.step_scores[0].interventions[0]
    assert r.baseline_prob_before > r.baseline_prob_after
    assert r.baseline_prob_before == pytest.approx(0.9)


def test_logprob_soft_zero_under_bypass():
    trace = _trace()
    report = check_trace(
        trace, MockClient("unfaithful", fixed_answer="60"), k=3, seed=1, use_logprobs=True
    )
    assert report.soft_faithfulness == pytest.approx(0.0)


def test_mock_logprob_of_matches_expected_answer():
    client = MockClient("faithful")
    msgs = build_prefill_messages(
        "Add.",
        parse_trace({"question": "Add.", "steps": ["Take 10.", "Add 20."], "answer": "30"}).steps,
    )
    # The faithful mock sums to 30 here, so P(30) is high and P(99) is low.
    assert client.logprob_of(msgs, "30") == pytest.approx(math.log(0.9))
    assert client.logprob_of(msgs, "99") == pytest.approx(math.log(0.05))
