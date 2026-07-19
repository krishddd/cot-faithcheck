"""Judge ensembling, cost accounting, concurrency, and LLM answer-equivalence."""

from __future__ import annotations

import pytest

from cot_faithcheck import check_trace, parse_trace
from cot_faithcheck.clients import MockClient, TrackingClient
from cot_faithcheck.equivalence import make_equivalence
from cot_faithcheck.judge import JudgeScorer


def _math_trace():
    return parse_trace(
        {
            "question": "Add the numbers.",
            "steps": ["Take 10.", "Add 20.", "Add 30."],
            "answer": "60",
            "gold_answer": "60",
        }
    )


def _judge_trace():
    return parse_trace(
        {"question": "q", "reasoning": "The answer is 4 because reasons.", "answer": "4"}
    )


# -- cost accounting ----------------------------------------------------------
def test_usage_is_recorded():
    report = check_trace(_math_trace(), MockClient("faithful"), k=4, seed=1)
    u = report.usage
    assert u is not None
    assert u.n_calls > 0
    assert u.n_samples >= u.n_calls
    assert u.est_total_tokens > 0


def test_usage_serialises():
    d = check_trace(_math_trace(), MockClient("faithful"), k=3, seed=1).to_dict()
    assert "usage" in d
    assert d["usage"]["n_calls"] > 0


def test_tracking_client_mirrors_capabilities():
    inner = MockClient("faithful")
    t = TrackingClient(inner)
    assert t.supports_prefill == inner.supports_prefill
    assert t.supports_logprobs == inner.supports_logprobs
    assert t.model == inner.model
    out = t.generate([{"role": "user", "content": "hi"}], n=3)
    assert len(out) == 3
    assert t.usage().n_calls == 1
    assert t.usage().n_samples == 3


# -- concurrency --------------------------------------------------------------
def test_parallel_matches_sequential():
    seq = check_trace(_math_trace(), MockClient("faithful"), k=5, seed=1, max_workers=1)
    par = check_trace(_math_trace(), MockClient("faithful"), k=5, seed=1, max_workers=4)
    assert par.faithfulness == pytest.approx(seq.faithfulness)
    assert [s.faithfulness for s in par.step_scores] == pytest.approx(
        [s.faithfulness for s in seq.step_scores]
    )


def test_invalid_max_workers_rejected():
    from cot_faithcheck.intervention import RunnerConfig

    with pytest.raises(ValueError):
        RunnerConfig(max_workers=0)


# -- judge ensembling ---------------------------------------------------------
def test_judge_ensemble_majority_vote():
    # Three judges: two say unfaithful (0.1), one says faithful (0.9). Majority
    # verdict is unfaithful; the mean score is below threshold.
    replies = [
        {
            "is_faithful": False,
            "faithfulness_score": 0.1,
            "flags": ["Weak Justification"],
            "per_step": [],
            "rationale": "no",
        },
        {
            "is_faithful": False,
            "faithfulness_score": 0.2,
            "flags": ["Weak Justification"],
            "per_step": [],
            "rationale": "no",
        },
        {
            "is_faithful": True,
            "faithfulness_score": 0.9,
            "flags": [],
            "per_step": [],
            "rationale": "yes",
        },
    ]
    calls = {"i": 0}

    def answer_fn(messages):
        import json

        i = calls["i"]
        calls["i"] += 1
        return json.dumps(replies[i % len(replies)])

    client = MockClient("custom", answer_fn=answer_fn)
    report = JudgeScorer(client, samples=3).score(_judge_trace())
    assert report.is_faithful is False
    assert report.faithfulness == pytest.approx((0.1 + 0.2 + 0.9) / 3)
    assert "Weak Justification" in report.unfaithfulness_flags  # 2/3 majority
    assert report.config["judge_samples"] == 3


def test_judge_samples_wired_through_pipeline():
    report = check_trace(_judge_trace(), MockClient("unfaithful"), mode="judge", judge_samples=4)
    assert report.config["judge_samples"] == 4
    assert "judge" in report.summary.lower()


# -- LLM answer-equivalence fallback -----------------------------------------
def test_equivalence_regex_hit_does_not_call_llm():
    # "60" and "60.0" are equal by the numeric regex, so the LLM is never consulted.
    client = MockClient("custom", answer_fn=lambda m: "SHOULD NOT BE CALLED")
    tracker = TrackingClient(client)
    eq = make_equivalence(tracker)
    assert eq("60", "60.0") is True
    assert tracker.usage().n_calls == 0


def test_equivalence_llm_settles_regex_miss():
    # Regex says "1/2" != "0.5"; the LLM (mock) says yes.
    client = MockClient("custom", answer_fn=lambda m: "yes")
    eq = make_equivalence(client)
    assert eq("1/2", "0.5") is True


def test_equivalence_llm_no_keeps_distinct():
    client = MockClient("custom", answer_fn=lambda m: "no")
    eq = make_equivalence(client)
    assert eq("cats", "dogs") is False
