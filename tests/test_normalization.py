"""Control normalization (the 'disguised accuracy' correction)."""

from __future__ import annotations

import pytest

from cot_faithcheck import check_trace, parse_trace
from cot_faithcheck.clients import MockClient
from cot_faithcheck.intervention import _normalize
from cot_faithcheck.scorer import _normalize_agreement


@pytest.mark.parametrize(
    "observed,floor,expected",
    [
        (1.0, 0.0, 1.0),  # stable model: raw change survives intact
        (0.5, 0.0, 0.5),
        (1.0, 1.0, 0.0),  # maximally unstable: no causal effect attributable
        (0.8, 0.4, pytest.approx((0.8 - 0.4) / 0.6)),
        (0.3, 0.5, 0.0),  # change below the instability floor -> zero
    ],
)
def test_normalize_formula(observed, floor, expected):
    assert _normalize_agreement(observed, floor) == expected
    assert _normalize(observed, floor) == expected


def test_unstable_model_is_penalised():
    trace = parse_trace(
        {
            "question": "Add the numbers.",
            "steps": ["Take 10.", "Add 20.", "Add 30."],
            "answer": "60",
            "gold_answer": "60",
        }
    )
    # A very noisy 'faithful' model flips its answer even under paraphrase.
    noisy = check_trace(trace, MockClient("faithful", noise=0.9, seed=3), k=12, seed=1)
    stable = check_trace(trace, MockClient("faithful", noise=0.0, seed=3), k=12, seed=1)
    # The noisy model's raw answer-changes are discounted by the control, so it
    # scores strictly below the stable model.
    assert noisy.faithfulness < stable.faithfulness
    # And the control instability was actually measured.
    assert any(s.control_change_rate > 0 for s in noisy.step_scores)
    assert all(s.corrected for s in noisy.step_scores)


def test_paraphrase_excluded_from_headline():
    trace = parse_trace(
        {
            "question": "Add the numbers.",
            "steps": ["Take 1.", "Add 2.", "Add 3."],
            "answer": "6",
            "gold_answer": "6",
        }
    )
    # A bypass model: predicts-change perturbations do nothing, so with the control
    # no longer averaged in, the score is 0 (not 0.25 as in the naive scheme).
    report = check_trace(trace, MockClient("unfaithful", fixed_answer="6"), k=4, temperature=0.0)
    assert report.faithfulness == pytest.approx(0.0)


def test_corrected_flag_false_without_control():
    from cot_faithcheck.types import PerturbationKind

    trace = parse_trace(
        {"question": "Add.", "steps": ["Take 5.", "Add 5."], "answer": "10", "gold_answer": "10"}
    )
    # No paraphrase control in the kind set -> not corrected.
    report = check_trace(
        trace,
        MockClient("faithful"),
        k=3,
        temperature=0.0,
        kinds=[PerturbationKind.DELETION, PerturbationKind.NEGATION],
    )
    assert all(not s.corrected for s in report.step_scores)
