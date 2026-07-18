"""Faithfulness scoring and aggregation."""

from __future__ import annotations

import pytest

from cot_faithcheck.scorer import FaithfulnessScorer, score_step
from cot_faithcheck.types import (
    InterventionResult,
    Perturbation,
    PerturbationKind,
    Quadrant,
    ReasoningStep,
    Trace,
)


def _result(kind, agreement, predicts_change=True, before=1.0, after=0.0):
    pert = Perturbation(
        step_index=0, kind=kind, perturbed_text="x", predicts_change=predicts_change
    )
    return InterventionResult(
        perturbation=pert,
        baseline_answer="A",
        perturbed_answers=["B"],
        changed_fraction=agreement if predicts_change else 1 - agreement,
        matched_expected_fraction=None,
        baseline_prob_before=before,
        baseline_prob_after=after,
        agreement=agreement,
    )


def test_score_step_means_agreement():
    results = [
        _result(PerturbationKind.DELETION, 1.0),
        _result(PerturbationKind.NEGATION, 0.0),
    ]
    ss = score_step(0, "text", results)
    assert ss.faithfulness == pytest.approx(0.5)
    assert ss.soft_faithfulness == pytest.approx(1.0)  # both drop 1.0 -> 0.0


def _trace():
    return Trace(
        question="q",
        steps=[ReasoningStep(0, "s0"), ReasoningStep(1, "s1")],
        final_answer="A",
        gold_answer="A",
    )


def test_scorer_faithful_quadrant():
    trace = _trace()
    by_step = {
        0: [_result(PerturbationKind.DELETION, 1.0)],
        1: [_result(PerturbationKind.NEGATION, 1.0)],
    }
    report = FaithfulnessScorer(threshold=0.5).score(trace, by_step, baseline_answer="A")
    assert report.is_faithful is True
    assert report.quadrant == Quadrant.CORRECT_FAITHFUL
    assert report.faithfulness == pytest.approx(1.0)


def test_scorer_flags_bypass():
    trace = _trace()
    by_step = {
        0: [_result(PerturbationKind.DELETION, 0.0)],
        1: [_result(PerturbationKind.NEGATION, 0.0)],
    }
    report = FaithfulnessScorer(threshold=0.5).score(trace, by_step, baseline_answer="A")
    assert report.is_faithful is False
    # Correct answer (baseline == gold) but unfaithful -> Type 2.
    assert report.quadrant == Quadrant.CORRECT_UNFAITHFUL
    # No step is load-bearing, so the whole trace is a causal bypass.
    assert report.n_critical_steps == 0
    assert "Causal Bypass" in report.unfaithfulness_flags


def test_scorer_flags_principle_on_critical_step():
    # Step 0 is load-bearing (deletion moves the answer) but its negation does not
    # propagate — a real per-step unfaithfulness signal on a critical step.
    trace = _trace()
    by_step = {
        0: [
            _result(PerturbationKind.DELETION, 1.0),
            _result(PerturbationKind.NEGATION, 0.0),
        ],
    }
    report = FaithfulnessScorer(threshold=0.5).score(trace, by_step, baseline_answer="A")
    assert report.n_critical_steps == 1
    assert "Invalid Reasoning Chains" in report.unfaithfulness_flags
    assert "Causal Bypass" not in report.unfaithfulness_flags


def test_peripheral_step_excluded_from_score():
    # One load-bearing step (agreement 1.0) and one peripheral step (agreement 0.0
    # under every corruption). The peripheral step must not drag the score down.
    trace = _trace()
    by_step = {
        0: [_result(PerturbationKind.DELETION, 1.0)],
        1: [_result(PerturbationKind.DELETION, 0.0)],
    }
    report = FaithfulnessScorer(threshold=0.5).score(trace, by_step, baseline_answer="A")
    assert report.n_critical_steps == 1
    assert report.n_peripheral_steps == 1
    assert report.faithfulness == pytest.approx(1.0)
    assert report.is_faithful is True


def test_scorer_unknown_quadrant_without_gold():
    trace = Trace(question="q", steps=[ReasoningStep(0, "s")], final_answer="A")
    by_step = {0: [_result(PerturbationKind.DELETION, 1.0)]}
    report = FaithfulnessScorer().score(trace, by_step)
    assert report.answer_correct is None
    assert report.quadrant == Quadrant.UNKNOWN


def test_scorer_incorrect_faithful_quadrant():
    trace = _trace()
    by_step = {0: [_result(PerturbationKind.DELETION, 1.0)]}
    # Model's stable answer diverges from gold, but reasoning is faithful -> Type 3.
    report = FaithfulnessScorer().score(trace, by_step, baseline_answer="WRONG")
    assert report.answer_correct is False
    assert report.quadrant == Quadrant.INCORRECT_FAITHFUL


def test_empty_results_scores_zero():
    trace = _trace()
    report = FaithfulnessScorer().score(trace, {})
    assert report.faithfulness == 0.0
    assert report.step_scores == []
