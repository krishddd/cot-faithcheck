"""Perturbation generation."""

from __future__ import annotations

from cot_faithcheck.perturb import PerturbationGenerator, _negate_text
from cot_faithcheck.types import PerturbationKind


def test_default_kinds_generated_per_step(math_trace):
    gen = PerturbationGenerator(seed=0)
    perts = gen.for_step(math_trace, math_trace.steps[0])
    kinds = {p.kind for p in perts}
    assert PerturbationKind.DELETION in kinds
    assert PerturbationKind.NEGATION in kinds
    assert PerturbationKind.NUMERIC in kinds
    assert PerturbationKind.PARAPHRASE in kinds


def test_deletion_has_no_text_and_predicts_change(math_trace):
    gen = PerturbationGenerator([PerturbationKind.DELETION])
    p = gen.for_step(math_trace, math_trace.steps[0])[0]
    assert p.perturbed_text is None
    assert p.predicts_change is True


def test_paraphrase_is_control(math_trace):
    gen = PerturbationGenerator([PerturbationKind.PARAPHRASE])
    p = gen.for_step(math_trace, math_trace.steps[0])[0]
    assert p.predicts_change is False
    # Meaning-preserving: the number is retained.
    assert "12" in p.perturbed_text


def test_numeric_changes_a_number(math_trace):
    gen = PerturbationGenerator([PerturbationKind.NUMERIC], seed=1)
    p = gen.for_step(math_trace, math_trace.steps[0])[0]
    assert "12" not in p.perturbed_text
    assert p.predicts_change is True


def test_numeric_skipped_without_numbers():
    from cot_faithcheck.types import ReasoningStep, Trace

    trace = Trace(question="q", steps=[ReasoningStep(0, "no digits here")], final_answer="x")
    gen = PerturbationGenerator([PerturbationKind.NUMERIC])
    assert gen.for_step(trace, trace.steps[0]) == []


def test_negation_swaps_or_wraps():
    assert "not" in _negate_text("x is true").lower()
    assert _negate_text("the value equals five") == "the value does not equal five"
    assert "NOT" in _negate_text("Add 8 apples")


def test_option_shuffle_predicts_letter_change(mcq_trace):
    gen = PerturbationGenerator([PerturbationKind.OPTION_SHUFFLE], seed=3)
    perts = gen.for_trace(mcq_trace)
    assert perts, "expected option-shuffle perturbations for an MCQ trace"
    p = perts[0]
    assert p.expected_answer is not None
    assert p.expected_answer != mcq_trace.final_answer
    assert p.predicts_change is True


def test_option_shuffle_absent_for_non_mcq(math_trace):
    gen = PerturbationGenerator([PerturbationKind.OPTION_SHUFFLE])
    assert gen.for_trace(math_trace) == []


def test_acausal_uses_client_when_available(math_trace):
    from cot_faithcheck.clients import MockClient

    # custom client returns a fixed rewrite for any rewrite request
    client = MockClient("custom", answer_fn=lambda m: "a fluent but wrong step")
    gen = PerturbationGenerator([PerturbationKind.ACAUSAL], client=client)
    p = gen.for_step(math_trace, math_trace.steps[0])[0]
    assert p.kind == PerturbationKind.ACAUSAL
    assert p.predicts_change is True
    assert p.perturbed_text
