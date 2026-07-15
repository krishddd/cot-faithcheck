"""The counterfactual intervention runner and k-run harness."""

from __future__ import annotations

from cot_faithcheck.intervention import InterventionRunner, RunnerConfig, _majority
from cot_faithcheck.perturb import PerturbationGenerator
from cot_faithcheck.types import PerturbationKind


def test_majority_vote_with_equivalence():
    assert _majority(["42", "42.0", "43"]) in ("42", "42.0")
    assert _majority([]) == ""


def test_deletion_moves_answer_for_faithful_model(math_trace, faithful_client):
    runner = InterventionRunner(faithful_client, RunnerConfig(k=3, temperature=0.0))
    gen = PerturbationGenerator([PerturbationKind.DELETION])
    p = gen.for_step(math_trace, math_trace.steps[0])[0]
    result = runner.run(math_trace, p)
    # Faithful model: deleting the first step changes the answer -> high agreement.
    assert result.changed_fraction == 1.0
    assert result.agreement == 1.0


def test_deletion_does_not_move_answer_for_bypass_model(math_trace, unfaithful_client):
    runner = InterventionRunner(unfaithful_client, RunnerConfig(k=3, temperature=0.0))
    gen = PerturbationGenerator([PerturbationKind.DELETION])
    p = gen.for_step(math_trace, math_trace.steps[0])[0]
    result = runner.run(math_trace, p)
    # Causal-bypass model: answer stays anchored -> agreement 0.
    assert result.changed_fraction == 0.0
    assert result.agreement == 0.0


def test_paraphrase_control_agreement(math_trace, faithful_client):
    runner = InterventionRunner(faithful_client, RunnerConfig(k=3, temperature=0.0))
    gen = PerturbationGenerator([PerturbationKind.PARAPHRASE])
    p = gen.for_step(math_trace, math_trace.steps[0])[0]
    result = runner.run(math_trace, p)
    # A meaning-preserving paraphrase should NOT change a faithful model's answer,
    # which (since the control predicts no change) is full agreement.
    assert result.changed_fraction == 0.0
    assert result.agreement == 1.0


def test_soft_metric_tracks_probability_mass(math_trace, faithful_client):
    runner = InterventionRunner(faithful_client, RunnerConfig(k=5, temperature=0.0))
    gen = PerturbationGenerator([PerturbationKind.DELETION])
    p = gen.for_step(math_trace, math_trace.steps[0])[0]
    result = runner.run(math_trace, p)
    # Baseline answer is fully supported before, fully drained after.
    assert result.baseline_prob_before == 1.0
    assert result.baseline_prob_after == 0.0


def test_krun_reduces_variance(math_trace):
    from cot_faithcheck.clients import MockClient

    # A noisy faithful model: with k=1 the baseline could be a noise sample; with a
    # larger k the majority vote recovers the true answer.
    noisy = MockClient("faithful", noise=0.2, seed=5)
    runner = InterventionRunner(noisy, RunnerConfig(k=15, temperature=0.9))
    gen = PerturbationGenerator([PerturbationKind.DELETION])
    p = gen.for_step(math_trace, math_trace.steps[1])[0]
    result = runner.run(math_trace, p)
    # Despite noise, the majority baseline is the true sum through step 1 (=20).
    assert result.baseline_answer == "20"


def test_baseline_cache_shared_across_perturbations(math_trace, faithful_client):
    runner = InterventionRunner(faithful_client, RunnerConfig(k=2, temperature=0.0))
    gen = PerturbationGenerator([PerturbationKind.DELETION, PerturbationKind.NEGATION])
    perts = gen.for_step(math_trace, math_trace.steps[0])
    runner.run_all(math_trace, perts)
    # Both perturbations target step 0, so exactly one baseline is cached for it.
    assert 0 in runner._baseline_cache
