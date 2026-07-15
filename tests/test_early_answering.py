"""Early-answering (Lanham truncation curve) analysis."""

from __future__ import annotations

import pytest

from cot_faithcheck import check_trace
from cot_faithcheck.clients import MockClient
from cot_faithcheck.intervention import InterventionRunner, RunnerConfig


def test_faithful_converges_late(math_trace, faithful_client):
    runner = InterventionRunner(faithful_client, RunnerConfig(k=3, temperature=0.0))
    ea = runner.early_answering(math_trace)
    assert ea is not None
    # The faithful model's answer only equals the full-reasoning answer once all
    # the numbers are present, so early truncations do NOT match -> AOC high.
    assert ea.aoc == pytest.approx(1.0)
    assert [frac for _, frac in ea.convergence] == [0.0, 0.0, 0.0]


def test_bypass_converges_immediately(math_trace):
    client = MockClient("unfaithful", fixed_answer="25")
    runner = InterventionRunner(client, RunnerConfig(k=3, temperature=0.0))
    ea = runner.early_answering(math_trace)
    # A bypass model already emits the final answer with zero reasoning -> AOC 0.
    assert ea.aoc == pytest.approx(0.0)
    assert all(frac == 1.0 for _, frac in ea.convergence)


def test_early_answering_wired_into_report(math_trace, faithful_client):
    report = check_trace(math_trace, faithful_client, k=3, temperature=0.0, early_answering=True)
    assert report.early_answering is not None
    assert report.early_answering.aoc == pytest.approx(1.0)


def test_early_answering_can_be_disabled(math_trace, faithful_client):
    report = check_trace(math_trace, faithful_client, k=3, temperature=0.0, early_answering=False)
    assert report.early_answering is None


def test_early_answering_reuses_final_answer(math_trace, faithful_client):
    runner = InterventionRunner(faithful_client, RunnerConfig(k=2, temperature=0.0))
    ea = runner.early_answering(math_trace, final_answer="25")
    assert ea.final_answer == "25"


def test_early_answering_serialises(math_trace, faithful_client):
    report = check_trace(math_trace, faithful_client, k=2, temperature=0.0)
    d = report.to_dict()
    assert "early_answering" in d
    assert d["early_answering"]["aoc"] == pytest.approx(1.0)
    assert isinstance(d["early_answering"]["convergence"][0], list)
