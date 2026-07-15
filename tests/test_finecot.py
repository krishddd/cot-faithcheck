"""FINE-CoT loading and validation metrics."""

from __future__ import annotations

import pytest

from cot_faithcheck.clients import MockClient
from cot_faithcheck.finecot import (
    ValidationMetrics,
    evaluate_predictions,
    gold_faithful_label,
    load_finecot,
    validate,
)
from cot_faithcheck.pipeline import check_traces


def test_load_finecot_jsonl(fixtures_dir):
    labeled = load_finecot(str(fixtures_dir / "finecot_sample.jsonl"))
    assert len(labeled) == 3
    assert labeled[0].gold_faithful is True
    assert labeled[1].gold_faithful is False
    assert labeled[0].domain == "AQuA"


@pytest.mark.parametrize(
    "record,expected",
    [
        ({"faithful": True}, True),
        ({"faithful": False}, False),
        ({"label": "unfaithful"}, False),
        ({"label": "faithful"}, True),
        ({"is_faithful": 0}, False),
        ({"is_faithful": 1}, True),
        ({"nothing": "here"}, None),
    ],
)
def test_gold_faithful_label(record, expected):
    assert gold_faithful_label(record) is expected


def test_evaluate_predictions_perfect():
    gold = [True, False, True, False]
    pred = [True, False, True, False]
    m = evaluate_predictions(gold, pred)
    assert m.accuracy == 1.0
    assert m.f1 == 1.0
    assert m.false_positive_rate == 0.0


def test_evaluate_predictions_counts():
    # positive class = "unfaithful" (i.e. NOT faithful)
    gold = [False, False, True, True]  # 2 unfaithful (positives)
    pred = [False, True, True, False]  # detector flags item 1 (FP) and item 2 (TP)
    m = evaluate_predictions(gold, pred)
    assert (m.tp, m.fp, m.tn, m.fn) == (1, 1, 1, 1)
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
    assert m.false_positive_rate == pytest.approx(0.5)


def test_evaluate_skips_unlabeled():
    m = evaluate_predictions([None, True, False], [True, False, False])
    assert m.n == 3
    assert m.n_labeled == 2


def test_validate_end_to_end(fixtures_dir):
    labeled = load_finecot(str(fixtures_dir / "finecot_sample.jsonl"))
    reports = check_traces(
        [lt.trace for lt in labeled], MockClient("faithful"), k=3, temperature=0.0
    )
    metrics = validate(labeled, reports)
    assert isinstance(metrics, ValidationMetrics)
    assert metrics.n_labeled == 3
    assert "FINE-CoT validation" in metrics.summary()


def test_validate_length_mismatch(fixtures_dir):
    labeled = load_finecot(str(fixtures_dir / "finecot_sample.jsonl"))
    with pytest.raises(ValueError):
        validate(labeled, [])
