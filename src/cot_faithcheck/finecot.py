"""Load and validate against the FINE-CoT dataset.

FINE-CoT (Faithfulness INstance Evaluation for Chain-of-Thought), the dataset
behind FaithCoT-Bench (arXiv 2510.04040, code: https://github.com/se7esx/FaithCoT-BENCH),
ships expert-annotated reasoning trajectories across AQuA, LogiQA, TruthfulQA and
HLE-Bio, each labelled faithful / unfaithful with step-level evidence.

This module converts those records into :class:`Trace` objects and scores the
detector against the human labels, reporting the standard binary-detection metrics
FaithCoT-Bench uses (accuracy, precision, recall, F1, plus false-positive rate).

The loader is field-tolerant because the released JSON has evolved; point it at
the dataset JSON/JSONL you downloaded and it will map the common key aliases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .parser import parse_trace
from .types import FaithfulnessReport, Trace

# Aliases seen across FINE-CoT / FaithCoT-Bench releases.
_LABEL_KEYS = ("faithful", "is_faithful", "label", "faithfulness_label", "gold_faithful")
_UNFAITHFUL_TOKENS = {"unfaithful", "false", "0", "no", "not_faithful", "0.0"}
_FAITHFUL_TOKENS = {"faithful", "true", "1", "yes", "1.0"}


def _read_records(source: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(source)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, dict):
        # Some releases wrap records under a top-level key.
        for key in ("data", "records", "instances", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError(f"unrecognised FINE-CoT structure in {source}")


def gold_faithful_label(record: Dict[str, Any]) -> Optional[bool]:
    """Extract the human faithful/unfaithful label from a FINE-CoT record."""
    for key in _LABEL_KEYS:
        if key in record and record[key] is not None:
            val = record[key]
            if isinstance(val, bool):
                return val
            token = str(val).strip().lower()
            if token in _UNFAITHFUL_TOKENS:
                return False
            if token in _FAITHFUL_TOKENS:
                return True
    return None


@dataclass
class LabeledTrace:
    """A FINE-CoT trace paired with its gold faithfulness label."""

    trace: Trace
    gold_faithful: Optional[bool]
    domain: Optional[str] = None


def load_finecot(source: Union[str, Path]) -> List[LabeledTrace]:
    """Load FINE-CoT records into labelled traces."""
    out: List[LabeledTrace] = []
    for i, rec in enumerate(_read_records(source)):
        rec.setdefault("id", rec.get("qid", i))
        trace = parse_trace(rec, trace_id=str(rec["id"]))
        out.append(
            LabeledTrace(
                trace=trace,
                gold_faithful=gold_faithful_label(rec),
                domain=rec.get("domain") or rec.get("dataset") or rec.get("source"),
            )
        )
    return out


@dataclass
class ValidationMetrics:
    """Binary-detection metrics for the unfaithfulness detector vs. gold labels.

    "Positive" = the detector's job of *flagging unfaithful*, matching the
    FaithCoT-Bench framing of unfaithfulness detection as the positive class.
    """

    n: int
    n_labeled: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def summary(self) -> str:
        return (
            f"FINE-CoT validation on {self.n_labeled} labelled traces: "
            f"acc={self.accuracy:.3f} precision={self.precision:.3f} "
            f"recall={self.recall:.3f} F1={self.f1:.3f} FPR={self.false_positive_rate:.3f}"
        )


def evaluate_predictions(
    gold_faithful: List[Optional[bool]],
    predicted_faithful: List[bool],
) -> ValidationMetrics:
    """Compute detection metrics; unlabelled items (gold is None) are skipped."""
    tp = fp = tn = fn = 0
    n_labeled = 0
    for gold, pred in zip(gold_faithful, predicted_faithful):
        if gold is None:
            continue
        n_labeled += 1
        gold_unfaithful = not gold
        pred_unfaithful = not pred
        if gold_unfaithful and pred_unfaithful:
            tp += 1
        elif not gold_unfaithful and pred_unfaithful:
            fp += 1
        elif not gold_unfaithful and not pred_unfaithful:
            tn += 1
        else:
            fn += 1

    def _safe(a: int, b: int) -> float:
        return a / b if b else 0.0

    precision = _safe(tp, tp + fp)
    recall = _safe(tp, tp + fn)
    f1 = _safe(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
    accuracy = _safe(tp + tn, n_labeled)
    fpr = _safe(fp, fp + tn)
    return ValidationMetrics(
        n=len(gold_faithful),
        n_labeled=n_labeled,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )


def validate(
    labeled: List[LabeledTrace],
    reports: List[FaithfulnessReport],
) -> ValidationMetrics:
    """Compare detector verdicts against FINE-CoT gold labels.

    ``reports[i]`` must correspond to ``labeled[i]``.
    """
    if len(labeled) != len(reports):
        raise ValueError("labeled traces and reports must align 1:1")
    gold = [lt.gold_faithful for lt in labeled]
    pred = [r.is_faithful for r in reports]
    return evaluate_predictions(gold, pred)
