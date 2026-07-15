"""Parse chain-of-thought traces into :class:`Trace` objects.

Accepts several common shapes so you can point the tool at whatever your logging
already produces:

1. **Explicit steps** - a list under ``steps`` / ``reasoning_steps`` /
   ``cot_steps`` (list of strings, or list of ``{"text": ...}`` dicts).
2. **A single reasoning blob** - a ``reasoning`` / ``rationale`` / ``cot`` string
   that is split into steps heuristically (numbered lists, "Step N:", bullet
   points, or sentence/newline fallback).

Required fields are the ``question`` and the ``final_answer`` (a range of key
aliases is accepted). ``options`` marks a multiple-choice item; ``gold_answer``
enables the correctness x faithfulness quadrant.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .errors import NoReasoningStepsError, TraceParseError
from .types import ReasoningStep, Trace

_QUESTION_KEYS = ("question", "query", "prompt_question", "problem", "input")
_STEPS_KEYS = ("steps", "reasoning_steps", "cot_steps", "rationale_steps")
_BLOB_KEYS = ("reasoning", "rationale", "cot", "chain_of_thought", "thinking")
_ANSWER_KEYS = ("final_answer", "answer", "prediction", "output", "response")
_GOLD_KEYS = ("gold_answer", "gold", "label", "target", "ground_truth", "correct_answer")

# Step-splitting patterns, tried in order of specificity.
_STEP_LINE_RE = re.compile(r"^\s*(?:step\s*\d+\s*[:.)-]|\d+\s*[:.)])\s*", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# An inline "Step N" marker, anywhere in the text (not only at line start), so a
# single-line blob like "Step 1: ... Step 2: ..." still splits into steps.
_INLINE_STEP_RE = re.compile(r"(?i)(?:(?<=^)|(?<=\s))step\s*\d+\s*[:.)-]\s*")


def _first_present(d: Dict[str, Any], keys) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def split_reasoning(blob: str) -> List[str]:
    """Split a free-text reasoning blob into ordered steps.

    Prefers explicit structure (numbered lines / "Step N:" / bullets); falls back
    to newline-delimited paragraphs, then to sentence segmentation.
    """
    text = (blob or "").strip()
    if not text:
        return []

    lines = [ln.rstrip() for ln in text.splitlines()]
    non_empty = [ln for ln in lines if ln.strip()]

    # 0. Inline "Step N:" markers (possibly all on one line) — split on them when
    #    there are at least two, regardless of newlines.
    if len(_INLINE_STEP_RE.findall(text)) >= 2:
        pieces = _INLINE_STEP_RE.split(text)
        steps = [" ".join(p.split()).strip() for p in pieces if p and p.strip()]
        if len(steps) >= 2:
            return steps

    # 1. Numbered / "Step N:" structure.
    if any(_STEP_LINE_RE.match(ln) for ln in non_empty):
        steps, current = [], []
        for ln in non_empty:
            if _STEP_LINE_RE.match(ln):
                if current:
                    steps.append(" ".join(current).strip())
                    current = []
                current.append(_STEP_LINE_RE.sub("", ln).strip())
            else:
                current.append(ln.strip())
        if current:
            steps.append(" ".join(current).strip())
        return [s for s in steps if s]

    # 2. Bullet list.
    if any(_BULLET_RE.match(ln) for ln in non_empty):
        return [_BULLET_RE.sub("", ln).strip() for ln in non_empty if _BULLET_RE.match(ln)]

    # 3. Multiple newline-separated lines -> one step per line.
    if len(non_empty) > 1:
        return [ln.strip() for ln in non_empty]

    # 4. Single line/paragraph -> sentence segmentation.
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sentences or [text]


def _coerce_steps(raw: Any) -> List[str]:
    if isinstance(raw, str):
        return split_reasoning(raw)
    if isinstance(raw, list):
        out: List[str] = []
        for item in raw:
            if isinstance(item, str):
                out.append(item.strip())
            elif isinstance(item, dict):
                txt = _first_present(item, ("text", "step", "content", "reasoning"))
                if txt:
                    out.append(str(txt).strip())
            else:
                out.append(str(item).strip())
        return [s for s in out if s]
    raise TraceParseError(f"cannot interpret steps of type {type(raw).__name__}")


def _mark_intervenable(steps: List[ReasoningStep], lo: float = 0.30, hi: float = 0.90) -> None:
    """Flag steps in the causally interesting middle band (C2-Faith 30%-90%).

    With very few steps this would exclude everything, so for <= 3 steps every
    step is intervenable.
    """
    n = len(steps)
    if n <= 3:
        for s in steps:
            s.is_intervenable = True
        return
    for s in steps:
        frac = (s.index + 0.5) / n
        s.is_intervenable = lo <= frac <= hi


def parse_trace(obj: Dict[str, Any], *, trace_id: Optional[str] = None) -> Trace:
    """Parse a single trace dict into a :class:`Trace`."""
    if not isinstance(obj, dict):
        raise TraceParseError("a trace must be a JSON object")

    question = _first_present(obj, _QUESTION_KEYS)
    if question is None:
        raise TraceParseError(f"trace is missing a question (looked for {_QUESTION_KEYS})")

    answer = _first_present(obj, _ANSWER_KEYS)
    if answer is None:
        raise TraceParseError(f"trace is missing a final answer (looked for {_ANSWER_KEYS})")

    raw_steps = _first_present(obj, _STEPS_KEYS)
    if raw_steps is None:
        blob = _first_present(obj, _BLOB_KEYS)
        if blob is None:
            raise NoReasoningStepsError(
                "trace has neither explicit steps nor a reasoning blob "
                f"(looked for {_STEPS_KEYS} or {_BLOB_KEYS})"
            )
        step_texts = _coerce_steps(blob)
    else:
        step_texts = _coerce_steps(raw_steps)

    if not step_texts:
        raise NoReasoningStepsError("no reasoning steps could be extracted from the trace")

    steps = [ReasoningStep(index=i, text=t) for i, t in enumerate(step_texts)]
    _mark_intervenable(steps)

    options = obj.get("options")
    if isinstance(options, list):  # ["A) ...", "B) ..."] -> {"A": "...", ...}
        options = _options_from_list(options)

    return Trace(
        question=str(question),
        steps=steps,
        final_answer=str(answer),
        prompt=str(obj.get("system", obj.get("prompt", ""))),
        options=options,
        gold_answer=_stringify(_first_present(obj, _GOLD_KEYS)),
        trace_id=trace_id or str(obj.get("id", obj.get("trace_id", "trace"))),
        metadata=obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {},
    )


def _stringify(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _options_from_list(items: List[Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, item in enumerate(items):
        text = str(item)
        m = re.match(r"\s*\(?([A-Za-z])\)?[.:)]\s*(.*)", text)
        if m:
            out[m.group(1).upper()] = m.group(2).strip()
        else:
            out[chr(ord("A") + i)] = text.strip()
    return out


def load_trace(source: Union[str, Path, Dict[str, Any]]) -> Trace:
    """Load one trace from a dict, a JSON string, or a path to a ``.json`` file."""
    if isinstance(source, dict):
        return parse_trace(source)
    if isinstance(source, (str, Path)) and _looks_like_path(source):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        if isinstance(data, list):
            if len(data) != 1:
                raise TraceParseError(
                    f"{source} contains {len(data)} traces; use load_traces() for multi-trace files"
                )
            return parse_trace(data[0])
        return parse_trace(data)
    # A raw JSON string.
    return parse_trace(json.loads(str(source)))


def load_traces(source: Union[str, Path, List[Any]]) -> List[Trace]:
    """Load a list of traces from a JSON array file / string / list."""
    if isinstance(source, list):
        data: Any = source
    elif _looks_like_path(source):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        data = json.loads(str(source))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise TraceParseError("expected a JSON array of traces")
    return [parse_trace(item, trace_id=str(item.get("id", i))) for i, item in enumerate(data)]


def _looks_like_path(source: Union[str, Path]) -> bool:
    if isinstance(source, Path):
        return True
    s = source.strip()
    if s.startswith("{") or s.startswith("["):
        return False
    return len(s) < 1024 and (s.endswith(".json") or s.endswith(".jsonl") or Path(s).exists())
