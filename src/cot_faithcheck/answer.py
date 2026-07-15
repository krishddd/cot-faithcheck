"""Answer extraction, normalisation, and equivalence.

To measure whether a perturbation *changed the answer*, we need to reduce a free-
text model continuation to a canonical answer token and compare two such tokens
robustly. The heuristics here cover the common shapes the FINE-CoT source domains
use (AQuA / LogiQA multiple choice, TruthfulQA short answers, HLE-Bio numeric /
short-phrase answers):

    * ``\\boxed{...}`` (math)
    * ``Answer: X`` / ``The answer is X`` / ``Final answer: X``
    * a lone multiple-choice letter (A-E), with or without parentheses
    * a trailing number
    * otherwise the last non-empty line, normalised
"""

from __future__ import annotations

import re
from typing import Optional

# Ordered from most to least specific. Each pattern's first group is the answer.
_ANSWER_PATTERNS = [
    re.compile(r"\\boxed\{([^}]*)\}"),
    re.compile(r"final\s+answer\s*(?:is|:|=)\s*(.+?)(?:[.\n]|$)", re.IGNORECASE),
    re.compile(r"\bthe\s+answer\s+is\s*:?\s*(.+?)(?:[.\n]|$)", re.IGNORECASE),
    re.compile(r"\banswer\s*:\s*(.+?)(?:[.\n]|$)", re.IGNORECASE),
    re.compile(r"\banswer\s+is\s+(.+?)(?:[.\n]|$)", re.IGNORECASE),
]

# A multiple-choice option letter, e.g. "(B)", "B.", "B)" or a standalone "B".
_OPTION_RE = re.compile(r"^\s*\(?([A-E])\)?[.:)]?\s*$")
_OPTION_IN_TEXT_RE = re.compile(r"\boption\s*\(?([A-E])\)?", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _canonical(text: str) -> str:
    """Lower-case, strip surrounding punctuation/quotes and collapse whitespace."""
    text = text.strip().strip("\"'`").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .:;,")
    return text.lower()


def _as_number(text: str) -> Optional[float]:
    m = _NUMBER_RE.search(text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_answer(text: str) -> str:
    """Extract a canonical answer token from a free-text model output.

    Returns an empty string when nothing answer-like can be found (an empty
    completion, for instance), which callers treat as "no answer".
    """
    if not text:
        return ""

    # 1. Explicit answer markers, searched over the whole text (last match wins,
    #    since models often restate the answer at the very end).
    for pat in _ANSWER_PATTERNS:
        matches = pat.findall(text)
        if matches:
            candidate = matches[-1].strip()
            # A marker like "The answer is B" should canonicalise to the letter.
            opt = _OPTION_RE.match(candidate)
            if opt:
                return opt.group(1).upper()
            return _canonical(candidate)

    # 2. Fall back to the last non-empty line.
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    last = lines[-1]

    opt = _OPTION_RE.match(last)
    if opt:
        return opt.group(1).upper()

    opt_in = _OPTION_IN_TEXT_RE.search(last)
    if opt_in:
        return opt_in.group(1).upper()

    return _canonical(last)


def answers_equivalent(a: str, b: str, *, numeric_tol: float = 1e-6) -> bool:
    """Return whether two extracted answers denote the same thing.

    Compares canonically, then numerically (so ``"42"`` == ``"42.0"`` and
    ``"$42"`` == ``"42"``), and treats a bare option letter as matching a longer
    string that ends in that option letter.
    """
    if a is None or b is None:
        return False
    ca, cb = _canonical(a), _canonical(b)
    if ca == cb:
        return True
    if not ca or not cb:
        return False

    na, nb = _as_number(ca), _as_number(cb)
    if na is not None and nb is not None:
        return abs(na - nb) <= numeric_tol

    # Option-letter tolerance: "b" vs "option b" / "b) 42".
    for x, y in ((ca, cb), (cb, ca)):
        if len(x) == 1 and x.isalpha():
            if re.search(rf"(^|\b){re.escape(x)}(\b|[).:])", y):
                return True
    return False
