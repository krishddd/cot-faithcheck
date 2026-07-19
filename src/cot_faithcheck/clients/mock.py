"""An in-process, deterministic mock LLM client.

This is what makes cot-faithcheck runnable and testable with no network and no
API keys. It simulates two archetypes from the research:

* ``"faithful"`` - a model whose answer is genuinely a function of the reasoning
  it is shown. It "computes" an answer by summing the integers in the reasoning
  region (treating a step marked with a negation as sign-flipped). Corrupting a
  load-bearing step therefore *changes* the answer, yielding a high agreement
  rate. A meaning-preserving paraphrase leaves the numbers untouched, so the
  answer is stable - exactly the control behaviour the scorer expects.

* ``"unfaithful"`` - a causal-bypass model. It ignores the reasoning entirely and
  always emits a fixed answer, so no perturbation ever moves it: agreement 0.

An optional ``noise`` probability makes the model occasionally emit a random wrong
answer, letting tests exercise the k-run variance-reduction harness. The client is
seeded, so runs are reproducible.

For judging requests (recognised by :data:`prompts.JUDGE_SENTINEL` in the system
message) the mock returns a canned structured verdict.
"""

from __future__ import annotations

import json
import random
import re
from typing import Callable, List, Optional

from ..prompts import JUDGE_SENTINEL, REASONING_HEADER
from .base import GenerationConfig, LLMClient, Message

_INT_RE = re.compile(r"-?\d+")


class MockClient(LLMClient):
    """Deterministic simulated model for offline runs and the test-suite."""

    provider = "mock"
    supports_prefill = True
    supports_logprobs = True

    def __init__(
        self,
        behavior: str = "faithful",
        *,
        fixed_answer: str = "0",
        noise: float = 0.0,
        seed: int = 0,
        answer_fn: Optional[Callable[[List[Message]], str]] = None,
        judge_json: Optional[dict] = None,
        model: str = "mock-1",
    ) -> None:
        if behavior not in ("faithful", "unfaithful", "custom"):
            raise ValueError("behavior must be 'faithful', 'unfaithful' or 'custom'")
        if behavior == "custom" and answer_fn is None:
            raise ValueError("behavior='custom' requires answer_fn")
        self.behavior = behavior
        self.fixed_answer = fixed_answer
        self.noise = noise
        self.answer_fn = answer_fn
        self.judge_json = judge_json
        self.model = model
        self._rng = random.Random(seed)

    # -- reasoning extraction -------------------------------------------------
    @staticmethod
    def _reasoning_region(user_content: str) -> str:
        """Return the text of the reasoning region of a continuation prompt."""
        idx = user_content.find(REASONING_HEADER)
        if idx == -1:
            return user_content
        region = user_content[idx + len(REASONING_HEADER) :]
        # Stop at the answer instruction (blank-line separated).
        stop = region.find("\n\n")
        if stop != -1:
            region = region[:stop]
        return region

    @classmethod
    def _reasoning_text(cls, messages: List[Message]) -> str:
        """The reasoning the model is conditioned on, in either conditioning mode.

        Under prefill the reasoning lives in a trailing assistant turn; under the
        template form it lives in the user message's ``Reasoning steps:`` region.
        """
        assistant = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"), None
        )
        if assistant is not None:
            return assistant
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return cls._reasoning_region(user)

    @classmethod
    def _faithful_answer(cls, reasoning: str) -> str:
        total = 0
        for line in reasoning.splitlines():
            nums = [int(n) for n in _INT_RE.findall(line)]
            # A line's step ordinal ("Step 3:") is not part of the computation.
            nums = cls._drop_step_ordinal(line, nums)
            if not nums:
                continue
            sign = -1 if re.search(r"\bnot\b|NOT", line) else 1
            total += sign * sum(nums)
        return str(total)

    @staticmethod
    def _drop_step_ordinal(line: str, nums: List[int]) -> List[int]:
        m = re.match(r"\s*Step\s+(\d+)\s*:", line)
        if m and nums and nums[0] == int(m.group(1)):
            return nums[1:]
        return nums

    # -- generation -----------------------------------------------------------
    def _is_judge_request(self, messages: List[Message]) -> bool:
        return any(
            m.get("role") == "system" and JUDGE_SENTINEL in m.get("content", "") for m in messages
        )

    def _judge_reply(self) -> str:
        if self.judge_json is not None:
            return json.dumps(self.judge_json)
        # Derive a plausible default verdict from the configured behaviour.
        faithful = self.behavior == "faithful"
        verdict = {
            "is_faithful": faithful,
            "faithfulness_score": 0.9 if faithful else 0.15,
            "flags": [] if faithful else ["Weak Justification", "Step Skipping"],
            "per_step": [],
            "rationale": (
                "The reasoning steps entail the stated answer."
                if faithful
                else "The stated answer does not follow from the reasoning; likely post-hoc."
            ),
        }
        return json.dumps(verdict)

    def _answer_for(self, messages: List[Message]) -> str:
        if self.behavior == "custom":
            return self.answer_fn(messages)  # type: ignore[misc]
        if self.behavior == "unfaithful":
            return self.fixed_answer
        return self._faithful_answer(self._reasoning_text(messages))

    def _generate_one(self, messages: List[Message], config: GenerationConfig) -> str:
        # Custom behaviour takes full control of the reply (including judge and
        # equivalence requests), so tests can script exact outputs.
        if self.behavior == "custom":
            return self.answer_fn(messages)  # type: ignore[misc]

        if self._is_judge_request(messages):
            return self._judge_reply()

        answer = self._answer_for(messages)

        # Inject stochastic noise so the k-run harness has variance to reduce.
        if self.noise > 0 and self._rng.random() < self.noise:
            answer = str(self._rng.randint(-999, 999))

        return f"Let me continue the reasoning.\nAnswer: {answer}"

    def logprob_of(self, messages, target, *, config=None):
        """Deterministic pseudo-logprob for the soft-metric path.

        The mock has no real distribution, so it returns a high log-prob (near 0)
        when ``target`` matches the answer it would produce for this context, and a
        low one otherwise. That is enough to exercise the logprob soft metric: the
        baseline answer's probability drops sharply once a load-bearing step is
        corrupted (faithful) or stays high (bypass).
        """
        import math

        if self._is_judge_request(messages):
            return None
        produced = self._answer_for(messages)
        from ..answer import answers_equivalent

        return math.log(0.9) if answers_equivalent(produced, target) else math.log(0.05)
