"""A transparent wrapper that counts LLM calls and estimates token usage.

Wrapping the client at the pipeline boundary means every path — baseline sampling,
perturbation runs, early answering, position-bias probes, logprob queries, and the
judge — is accounted for in one place, without threading a counter through the
whole call graph. Token counts are deliberately rough (``~len/4`` characters per
token); they are labelled *estimated* everywhere they surface.
"""

from __future__ import annotations

import threading
from typing import List, Optional

from ..types import Usage
from .base import GenerationConfig, LLMClient, Message


def _est_tokens(text: str) -> int:
    """A cheap, provider-agnostic token estimate (~4 characters per token)."""
    return max(1, len(text) // 4) if text else 0


def _messages_tokens(messages: List[Message]) -> int:
    return sum(_est_tokens(m.get("content", "")) for m in messages)


class TrackingClient(LLMClient):
    """Delegates to an inner client while accumulating :class:`Usage`.

    Capability flags and the model identity are mirrored from the inner client so
    the wrapper is a drop-in replacement (prefill / logprob paths keep working).
    """

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.supports_prefill = getattr(inner, "supports_prefill", False)
        self.supports_logprobs = getattr(inner, "supports_logprobs", False)
        self._lock = threading.Lock()
        self._usage = Usage()

    # -- accounting -----------------------------------------------------------
    def usage(self) -> Usage:
        with self._lock:
            return Usage(
                n_calls=self._usage.n_calls,
                n_samples=self._usage.n_samples,
                est_prompt_tokens=self._usage.est_prompt_tokens,
                est_completion_tokens=self._usage.est_completion_tokens,
            )

    def _record(self, messages: List[Message], completions: List[str], n: int) -> None:
        prompt = _messages_tokens(messages)
        completion = sum(_est_tokens(c) for c in completions)
        with self._lock:
            self._usage.n_calls += 1
            self._usage.n_samples += n
            # The prompt is (re)sent once per sampled continuation.
            self._usage.est_prompt_tokens += prompt * max(1, n)
            self._usage.est_completion_tokens += completion

    # -- delegation -----------------------------------------------------------
    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop=(),
        n: int = 1,
    ) -> List[str]:
        out = self.inner.generate(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            n=n,
        )
        self._record(messages, out, n)
        return out

    def logprob_of(
        self, messages: List[Message], target: str, *, config: Optional[GenerationConfig] = None
    ) -> Optional[float]:
        lp = self.inner.logprob_of(messages, target, config=config)
        # A logprob query is one call with no sampled continuation.
        self._record(messages, [], 1)
        return lp

    def _generate_one(self, messages: List[Message], config: GenerationConfig) -> str:
        # Not used (generate is overridden), but required by the ABC.
        return self.inner._generate_one(messages, config)
