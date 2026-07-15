"""The small adapter interface every LLM provider plugs into.

A client's only job is to turn a list of chat messages into text completions. The
faithfulness pipeline never touches provider SDKs directly — it speaks to this
interface, so OpenAI-compatible endpoints, Anthropic, Ollama, or an in-process
mock are all interchangeable.

    client = client_from_env()            # or OpenAICompatibleClient(...), etc.
    texts = client.generate(messages, temperature=0.7, n=5)

``generate`` returns *n* independent samples (the k-run harness asks for k). A
provider that cannot sample server-side falls back to *n* sequential calls via
:meth:`LLMClient._sample_sequentially`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Dict, List, Sequence

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


@dataclass
class GenerationConfig:
    """Decoding parameters shared across providers."""

    temperature: float = 0.7
    max_tokens: int = 512
    top_p: float = 1.0
    stop: Sequence[str] = ()


class LLMClient(abc.ABC):
    """Abstract provider adapter.

    Subclasses implement :meth:`_generate_one`; the base class handles n-sampling
    and exposes the public :meth:`generate`.
    """

    #: Human-readable provider name, surfaced in reports.
    provider: str = "llm"
    #: The model identifier passed to the provider.
    model: str = ""

    @abc.abstractmethod
    def _generate_one(self, messages: List[Message], config: GenerationConfig) -> str:
        """Return a single completion string for ``messages``."""

    def _supports_native_n(self) -> bool:
        """Whether the provider can return *n* samples in one request."""
        return False

    def _generate_n(self, messages: List[Message], config: GenerationConfig, n: int) -> List[str]:
        """Override when the provider supports native n-sampling."""
        return self._sample_sequentially(messages, config, n)

    def _sample_sequentially(
        self, messages: List[Message], config: GenerationConfig, n: int
    ) -> List[str]:
        return [self._generate_one(messages, config) for _ in range(n)]

    def generate(
        self,
        messages: List[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 1.0,
        stop: Sequence[str] = (),
        n: int = 1,
    ) -> List[str]:
        """Return ``n`` completions for ``messages``.

        The k-run harness calls this with ``n=k``; at ``temperature=0`` all
        samples are (approximately) identical and callers may pass ``n=1``.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        config = GenerationConfig(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, stop=tuple(stop)
        )
        if n == 1:
            return [self._generate_one(messages, config)]
        if self._supports_native_n():
            return self._generate_n(messages, config, n)
        return self._sample_sequentially(messages, config, n)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(provider={self.provider!r}, model={self.model!r})"
