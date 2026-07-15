"""Provider-agnostic LLM clients.

All clients implement the small :class:`LLMClient` interface (``generate`` returns
n text completions), so the faithfulness pipeline is provider-agnostic:

    from cot_faithcheck.clients import client_from_env, MockClient

    client = client_from_env()          # picks a provider from env vars
    client = MockClient("faithful")     # offline, deterministic
"""

from __future__ import annotations

import os

from ..errors import ClientConfigError
from .anthropic import AnthropicClient
from .base import GenerationConfig, LLMClient, Message
from .mock import MockClient
from .ollama import OllamaClient
from .openai_compat import OpenAICompatibleClient

__all__ = [
    "LLMClient",
    "GenerationConfig",
    "Message",
    "MockClient",
    "OpenAICompatibleClient",
    "AnthropicClient",
    "OllamaClient",
    "client_from_env",
]

_PROVIDERS = {
    "openai": OpenAICompatibleClient,
    "openai_compatible": OpenAICompatibleClient,
    "anthropic": AnthropicClient,
    "ollama": OllamaClient,
    "mock": MockClient,
}


def client_from_env() -> LLMClient:
    """Construct a client from environment variables.

    ``COT_FAITHCHECK_PROVIDER`` selects the provider (``openai`` | ``anthropic`` |
    ``ollama`` | ``mock``). When unset, the provider is inferred: an
    ``ANTHROPIC_API_KEY`` implies Anthropic, an ``OPENAI_API_KEY`` implies OpenAI,
    otherwise Ollama (assumed running locally). See each client's ``from_env``.
    """
    provider = os.environ.get("COT_FAITHCHECK_PROVIDER", "").strip().lower()
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "ollama"

    if provider == "mock":
        return MockClient(os.environ.get("COT_FAITHCHECK_MOCK_BEHAVIOR", "faithful"))
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise ClientConfigError(
            f"unknown provider {provider!r}; expected one of {sorted(_PROVIDERS)}"
        )
    return cls.from_env()  # type: ignore[attr-defined]
