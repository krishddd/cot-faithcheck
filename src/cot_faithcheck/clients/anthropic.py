"""Anthropic Messages API client.

Talks to ``/v1/messages`` over HTTP (no ``anthropic`` SDK required). Anthropic has
no server-side ``n`` parameter, so k-run sampling is done with k sequential
requests via the base class.
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..errors import ClientConfigError, ClientError
from ._http import post_json
from .base import GenerationConfig, LLMClient, Message

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient(LLMClient):
    """Client for Anthropic's Messages API."""

    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com/v1",
        timeout: float = 60.0,
        anthropic_version: str = _ANTHROPIC_VERSION,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.timeout = timeout
        self.anthropic_version = anthropic_version

    def _split_system(self, messages: List[Message]):
        """Anthropic takes ``system`` as a top-level field, not a message role."""
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        convo = [m for m in messages if m.get("role") != "system"]
        return ("\n\n".join(system_parts), convo)

    def _generate_one(self, messages: List[Message], config: GenerationConfig) -> str:
        system, convo = self._split_system(messages)
        payload = {
            "model": self.model,
            "messages": convo,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        if system:
            payload["system"] = system
        if config.stop:
            payload["stop_sequences"] = list(config.stop)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
        }
        resp = post_json(
            f"{self.base_url}/messages", payload, headers=headers, timeout=self.timeout
        )
        blocks = resp.get("content")
        if not blocks:
            raise ClientError(f"no content in response: {str(resp)[:300]}")
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        return text

    @classmethod
    def from_env(cls) -> "AnthropicClient":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ClientConfigError("set ANTHROPIC_API_KEY to use the Anthropic client")
        model = os.environ.get("COT_FAITHCHECK_MODEL", "claude-sonnet-5")
        return cls(model)
