"""OpenAI-compatible chat-completions client.

Works with the OpenAI API and any endpoint that mirrors ``/v1/chat/completions``
(vLLM, Together, Groq, Fireworks, LM Studio, OpenRouter, ...). Only the base URL,
API key, and model change. Uses the standard library over HTTP, so the ``openai``
SDK is not required — install the ``openai`` extra only if you prefer it.
"""

from __future__ import annotations

import os
from typing import List, Optional

from ..errors import ClientConfigError, ClientError
from ._http import post_json
from .base import GenerationConfig, LLMClient, Message


class OpenAICompatibleClient(LLMClient):
    """Chat-completions client for any OpenAI-shaped endpoint."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        organization: Optional[str] = None,
        native_n: bool = True,
        supports_prefill: bool = False,
        supports_logprobs: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.organization = organization
        self._native_n = native_n
        # The hosted OpenAI API does not continue a trailing assistant message, but
        # many OpenAI-compatible servers (vLLM, SGLang, ...) do via
        # ``continue_final_message``. Enable per-endpoint.
        self.supports_prefill = supports_prefill
        self.supports_logprobs = supports_logprobs

    def _headers(self) -> dict:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    def _payload(self, messages: List[Message], config: GenerationConfig, n: int) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "n": n,
        }
        if config.stop:
            payload["stop"] = list(config.stop)
        # Continue a trailing assistant turn (true prefill) when the endpoint
        # supports it, instead of starting a fresh assistant message.
        if self.supports_prefill and messages and messages[-1].get("role") == "assistant":
            payload["continue_final_message"] = True
            payload["add_generation_prompt"] = False
        return payload

    def _request(self, messages: List[Message], config: GenerationConfig, n: int) -> List[str]:
        resp = post_json(
            f"{self.base_url}/chat/completions",
            self._payload(messages, config, n),
            headers=self._headers(),
            timeout=self.timeout,
        )
        choices = resp.get("choices")
        if not choices:
            raise ClientError(f"no choices in response: {str(resp)[:300]}")
        out = []
        for ch in choices:
            content = (ch.get("message") or {}).get("content")
            out.append(content or "")
        return out

    def _supports_native_n(self) -> bool:
        return self._native_n

    def _generate_n(self, messages: List[Message], config: GenerationConfig, n: int) -> List[str]:
        return self._request(messages, config, n)

    def _generate_one(self, messages: List[Message], config: GenerationConfig) -> str:
        return self._request(messages, config, 1)[0]

    def logprob_of(self, messages, target, *, config=None):
        """Best-effort P(``target``) from the first answer token's log-probability.

        Follows the standard answer-tracing recipe: constrain the model to emit only
        the answer, request ``logprobs``/``top_logprobs``, and read the probability
        the first generated token position assigns to the target's leading token
        (matching the bare and leading-space variants). Works well for
        multiple-choice letters and short answers. Returns ``None`` when the
        endpoint does not return token log-probabilities.
        """
        if not self.supports_logprobs:
            return None
        msgs = list(messages) + [
            {"role": "user", "content": "Respond with ONLY the final answer, nothing else."}
        ]
        payload = {
            "model": self.model,
            "messages": msgs,
            "temperature": 0.0,
            "max_tokens": 6,
            "logprobs": True,
            "top_logprobs": 20,
            "n": 1,
        }
        try:
            resp = post_json(
                f"{self.base_url}/chat/completions",
                payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except ClientError:
            return None
        content = ((resp.get("choices") or [{}])[0].get("logprobs") or {}).get("content") or []
        if not content:
            return None
        first = content[0]
        top = first.get("top_logprobs") or [
            {"token": first.get("token"), "logprob": first.get("logprob")}
        ]
        want = target.strip().lower()[:1]
        if not want:
            return None
        best = None
        for entry in top:
            tok = str(entry.get("token", "")).strip().lower()
            if tok[:1] == want:
                lp = entry.get("logprob")
                if lp is not None and (best is None or lp > best):
                    best = lp
        return best

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        model = os.environ.get("COT_FAITHCHECK_MODEL") or os.environ.get("OPENAI_MODEL")
        if not model:
            raise ClientConfigError(
                "set COT_FAITHCHECK_MODEL (or OPENAI_MODEL) to an OpenAI-compatible model id"
            )
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(model, base_url=base_url)
