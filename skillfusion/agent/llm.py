"""Nemotron client for Nebius Token Factory (OpenAI-compatible). Only needs httpx.

Nemotron 3 models are *reasoning* models: the answer normally arrives in ``content`` but may be
empty (all tokens spent thinking) with the text in ``reasoning_content``. We handle both, and
we extract JSON defensively so a chatty model does not break the agent loop.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
# Verify exact ids in the Token Factory model catalogue; override with NEMOTRON_MODEL.
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
FAST_MODEL = "nvidia/nvidia-nemotron-3-nano-30b-a3b"


class LLMError(RuntimeError):
    pass


@dataclass
class NemotronClient:
    api_key: str | None = None
    model: str = field(default_factory=lambda: os.environ.get("NEMOTRON_MODEL", DEFAULT_MODEL))
    base_url: str = field(default_factory=lambda: os.environ.get("NEBIUS_BASE_URL", BASE_URL))
    timeout: float = 120.0
    transport: Any = None          # httpx transport, injectable for tests
    temperature: float = 0.2

    def __post_init__(self):
        self.api_key = self.api_key or os.environ.get("NEBIUS_API_KEY")
        self._http = httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self.transport)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict], *, max_tokens: int = 2048, model: str | None = None,
             json_mode: bool = False, thinking: bool = False) -> str:
        if not self.api_key:
            raise LLMError("NEBIUS_API_KEY not set")
        body: dict[str, Any] = {"model": model or self.model, "messages": messages,
                                "temperature": self.temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        # Nemotron 3 toggles reasoning via chat_template_kwargs on OpenAI-compatible servers.
        body["chat_template_kwargs"] = {"enable_thinking": bool(thinking)}
        r = self._http.post("chat/completions", json=body,
                            headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code >= 400 and ("chat_template_kwargs" in r.text or "response_format" in r.text):
            body.pop("chat_template_kwargs", None)             # server rejected an optional field
            if json_mode:
                body.pop("response_format", None)
            r = self._http.post("chat/completions", json=body,
                                headers={"Authorization": f"Bearer {self.api_key}"})
        if r.status_code >= 400:
            raise LLMError(f"Token Factory {r.status_code}: {r.text[:300]}")
        return extract_text(r.json())

    def chat_json(self, messages: list[dict], *, retries: int = 2, **kw) -> Any:
        msgs = list(messages)
        last = ""
        for _ in range(retries + 1):
            last = self.chat(msgs, json_mode=True, **kw)
            try:
                return parse_json(last)
            except ValueError as e:
                msgs = msgs + [{"role": "assistant", "content": last},
                               {"role": "user", "content": f"That was not valid JSON ({e}). Reply with JSON only."}]
        raise LLMError(f"model never produced valid JSON: {last[:200]!r}")


def extract_text(resp: dict) -> str:
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"malformed response: {str(resp)[:200]}") from e
    text = (msg.get("content") or "").strip()
    if not text:
        text = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if not text:
        raise LLMError("empty completion (raise max_tokens or disable thinking)")
    return text


def parse_json(text: str) -> Any:
    """Parse the first JSON object/array in ``text`` (handles ```json fences and prose around it)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, flags=re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(t[i:])
                return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found")
