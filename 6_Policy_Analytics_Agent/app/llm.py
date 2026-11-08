from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Protocol

import httpx

from .config import Settings, get_settings
from .errors import LLMError

logger = logging.getLogger("agent.llm")

Messages = list[dict[str, str]]


class LLMClient(Protocol):
    def chat(self, messages: Messages, max_tokens: int = 512, temperature: float = 0.0) -> str: ...


class OpenRouterClient:
    """Minimal OpenRouter client with retries on 429/5xx and optional model fallbacks."""

    def __init__(self, settings: Settings | None = None, http_client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self._http = http_client or httpx.Client(timeout=self.settings.llm_timeout_s)

    @property
    def configured(self) -> bool:
        return bool(self.settings.openrouter_api_key)

    def chat(self, messages: Messages, max_tokens: int = 512, temperature: float = 0.0) -> str:
        s = self.settings
        if not s.openrouter_api_key:
            raise LLMError("OPENROUTER_API_KEY is not set.")

        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if s.openrouter_fallback_models:
            payload["models"] = [s.openrouter_model, *s.openrouter_fallback_models]
        else:
            payload["model"] = s.openrouter_model

        headers = {
            "Authorization": f"Bearer {s.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",       # optional attribution headers
            "X-Title": "Policy-Aware Analytics Agent",
        }
        url = f"{s.openrouter_base_url.rstrip('/')}/chat/completions"

        last_error = "unknown error"
        for attempt in range(s.llm_http_retries + 1):
            try:
                resp = self._http.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = f"network error: {type(exc).__name__}"
            else:
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = f"HTTP {resp.status_code}"
                elif resp.status_code >= 400:
                    raise LLMError(f"OpenRouter HTTP {resp.status_code}: {resp.text[:200]}")
                else:
                    return self._parse(resp)
            if attempt < s.llm_http_retries:
                delay = 1.5 * (attempt + 1)
                logger.warning("LLM call failed (%s); retrying in %.1fs", last_error, delay)
                time.sleep(delay)
        raise LLMError(f"OpenRouter request failed: {last_error}")

    @staticmethod
    def _parse(resp: httpx.Response) -> str:
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError("OpenRouter returned a non-JSON response.") from exc
        if isinstance(data, dict) and data.get("error"):
            raise LLMError(f"OpenRouter error: {str(data['error'])[:200]}")
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected OpenRouter response shape.") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("The model returned an empty response.")
        return content.strip()


def parse_json_object(text: str) -> dict:
    
    for candidate in re.findall(r"\{.*?\}", text or "", re.DOTALL):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    # nested objects/lists defeat the non-greedy regex; try the outermost braces too
    start, end = (text or "").find("{"), (text or "").rfind("}")
    if 0 <= start < end:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
    return {}
