from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_GEMMA_MAX_TOKENS = 2048
DEFAULT_TOP_K = 40


@dataclass(frozen=True, slots=True)
class GemmaConfig:
    api_key: str
    base_url: str
    model: str
    max_tokens: int = DEFAULT_GEMMA_MAX_TOKENS
    use_response_format: bool = False
    timeout_seconds: float = 60.0


def load_gemma_config(env: Mapping[str, str] | None = None) -> GemmaConfig | None:
    values = env if env is not None else os.environ
    api_key = values.get("GEMMA_API_KEY", "").strip() or values.get("FIREWORKS_API_KEY", "").strip()
    base_url = values.get("GEMMA_BASE_URL", "").strip() or DEFAULT_FIREWORKS_BASE_URL
    model = values.get("GEMMA_MODEL", "").strip()
    max_tokens = _parse_max_tokens(values.get("GEMMA_MAX_TOKENS"))
    use_response_format = _parse_bool(values.get("GEMMA_USE_RESPONSE_FORMAT"))
    if not api_key or not base_url or not model:
        return None
    return GemmaConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        use_response_format=use_response_format,
    )


class GemmaClient:
    def __init__(self, config: GemmaConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    def chat_completion_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
    ) -> dict[str, Any]:
        payload = self._post_chat_completion(messages=messages, temperature=temperature)
        content = extract_message_text(payload)
        return parse_json_object(content)

    def _post_chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
    ) -> dict[str, Any]:
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        request_payload = build_chat_completion_payload(self._config, messages=messages, temperature=temperature)
        try:
            response = self._client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Gemma API request failed: {exc}") from exc

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Gemma API returned invalid JSON.") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Gemma API returned an unexpected response shape.")
        return payload


def build_chat_completion_payload(
    config: GemmaConfig,
    *,
    messages: Sequence[Mapping[str, Any]],
    temperature: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": config.max_tokens,
        "top_k": DEFAULT_TOP_K,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    if config.use_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def extract_message_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Gemma response did not include any choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, Mapping):
        raise ValueError("Gemma response choice is not an object.")

    message = first_choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("Gemma response did not include a message object.")

    content = message.get("content")
    if isinstance(content, str):
        text = content.strip()
        if not text:
            raise ValueError("Gemma response content was empty.")
        return text

    if isinstance(content, list):
        fragments: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                fragments.append(text_value.strip())
        joined = "\n".join(fragments).strip()
        if joined:
            return joined

    raise ValueError("Gemma response content did not contain text.")


def parse_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidate = text.strip()

    fenced = _extract_fenced_json(candidate)
    if fenced is not None:
        candidate = fenced

    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not extract a JSON object from the model response.")


def _extract_fenced_json(text: str) -> str | None:
    marker = "```"
    if marker not in text:
        return None

    for block in text.split(marker):
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped
    return None


def _parse_max_tokens(value: str | None) -> int:
    if value is None or not value.strip():
        return DEFAULT_GEMMA_MAX_TOKENS
    try:
        parsed = int(value)
    except ValueError:
        return DEFAULT_GEMMA_MAX_TOKENS
    return parsed if parsed > 0 else DEFAULT_GEMMA_MAX_TOKENS


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() == "true"
