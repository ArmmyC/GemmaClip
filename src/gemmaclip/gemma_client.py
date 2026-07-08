from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_GEMMA_MODEL = "accounts/fireworks/models/gemma-4-31b-it"
DEFAULT_GEMMA_VISION_MODEL = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_GEMMA_TEXT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
DEFAULT_FALLBACK_MODELS = ("accounts/fireworks/models/kimi-k2p6",)
DEFAULT_GEMMA_MAX_TOKENS = 2048
DEFAULT_TOP_K = 40
LOGGER = logging.getLogger("gemmaclip.gemma_client")


@dataclass(frozen=True, slots=True)
class GemmaModelConfig:
    api_key: str
    base_url: str
    model: str
    fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    max_tokens: int = DEFAULT_GEMMA_MAX_TOKENS
    use_response_format: bool = False
    timeout_seconds: float = 60.0

    @property
    def model_candidates(self) -> tuple[str, ...]:
        return (self.model, *self.fallback_models)


@dataclass(frozen=True, slots=True)
class GemmaConfig:
    api_key: str
    base_url: str
    vision_model: str
    text_model: str
    fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    max_tokens: int = DEFAULT_GEMMA_MAX_TOKENS
    use_response_format: bool = False
    timeout_seconds: float = 60.0

    def vision_model_config(self) -> GemmaModelConfig:
        return GemmaModelConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.vision_model,
            fallback_models=_filter_fallback_models(self.fallback_models, primary_model=self.vision_model),
            max_tokens=self.max_tokens,
            use_response_format=False,
            timeout_seconds=self.timeout_seconds,
        )

    def text_model_config(self) -> GemmaModelConfig:
        return GemmaModelConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.text_model,
            fallback_models=_filter_fallback_models(self.fallback_models, primary_model=self.text_model),
            max_tokens=self.max_tokens,
            use_response_format=self.use_response_format,
            timeout_seconds=self.timeout_seconds,
        )


def load_gemma_config(env: Mapping[str, str] | None = None) -> GemmaConfig | None:
    values = env if env is not None else os.environ
    api_key = values.get("GEMMA_API_KEY", "").strip() or values.get("FIREWORKS_API_KEY", "").strip()
    base_url = values.get("GEMMA_BASE_URL", "").strip() or DEFAULT_FIREWORKS_BASE_URL
    model_override = values.get("GEMMA_MODEL", "").strip()
    vision_model = model_override or values.get("GEMMA_VISION_MODEL", "").strip() or DEFAULT_GEMMA_VISION_MODEL
    text_model = model_override or values.get("GEMMA_TEXT_MODEL", "").strip() or DEFAULT_GEMMA_TEXT_MODEL
    fallback_models = _parse_fallback_models(values.get("GEMMA_FALLBACK_MODELS"))
    max_tokens = _parse_max_tokens(values.get("GEMMA_MAX_TOKENS"))
    use_response_format = _parse_bool(values.get("GEMMA_USE_RESPONSE_FORMAT"))
    if not api_key or not base_url:
        return None
    return GemmaConfig(
        api_key=api_key,
        base_url=base_url,
        vision_model=vision_model,
        text_model=text_model,
        fallback_models=fallback_models,
        max_tokens=max_tokens,
        use_response_format=use_response_format,
    )


class GemmaClient:
    def __init__(self, config: GemmaModelConfig, client: httpx.Client | None = None) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=config.timeout_seconds)
        self._working_model: str | None = None

    def chat_completion_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        *,
        use_response_format: bool | None = None,
    ) -> str:
        payload = self._post_chat_completion(
            messages=messages,
            temperature=temperature,
            use_response_format=use_response_format,
        )
        return extract_message_text(payload)

    def chat_completion_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        *,
        use_response_format: bool | None = None,
    ) -> dict[str, Any]:
        content = self.chat_completion_text(
            messages=messages,
            temperature=temperature,
            use_response_format=use_response_format,
        )
        return parse_json_object(content)

    def _post_chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        *,
        use_response_format: bool | None = None,
    ) -> dict[str, Any]:
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        attempted_failures: list[tuple[str, httpx.HTTPError]] = []
        candidate_models = self._ordered_model_candidates()
        for model in candidate_models:
            request_payload = build_chat_completion_payload(
                self._config,
                messages=messages,
                temperature=temperature,
                model=model,
                use_response_format=use_response_format,
            )
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
                if _should_try_next_model(exc) and model != candidate_models[-1]:
                    attempted_failures.append((model, exc))
                    continue
                raise RuntimeError(f"Gemma API request failed for model {model}: {exc}") from exc

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise RuntimeError("Gemma API returned invalid JSON.") from exc

            if not isinstance(payload, dict):
                raise RuntimeError("Gemma API returned an unexpected response shape.")

            if attempted_failures:
                LOGGER.warning(
                    "Gemma model fallback selected %s after failures from %s.",
                    model,
                    ", ".join(failed_model for failed_model, _ in attempted_failures),
                )
            self._working_model = model
            return payload

        if attempted_failures:
            failed_models = ", ".join(model for model, _ in attempted_failures)
            last_error = attempted_failures[-1][1]
            raise RuntimeError(f"Gemma API request failed for configured models: {failed_models}") from last_error

        raise RuntimeError("Gemma API request failed before any model request was attempted.")

    def _ordered_model_candidates(self) -> tuple[str, ...]:
        candidates = list(self._config.model_candidates)
        if self._working_model and self._working_model in candidates:
            return (self._working_model, *(candidate for candidate in candidates if candidate != self._working_model))
        return tuple(candidates)


def build_chat_completion_payload(
    config: GemmaModelConfig,
    *,
    messages: Sequence[Mapping[str, Any]],
    temperature: float,
    model: str | None = None,
    use_response_format: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model or config.model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": config.max_tokens,
        "top_k": DEFAULT_TOP_K,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    if use_response_format if use_response_format is not None else config.use_response_format:
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


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for candidate in _extract_json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def parse_json_object(text: str) -> dict[str, Any]:
    for candidate in reversed(extract_json_objects(text)):
        if candidate:
            return candidate
    raise ValueError("Could not extract a non-empty JSON object from the model response.")


def _extract_json_candidates(text: str) -> list[str]:
    candidates = _extract_fenced_json_blocks(text)
    candidates.extend(_extract_balanced_brace_objects(text))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        stripped = candidate.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        deduped.append(stripped)
    return deduped


def _extract_fenced_json_blocks(text: str) -> list[str]:
    marker = "```"
    if marker not in text:
        return []

    blocks: list[str] = []
    for block in text.split(marker):
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            blocks.append(stripped)
    return blocks


def _extract_balanced_brace_objects(text: str) -> list[str]:
    candidates: list[str] = []
    start_index: int | None = None
    depth = 0
    in_string = False
    escape_next = False

    for index, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == "\\" and in_string:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            if depth == 0:
                start_index = index
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_index is not None:
                candidates.append(text[start_index:index + 1])
                start_index = None

    return candidates


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


def _parse_fallback_models(value: str | None) -> tuple[str, ...]:
    raw_models = DEFAULT_FALLBACK_MODELS if value is None or not value.strip() else tuple(
        item.strip() for item in value.split(",")
    )
    normalized: list[str] = []
    for model in raw_models:
        if not model or model in normalized:
            continue
        normalized.append(model)
    return tuple(normalized)


def _filter_fallback_models(
    fallback_models: Sequence[str],
    *,
    primary_model: str,
) -> tuple[str, ...]:
    return tuple(model for model in fallback_models if model != primary_model)


def _should_try_next_model(exc: httpx.HTTPError) -> bool:
    if not isinstance(exc, httpx.HTTPStatusError):
        return False

    status_code = exc.response.status_code
    body = exc.response.text.lower()
    if status_code in {401, 403, 404, 422}:
        return True
    if status_code != 400:
        return False

    return any(
        phrase in body
        for phrase in (
            "model",
            "unavailable",
            "unsupported",
            "not found",
            "unauthorized",
            "forbidden",
            "permission",
        )
    )
