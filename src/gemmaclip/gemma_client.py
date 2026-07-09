from __future__ import annotations

import base64
import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

DEFAULT_PROVIDER_FIREWORKS = "fireworks"
DEFAULT_PROVIDER_GOOGLE = "google"
VALID_PROVIDERS = {DEFAULT_PROVIDER_FIREWORKS, DEFAULT_PROVIDER_GOOGLE}
DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_GEMMA_MODEL = "accounts/fireworks/models/gemma-4-31b-it"
DEFAULT_GEMMA_VISION_MODEL = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_GEMMA_TEXT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
DEFAULT_FALLBACK_MODELS = ("accounts/fireworks/models/kimi-k2p6",)
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMMA_MAX_TOKENS = 2048
DEFAULT_TOP_K = 40
DEFAULT_GOOGLE_MAX_RETRIES = 3
LOGGER = logging.getLogger("gemmaclip.gemma_client")


@dataclass(frozen=True, slots=True)
class GemmaModelConfig:
    api_key: str
    base_url: str | None
    model: str
    fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    max_tokens: int = DEFAULT_GEMMA_MAX_TOKENS
    use_response_format: bool = False
    timeout_seconds: float = 60.0
    provider: str = DEFAULT_PROVIDER_FIREWORKS

    @property
    def model_candidates(self) -> tuple[str, ...]:
        if self.provider != DEFAULT_PROVIDER_FIREWORKS:
            return (self.model,)
        return (self.model, *self.fallback_models)


@dataclass(frozen=True, slots=True)
class GemmaConfig:
    api_key: str
    base_url: str | None
    vision_model: str
    text_model: str
    fallback_models: tuple[str, ...] = DEFAULT_FALLBACK_MODELS
    max_tokens: int = DEFAULT_GEMMA_MAX_TOKENS
    use_response_format: bool = False
    timeout_seconds: float = 60.0
    provider: str = DEFAULT_PROVIDER_FIREWORKS

    def vision_model_config(self) -> GemmaModelConfig:
        return GemmaModelConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.vision_model,
            fallback_models=(
                _filter_fallback_models(self.fallback_models, primary_model=self.vision_model)
                if self.provider == DEFAULT_PROVIDER_FIREWORKS
                else ()
            ),
            max_tokens=self.max_tokens,
            use_response_format=False,
            timeout_seconds=self.timeout_seconds,
            provider=self.provider,
        )

    def text_model_config(self) -> GemmaModelConfig:
        return GemmaModelConfig(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.text_model,
            fallback_models=(
                _filter_fallback_models(self.fallback_models, primary_model=self.text_model)
                if self.provider == DEFAULT_PROVIDER_FIREWORKS
                else ()
            ),
            max_tokens=self.max_tokens,
            use_response_format=self.use_response_format,
            timeout_seconds=self.timeout_seconds,
            provider=self.provider,
        )


def load_gemma_config(env: Mapping[str, str] | None = None) -> GemmaConfig | None:
    values = env if env is not None else os.environ
    provider = _resolve_provider(values)

    if provider == DEFAULT_PROVIDER_GOOGLE:
        api_key = values.get("GEMINI_API_KEY", "").strip() or values.get("GOOGLE_API_KEY", "").strip()
        model = values.get("GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_MODEL
        if not api_key:
            return None
        return GemmaConfig(
            api_key=api_key,
            base_url=None,
            vision_model=model,
            text_model=model,
            fallback_models=(),
            max_tokens=_parse_max_tokens(values.get("GEMMA_MAX_TOKENS")),
            use_response_format=False,
            provider=provider,
        )

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
        provider=provider,
    )


def create_model_client(config: GemmaModelConfig) -> GemmaClient | GoogleGeminiClient:
    if config.provider == DEFAULT_PROVIDER_GOOGLE:
        return GoogleGeminiClient(config)
    return GemmaClient(config)


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
        if not self._config.base_url:
            raise RuntimeError("Fireworks client requires a base URL.")

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


class GoogleGeminiClient:
    def __init__(
        self,
        config: GemmaModelConfig,
        client: Any | None = None,
        *,
        sleeper: Any = time.sleep,
    ) -> None:
        self._config = config
        self._client = client or _build_google_client(config)
        self._sleeper = sleeper

    def chat_completion_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        *,
        use_response_format: bool | None = None,
    ) -> str:
        types = _load_google_types()
        contents, system_instruction = _convert_messages_to_google_contents(messages, types)
        request_config = _build_google_generation_config(
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=self._config.max_tokens,
            use_json=_google_use_json_response(use_response_format),
        )

        last_error: Exception | None = None
        for attempt in range(DEFAULT_GOOGLE_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self._config.model,
                    contents=contents,
                    config=request_config,
                )
                return _extract_google_response_text(response)
            except Exception as exc:  # pragma: no cover - vendor exceptions vary by SDK version
                last_error = exc
                status_code = _extract_exception_status_code(exc)
                if status_code is None or not _is_retryable_google_status(status_code) or attempt == DEFAULT_GOOGLE_MAX_RETRIES - 1:
                    raise RuntimeError(f"Google Gemini request failed for model {self._config.model}: {exc}") from exc
                delay_seconds = float(2 ** attempt)
                LOGGER.warning(
                    "Google Gemini request for model %s failed with status %s; retrying in %.1fs.",
                    self._config.model,
                    status_code,
                    delay_seconds,
                )
                self._sleeper(delay_seconds)

        raise RuntimeError(f"Google Gemini request failed for model {self._config.model}: {last_error}")

    def chat_completion_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        *,
        use_response_format: bool | None = None,
    ) -> dict[str, Any]:
        text = self.chat_completion_text(
            messages,
            temperature,
            use_response_format=use_response_format,
        )
        return parse_json_object(text)


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


def _resolve_provider(values: Mapping[str, str]) -> str:
    requested_provider = values.get("GEMMACLIP_PROVIDER", "").strip().lower()
    if requested_provider in VALID_PROVIDERS:
        return requested_provider
    if values.get("GEMINI_API_KEY", "").strip() or values.get("GOOGLE_API_KEY", "").strip():
        return DEFAULT_PROVIDER_GOOGLE
    return DEFAULT_PROVIDER_FIREWORKS


def _build_google_client(config: GemmaModelConfig) -> Any:
    genai = _load_google_genai_module()
    return genai.Client(api_key=config.api_key)


def _load_google_genai_module():
    try:
        from google import genai
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install state
        raise RuntimeError("google-genai is required for the Google Gemini provider.") from exc
    return genai


def _load_google_types():
    try:
        from google.genai import types
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install state
        raise RuntimeError("google-genai is required for the Google Gemini provider.") from exc
    return types


def _convert_messages_to_google_contents(messages: Sequence[Mapping[str, Any]], types) -> tuple[list[Any], str | None]:
    system_fragments: list[str] = []
    contents: list[Any] = []

    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")
        if role == "system":
            system_text = _flatten_message_text(content)
            if system_text:
                system_fragments.append(system_text)
            continue

        contents.extend(_convert_message_content_to_google_parts(content, types))

    if not contents:
        raise ValueError("Google Gemini request did not include any user content.")

    system_instruction = "\n\n".join(fragment for fragment in system_fragments if fragment).strip() or None
    return contents, system_instruction


def _flatten_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    fragments: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            fragments.append(text.strip())
    return "\n".join(fragments).strip()


def _convert_message_content_to_google_parts(content: Any, types) -> list[Any]:
    if isinstance(content, str):
        return [content.strip()] if content.strip() else []
    if not isinstance(content, list):
        return []

    parts: list[Any] = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
            continue
        if item_type == "image_url":
            image_url = item.get("image_url")
            if not isinstance(image_url, Mapping):
                continue
            url = image_url.get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            mime_type, data = _decode_image_payload(url.strip())
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
            continue
        mime_type = item.get("mime_type")
        data = item.get("data")
        if isinstance(mime_type, str) and isinstance(data, bytes):
            parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
            continue
        if isinstance(mime_type, str) and isinstance(data, str):
            parts.append(types.Part.from_bytes(data=base64.b64decode(data), mime_type=mime_type))
    return parts


def _decode_image_payload(value: str) -> tuple[str, bytes]:
    if not value.startswith("data:"):
        raise ValueError("Google Gemini image input must be a data URL.")
    header, encoded = value.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Google Gemini image data URL must be base64 encoded.")
    mime_type = header[5:].split(";", 1)[0].strip() or "image/jpeg"
    return mime_type, base64.b64decode(encoded)


def _build_google_generation_config(
    *,
    system_instruction: str | None,
    temperature: float,
    max_tokens: int,
    use_json: bool,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "top_k": DEFAULT_TOP_K,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    if system_instruction:
        config["system_instruction"] = system_instruction
    if use_json:
        config["response_mime_type"] = "application/json"
    return config


def _google_use_json_response(use_response_format: bool | None) -> bool:
    return True


def _extract_google_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list):
        fragments: list[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if not isinstance(parts, list):
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    fragments.append(part_text.strip())
        joined = "\n".join(fragments).strip()
        if joined:
            return joined

    raise ValueError("Google Gemini response did not contain text.")


def _extract_exception_status_code(exc: Exception) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


def _is_retryable_google_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600


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
