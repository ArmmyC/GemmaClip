from __future__ import annotations

import base64
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx

DEFAULT_PROVIDER_FIREWORKS = "fireworks"
DEFAULT_PROVIDER_GOOGLE = "google"
DEFAULT_PROVIDER_OPENROUTER = "openrouter"
DEFAULT_PROVIDER_FIREWORKS_JUDGE = "fireworks_judge"
DEFAULT_PROVIDER_ROUTED_GEMMA = "routed_gemma"
DEFAULT_PROVIDER_AMD_CLOUD = "amd_cloud"
VALID_PROVIDERS = {
    DEFAULT_PROVIDER_FIREWORKS,
    DEFAULT_PROVIDER_GOOGLE,
    DEFAULT_PROVIDER_OPENROUTER,
    DEFAULT_PROVIDER_FIREWORKS_JUDGE,
    DEFAULT_PROVIDER_ROUTED_GEMMA,
}
DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_GEMMA_MODEL = "accounts/fireworks/models/gemma-4-31b-it"
DEFAULT_GEMMA_VISION_MODEL = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_GEMMA_TEXT_MODEL = "accounts/fireworks/models/deepseek-v4-pro"
DEFAULT_FALLBACK_MODELS = ("accounts/fireworks/models/kimi-k2p6",)
DEFAULT_GOOGLE_GEMMA_MODEL = "gemma-4-26b-a4b-it"
DEFAULT_FIREWORKS_JUDGE_VISION_MODEL = "accounts/fireworks/models/minimax-m3"
DEFAULT_FIREWORKS_JUDGE_FALLBACK_VISION_MODEL = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_FIREWORKS_GEMMA_VISUAL_MODEL = "accounts/fireworks/models/gemma-4-26b-a4b-it"
DEFAULT_FIREWORKS_GEMMA_AUDIO_VISUAL_MODEL = "accounts/fireworks/models/gemma-4-12b-unified-it"
DEFAULT_FIREWORKS_GEMMA_CAPTION_MODEL = "accounts/fireworks/models/gemma-4-31b-it"
DEFAULT_GOOGLE_GEMMA_VISUAL_MODEL = "gemma-4-31b-it"
DEFAULT_GOOGLE_GEMMA_AUDIO_VISUAL_MODEL = "gemma-4-12b-unified-it"
DEFAULT_GOOGLE_GEMMA_CAPTION_MODEL = "gemma-4-31b-it"
DEFAULT_AMD_GEMMA_AUDIO_VISUAL_MODEL = "gemma-4-12b-it"
DEFAULT_GEMMA_MAX_TOKENS = 2048
DEFAULT_TOP_K = 40
DEFAULT_GOOGLE_MAX_RETRIES = 2
DEFAULT_GOOGLE_RETRY_BACKOFF_SECONDS = (0.5, 1.0)
DEFAULT_OPENROUTER_MAX_RETRIES = 2
DEFAULT_OPENROUTER_RETRY_BACKOFF_SECONDS = (0.5, 1.0)
GOOGLE_IMAGE_MAX_SIDE = 1536
GOOGLE_IMAGE_QUALITY = 85
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


@dataclass(frozen=True, slots=True)
class FireworksJudgeConfig:
    api_key: str
    base_url: str
    vision_model: str
    fallback_vision_model: str
    max_tokens: int = DEFAULT_GEMMA_MAX_TOKENS
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 45.0
    write_timeout_seconds: float = 30.0
    pool_timeout_seconds: float = 10.0
    provider: str = DEFAULT_PROVIDER_FIREWORKS_JUDGE


@dataclass(frozen=True, slots=True)
class RoutedGemmaConfig:
    fireworks_api_key: str
    google_api_key: str
    fireworks_base_url: str
    fireworks_visual_model: str
    fireworks_audio_visual_model: str
    fireworks_caption_model: str
    google_visual_model: str
    google_audio_visual_model: str
    google_caption_model: str
    amd_audio_visual_api_key: str = ""
    amd_audio_visual_base_url: str = ""
    amd_audio_visual_model: str = DEFAULT_AMD_GEMMA_AUDIO_VISUAL_MODEL
    max_tokens: int = DEFAULT_GEMMA_MAX_TOKENS
    provider: str = DEFAULT_PROVIDER_ROUTED_GEMMA

    @property
    def has_credentials(self) -> bool:
        return bool(
            self.fireworks_api_key
            or self.google_api_key
            or (self.amd_audio_visual_api_key and self.amd_audio_visual_base_url)
        )

    @property
    def amd_audio_visual_configured(self) -> bool:
        return bool(self.amd_audio_visual_api_key and self.amd_audio_visual_base_url)

    def role_configs(self, role: str) -> tuple[GemmaModelConfig, ...]:
        fireworks_models = {
            "visual": self.fireworks_visual_model,
            "audio_visual": self.fireworks_audio_visual_model,
            "caption": self.fireworks_caption_model,
        }
        google_models = {
            "visual": self.google_visual_model,
            "audio_visual": self.google_audio_visual_model,
            "caption": self.google_caption_model,
        }
        if role not in fireworks_models:
            raise ValueError(f"Unknown routed Gemma model role: {role}")
        configs: list[GemmaModelConfig] = []
        if role == "audio_visual" and self.amd_audio_visual_configured:
            configs.append(GemmaModelConfig(
                api_key=self.amd_audio_visual_api_key, base_url=self.amd_audio_visual_base_url,
                model=self.amd_audio_visual_model, fallback_models=(), max_tokens=self.max_tokens,
                provider=DEFAULT_PROVIDER_AMD_CLOUD,
            ))
        elif self.fireworks_api_key:
            configs.append(GemmaModelConfig(
                api_key=self.fireworks_api_key, base_url=self.fireworks_base_url,
                model=fireworks_models[role], fallback_models=(), max_tokens=self.max_tokens,
                provider=DEFAULT_PROVIDER_FIREWORKS,
            ))
        if self.google_api_key:
            configs.append(GemmaModelConfig(
                api_key=self.google_api_key, base_url=None, model=google_models[role],
                fallback_models=(), max_tokens=self.max_tokens, provider=DEFAULT_PROVIDER_GOOGLE,
            ))
        return tuple(configs)


def load_gemma_config(
    env: Mapping[str, str] | None = None,
) -> GemmaConfig | FireworksJudgeConfig | RoutedGemmaConfig | None:
    values = env if env is not None else os.environ
    provider = _resolve_provider(values)

    if provider == DEFAULT_PROVIDER_GOOGLE:
        return load_google_provider_config(values)

    if provider == DEFAULT_PROVIDER_OPENROUTER:
        return load_openrouter_provider_config(values)

    if provider == DEFAULT_PROVIDER_FIREWORKS_JUDGE:
        return load_fireworks_judge_provider_config(values)

    if provider == DEFAULT_PROVIDER_ROUTED_GEMMA:
        return load_routed_gemma_config(values)

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


def load_google_provider_config(env: Mapping[str, str] | None = None) -> GemmaConfig | None:
    values = env if env is not None else os.environ
    api_key = values.get("GEMINI_API_KEY", "").strip() or values.get("GOOGLE_API_KEY", "").strip()
    model = values.get("GEMINI_MODEL", "").strip() or DEFAULT_GOOGLE_GEMMA_MODEL
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
        provider=DEFAULT_PROVIDER_GOOGLE,
    )


def load_openrouter_provider_config(env: Mapping[str, str] | None = None) -> GemmaConfig | None:
    values = env if env is not None else os.environ
    api_key = values.get("OPENROUTER_API_KEY", "").strip()
    model = values.get("OPENROUTER_MODEL", "").strip()
    if not api_key or not model:
        return None
    return GemmaConfig(
        api_key=api_key,
        base_url=DEFAULT_OPENROUTER_CHAT_COMPLETIONS_URL,
        vision_model=model,
        text_model=model,
        fallback_models=(),
        max_tokens=_parse_max_tokens(values.get("GEMMA_MAX_TOKENS")),
        use_response_format=False,
        provider=DEFAULT_PROVIDER_OPENROUTER,
    )


def load_fireworks_judge_provider_config(
    env: Mapping[str, str] | None = None,
) -> FireworksJudgeConfig | None:
    values = env if env is not None else os.environ
    api_key = values.get("FIREWORKS_API_KEY", "").strip()
    base_url = values.get("FIREWORKS_BASE_URL", "").strip() or DEFAULT_FIREWORKS_BASE_URL
    vision_model = values.get("FIREWORKS_VISION_MODEL", "").strip() or DEFAULT_FIREWORKS_JUDGE_VISION_MODEL
    fallback_vision_model = (
        values.get("FIREWORKS_FALLBACK_VISION_MODEL", "").strip()
        or DEFAULT_FIREWORKS_JUDGE_FALLBACK_VISION_MODEL
    )
    if not api_key or not base_url:
        return None
    return FireworksJudgeConfig(
        api_key=api_key,
        base_url=base_url,
        vision_model=vision_model,
        fallback_vision_model=fallback_vision_model,
        max_tokens=_parse_max_tokens(values.get("GEMMA_MAX_TOKENS")),
    )


def load_routed_gemma_config(env: Mapping[str, str] | None = None) -> RoutedGemmaConfig:
    values = env if env is not None else os.environ
    return RoutedGemmaConfig(
        fireworks_api_key=values.get("FIREWORKS_API_KEY", "").strip() or values.get("GEMMA_API_KEY", "").strip(),
        google_api_key=values.get("GEMINI_API_KEY", "").strip() or values.get("GOOGLE_API_KEY", "").strip(),
        fireworks_base_url=values.get("FIREWORKS_BASE_URL", "").strip() or DEFAULT_FIREWORKS_BASE_URL,
        fireworks_visual_model=values.get("FIREWORKS_GEMMA_VISUAL_MODEL", "").strip() or DEFAULT_FIREWORKS_GEMMA_VISUAL_MODEL,
        fireworks_audio_visual_model=values.get("FIREWORKS_GEMMA_AUDIO_VISUAL_MODEL", "").strip() or DEFAULT_FIREWORKS_GEMMA_AUDIO_VISUAL_MODEL,
        fireworks_caption_model=values.get("FIREWORKS_GEMMA_CAPTION_MODEL", "").strip() or DEFAULT_FIREWORKS_GEMMA_CAPTION_MODEL,
        google_visual_model=values.get("GOOGLE_GEMMA_VISUAL_MODEL", "").strip() or DEFAULT_GOOGLE_GEMMA_VISUAL_MODEL,
        google_audio_visual_model=values.get("GOOGLE_GEMMA_AUDIO_VISUAL_MODEL", "").strip() or DEFAULT_GOOGLE_GEMMA_AUDIO_VISUAL_MODEL,
        google_caption_model=values.get("GOOGLE_GEMMA_CAPTION_MODEL", "").strip() or DEFAULT_GOOGLE_GEMMA_CAPTION_MODEL,
        amd_audio_visual_api_key=values.get("AMD_GEMMA_AUDIO_VISUAL_API_KEY", "").strip(),
        amd_audio_visual_base_url=values.get("AMD_GEMMA_AUDIO_VISUAL_BASE_URL", "").strip(),
        amd_audio_visual_model=values.get("AMD_GEMMA_AUDIO_VISUAL_MODEL", "").strip() or DEFAULT_AMD_GEMMA_AUDIO_VISUAL_MODEL,
        max_tokens=_parse_max_tokens(values.get("GEMMA_MAX_TOKENS")),
    )


def create_model_client(
    config: GemmaModelConfig | FireworksJudgeConfig,
) -> GemmaClient | GoogleGeminiClient | OpenRouterClient | FireworksVisionClient:
    if isinstance(config, FireworksJudgeConfig):
        return FireworksVisionClient(config)
    if config.provider == DEFAULT_PROVIDER_GOOGLE:
        return GoogleGeminiClient(config)
    if config.provider == DEFAULT_PROVIDER_OPENROUTER:
        return OpenRouterClient(config)
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
        max_tokens: int | None = None,
    ) -> str:
        payload = self._post_chat_completion(
            messages=messages,
            temperature=temperature,
            use_response_format=use_response_format,
            max_tokens=max_tokens,
        )
        return extract_message_text(payload)

    def chat_completion_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        *,
        use_response_format: bool | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        content = self.chat_completion_text(
            messages=messages,
            temperature=temperature,
            use_response_format=use_response_format,
            max_tokens=max_tokens,
        )
        return parse_json_object(content)

    def _post_chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        *,
        use_response_format: bool | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if not self._config.base_url:
            raise RuntimeError("OpenAI-compatible Gemma client requires a base URL.")

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
                max_tokens=max_tokens,
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


class FireworksRuntimeBudgetError(RuntimeError):
    """Raised before an API attempt when the batch deadline no longer permits it."""


class FireworksVisionRequestError(RuntimeError):
    """A failed Fireworks operation with a retained retryability classification."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class FireworksVisionClient:
    """OpenAI-compatible multimodal client with explicit model fallback policy."""

    def __init__(
        self,
        config: FireworksJudgeConfig,
        client: httpx.Client | None = None,
        *,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(
                connect=config.connect_timeout_seconds,
                read=config.read_timeout_seconds,
                write=config.write_timeout_seconds,
                pool=config.pool_timeout_seconds,
            )
        )
        self._sleeper = sleeper
        self._clock = clock

    def complete_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float,
        validator: Callable[[dict[str, Any]], Any],
        remaining_time_fn: Callable[[], float] | None = None,
        minimum_remaining_seconds: float = 0.0,
        operation: str = "generation",
        validation_failure_handler: Callable[[str, int, str], None] | None = None,
        response_handler: Callable[[str, int, str], None] | None = None,
        model_attempts: Sequence[tuple[str, int]] | None = None,
    ) -> Any:
        failures: list[str] = []
        last_error: Exception | None = None
        had_retryable_failure = False
        request_attempt = 0
        configured_attempts = model_attempts or (
            (self._config.vision_model, 2),
            (self._config.fallback_vision_model, 1),
        )
        for model_index, (model, attempts) in enumerate(configured_attempts):
            for attempt in range(1, attempts + 1):
                response_text = ""
                try:
                    self._ensure_attempt_budget(remaining_time_fn, minimum_remaining_seconds)
                    request_attempt += 1
                    response_text = self._request_text(messages, model=model, attempt=attempt, temperature=temperature)
                    if response_handler is not None:
                        response_handler(model, request_attempt, response_text)
                    try:
                        response_object = parse_json_object(response_text)
                    except ValueError as exc:
                        raise ValueError("no JSON object found") from exc
                    result = validator(response_object)
                    return result
                except FireworksRuntimeBudgetError:
                    raise
                except Exception as exc:
                    retryable = _is_retryable_fireworks_error(exc)
                    last_error = exc
                    had_retryable_failure = had_retryable_failure or retryable
                    failures.append(f"{model}: {exc.__class__.__name__}")
                    if response_text and isinstance(exc, ValueError) and validation_failure_handler is not None:
                        validation_failure_handler(model, request_attempt, response_text)
                    LOGGER.warning(
                        "Fireworks vision operation=%s provider=%s model=%s attempt=%s failed retryable=%s "
                        "exception=%s message=%s",
                        operation,
                        self._config.provider,
                        model,
                        attempt,
                        retryable,
                        exc.__class__.__name__,
                        _sanitize_fireworks_diagnostic(str(exc), self._config.api_key),
                    )
                    if attempt < attempts and retryable:
                        self._sleeper(0.5)
                        continue
                    break

            if model_index == 0 and len(configured_attempts) > 1:
                LOGGER.warning(
                    "Fireworks vision primary model %s failed; switching to configured fallback model %s.",
                    self._config.vision_model,
                    self._config.fallback_vision_model,
                )

        error = FireworksVisionRequestError(
            f"Fireworks vision request failed for configured models: {', '.join(failures)}",
            retryable=had_retryable_failure,
        )
        if last_error is not None:
            raise error from last_error
        raise error

    @staticmethod
    def _ensure_attempt_budget(
        remaining_time_fn: Callable[[], float] | None,
        minimum_remaining_seconds: float,
    ) -> None:
        if remaining_time_fn is None:
            return
        remaining_seconds = max(0.0, remaining_time_fn())
        if remaining_seconds < minimum_remaining_seconds:
            raise FireworksRuntimeBudgetError(
                f"Only {remaining_seconds:.1f}s remain; need {minimum_remaining_seconds:.1f}s before a Fireworks request."
            )

    def _request_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        attempt: int,
        temperature: float,
    ) -> str:
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": self._config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        started_at = self._clock()
        status_code: int | str = "timeout"
        try:
            response = self._client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self._config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            status_code = response.status_code
            response.raise_for_status()
            response_payload = response.json()
            if not isinstance(response_payload, dict):
                raise ValueError("Fireworks vision returned an unexpected response shape.")
            try:
                return extract_message_text(response_payload)
            except ValueError as exc:
                if "empty" in str(exc).lower():
                    raise ValueError("response content empty") from exc
                raise
        finally:
            LOGGER.info(
                "Fireworks vision request provider=%s model=%s attempt=%s status_code=%s elapsed_seconds=%.3f",
                self._config.provider,
                model,
                attempt,
                status_code,
                self._clock() - started_at,
            )


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
        max_tokens: int | None = None,
    ) -> str:
        types = _load_google_types()
        contents, system_instruction, uploaded_file_names = _convert_messages_to_google_contents(
            messages,
            types,
            self._client,
        )
        request_config = _build_google_generation_config(
            system_instruction=system_instruction,
            temperature=temperature,
            max_tokens=max_tokens or self._config.max_tokens,
            use_json=_google_use_json_response(use_response_format),
        )

        last_error: Exception | None = None
        max_attempts = DEFAULT_GOOGLE_MAX_RETRIES + 1
        try:
            for attempt in range(max_attempts):
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
                    if status_code is None or not _is_retryable_google_status(status_code) or attempt == max_attempts - 1:
                        raise RuntimeError(f"Google Gemini request failed for model {self._config.model}: {exc}") from exc
                    delay_seconds = DEFAULT_GOOGLE_RETRY_BACKOFF_SECONDS[
                        min(attempt, len(DEFAULT_GOOGLE_RETRY_BACKOFF_SECONDS) - 1)
                    ]
                    LOGGER.warning(
                        "Google Gemini request for model %s failed with status %s; retrying in %.1fs.",
                        self._config.model,
                        status_code,
                        delay_seconds,
                    )
                    self._sleeper(delay_seconds)
        finally:
            _delete_google_uploaded_files(self._client, uploaded_file_names)

        raise RuntimeError(f"Google Gemini request failed for model {self._config.model}: {last_error}")

    def chat_completion_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        *,
        use_response_format: bool | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        text = self.chat_completion_text(
            messages,
            temperature,
            use_response_format=use_response_format,
            max_tokens=max_tokens,
        )
        return parse_json_object(text)


class OpenRouterClient:
    def __init__(
        self,
        config: GemmaModelConfig,
        client: httpx.Client | None = None,
        *,
        sleeper: Any = time.sleep,
    ) -> None:
        self._config = config
        self._client = client or httpx.Client(timeout=config.timeout_seconds)
        self._sleeper = sleeper

    def chat_completion_text(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        *,
        use_response_format: bool | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self._config.base_url:
            raise RuntimeError("OpenRouter client requires a base URL.")

        request_payload = build_openrouter_chat_completion_payload(
            self._config,
            messages=_convert_messages_to_openrouter_messages(messages),
            temperature=temperature,
            use_response_format=use_response_format,
            max_tokens=max_tokens,
        )
        max_attempts = DEFAULT_OPENROUTER_MAX_RETRIES + 1
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                response = self._client.post(
                    self._config.base_url,
                    headers={
                        "Authorization": f"Bearer {self._config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                )
                LOGGER.info(
                    "OpenRouter request provider=%s model=%s status_code=%s",
                    self._config.provider,
                    self._config.model,
                    response.status_code,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                LOGGER.warning(
                    "OpenRouter request provider=%s model=%s status_code=%s",
                    self._config.provider,
                    self._config.model,
                    status_code,
                )
                if not _is_retryable_openrouter_status(status_code) or attempt == max_attempts - 1:
                    raise RuntimeError(f"OpenRouter request failed for model {self._config.model}: {exc}") from exc
                self._sleeper(
                    DEFAULT_OPENROUTER_RETRY_BACKOFF_SECONDS[
                        min(attempt, len(DEFAULT_OPENROUTER_RETRY_BACKOFF_SECONDS) - 1)
                    ]
                )
                continue
            except httpx.HTTPError as exc:
                LOGGER.warning(
                    "OpenRouter request provider=%s model=%s transport_error=%s",
                    self._config.provider,
                    self._config.model,
                    exc.__class__.__name__,
                )
                raise RuntimeError(f"OpenRouter request failed for model {self._config.model}: {exc}") from exc

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                raise RuntimeError("OpenRouter returned invalid JSON.") from exc

            if not isinstance(payload, dict):
                raise RuntimeError("OpenRouter returned an unexpected response shape.")
            return extract_message_text(payload)

        raise RuntimeError(f"OpenRouter request failed for model {self._config.model}: {last_error}")

    def chat_completion_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        temperature: float,
        *,
        use_response_format: bool | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        text = self.chat_completion_text(
            messages,
            temperature,
            use_response_format=use_response_format,
            max_tokens=max_tokens,
        )
        return parse_json_object(text)


def build_chat_completion_payload(
    config: GemmaModelConfig,
    *,
    messages: Sequence[Mapping[str, Any]],
    temperature: float,
    model: str | None = None,
    use_response_format: bool | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model or config.model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens or config.max_tokens,
        "top_k": DEFAULT_TOP_K,
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }
    if use_response_format if use_response_format is not None else config.use_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def build_openrouter_chat_completion_payload(
    config: GemmaModelConfig,
    *,
    messages: Sequence[Mapping[str, Any]],
    temperature: float,
    use_response_format: bool | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": list(messages),
        "temperature": temperature,
        "max_tokens": max_tokens or config.max_tokens,
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


def _convert_messages_to_google_contents(
    messages: Sequence[Mapping[str, Any]],
    types,
    google_client: Any,
) -> tuple[list[Any], str | None, list[str]]:
    system_fragments: list[str] = []
    contents: list[Any] = []
    uploaded_file_names: list[str] = []

    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")
        if role == "system":
            system_text = _flatten_message_text(content)
            if system_text:
                system_fragments.append(system_text)
            continue

        content_parts, new_uploaded_file_names = _convert_message_content_to_google_parts(content, types, google_client)
        contents.extend(content_parts)
        uploaded_file_names.extend(new_uploaded_file_names)

    if not contents:
        raise ValueError("Google Gemini request did not include any user content.")

    system_instruction = "\n\n".join(fragment for fragment in system_fragments if fragment).strip() or None
    return contents, system_instruction, uploaded_file_names


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


def _convert_messages_to_openrouter_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower() or "user"
        content = message.get("content")
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            converted.append({"role": role, "content": ""})
            continue

        content_parts: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    content_parts.append({"type": "text", "text": text.strip()})
                continue
            if item_type == "image_file":
                path = item.get("path")
                if isinstance(path, str) and path.strip():
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": _make_image_file_data_url(Path(path.strip()))},
                        }
                    )
                continue
            if item_type == "image_url":
                image_url = item.get("image_url")
                if isinstance(image_url, Mapping):
                    url = image_url.get("url")
                    if isinstance(url, str) and url.strip():
                        content_parts.append({"type": "image_url", "image_url": {"url": url.strip()}})
                continue
            mime_type = item.get("mime_type")
            data = item.get("data")
            if isinstance(mime_type, str) and isinstance(data, bytes):
                content_parts.append({"type": "image_url", "image_url": {"url": _encode_data_url(mime_type, data)}})
                continue
            if isinstance(mime_type, str) and isinstance(data, str):
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _encode_data_url(mime_type, base64.b64decode(data))},
                    }
                )
        converted.append({"role": role, "content": content_parts})
    return converted


def _convert_message_content_to_google_parts(
    content: Any,
    types,
    google_client: Any,
) -> tuple[list[Any], list[str]]:
    if isinstance(content, str):
        return ([content.strip()] if content.strip() else []), []
    if not isinstance(content, list):
        return [], []

    parts: list[Any] = []
    uploaded_file_names: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
            continue
        if item_type == "image_file":
            path = item.get("path")
            if not isinstance(path, str) or not path.strip():
                continue
            uploaded_file = _upload_google_image_path(google_client, Path(path.strip()))
            parts.append(uploaded_file)
            uploaded_file_name = getattr(uploaded_file, "name", None)
            if isinstance(uploaded_file_name, str) and uploaded_file_name.strip():
                uploaded_file_names.append(uploaded_file_name.strip())
            continue
        if item_type == "audio_file":
            path = item.get("path")
            if not isinstance(path, str) or not path.strip():
                continue
            uploaded_file = google_client.files.upload(file=path.strip(), config={"mime_type": "audio/wav"})
            parts.append(uploaded_file)
            uploaded_file_name = getattr(uploaded_file, "name", None)
            if isinstance(uploaded_file_name, str) and uploaded_file_name.strip():
                uploaded_file_names.append(uploaded_file_name.strip())
            continue
        if item_type == "image_url":
            image_url = item.get("image_url")
            if not isinstance(image_url, Mapping):
                continue
            url = image_url.get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            mime_type, data = _decode_image_payload(url.strip())
            uploaded_file = _upload_google_image_bytes(google_client, data=data, mime_type=mime_type)
            parts.append(uploaded_file)
            uploaded_file_name = getattr(uploaded_file, "name", None)
            if isinstance(uploaded_file_name, str) and uploaded_file_name.strip():
                uploaded_file_names.append(uploaded_file_name.strip())
            continue
        mime_type = item.get("mime_type")
        data = item.get("data")
        if isinstance(mime_type, str) and isinstance(data, bytes):
            uploaded_file = _upload_google_image_bytes(google_client, data=data, mime_type=mime_type)
            parts.append(uploaded_file)
            uploaded_file_name = getattr(uploaded_file, "name", None)
            if isinstance(uploaded_file_name, str) and uploaded_file_name.strip():
                uploaded_file_names.append(uploaded_file_name.strip())
            continue
        if isinstance(mime_type, str) and isinstance(data, str):
            uploaded_file = _upload_google_image_bytes(google_client, data=base64.b64decode(data), mime_type=mime_type)
            parts.append(uploaded_file)
            uploaded_file_name = getattr(uploaded_file, "name", None)
            if isinstance(uploaded_file_name, str) and uploaded_file_name.strip():
                uploaded_file_names.append(uploaded_file_name.strip())
    return parts, uploaded_file_names


def _decode_image_payload(value: str) -> tuple[str, bytes]:
    if not value.startswith("data:"):
        raise ValueError("Google Gemini image input must be a data URL.")
    header, encoded = value.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Google Gemini image data URL must be base64 encoded.")
    mime_type = header[5:].split(";", 1)[0].strip() or "image/jpeg"
    return mime_type, base64.b64decode(encoded)


def _make_image_file_data_url(image_path: Path) -> str:
    mime_type = "image/jpeg"
    return _encode_data_url(mime_type, image_path.read_bytes())


def _encode_data_url(mime_type: str, data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_google_generation_config(
    *,
    system_instruction: str | None,
    temperature: float,
    max_tokens: int,
    use_json: bool,
) -> Any:
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
    return _with_google_thinking_config(config)


def _google_use_json_response(use_response_format: bool | None) -> bool:
    return use_response_format is True


def _with_google_thinking_config(config: dict[str, Any]) -> Any:
    try:
        types = _load_google_types()
        generate_config_type = getattr(types, "GenerateContentConfig")
        thinking_config_type = getattr(types, "ThinkingConfig")
        thinking_config = thinking_config_type(thinking_level="minimal")
        return generate_config_type(**config, thinking_config=thinking_config)
    except Exception:  # pragma: no cover - depends on SDK surface/version
        return config


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


def _is_retryable_openrouter_status(status_code: int) -> bool:
    return status_code in {429, 500, 502, 503, 504}


def _is_retryable_fireworks_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    # The provider contract requires one retry for malformed JSON and invalid caption structure.
    return isinstance(exc, ValueError)


def _sanitize_fireworks_diagnostic(message: str, api_key: str) -> str:
    sanitized = message.replace(api_key, "[redacted]") if api_key else message
    sanitized = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(
        r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+",
        "[redacted image data]",
        sanitized,
        flags=re.IGNORECASE,
    )
    return " ".join(sanitized.split())[:300]


def _upload_google_image_path(google_client: Any, image_path: Path) -> Any:
    resized_bytes = _make_google_resized_jpeg_bytes(image_path)
    return _upload_google_image_bytes(google_client, data=resized_bytes, mime_type="image/jpeg")


def _upload_google_image_bytes(
    google_client: Any,
    *,
    data: bytes,
    mime_type: str,
) -> Any:
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as handle:
            handle.write(data)
            temp_path = handle.name
        return google_client.files.upload(file=temp_path, config={"mime_type": mime_type})
    finally:
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _make_google_resized_jpeg_bytes(image_path: Path) -> bytes:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on install state
        raise RuntimeError("Pillow is required for Google Gemini image uploads.") from exc

    output = BytesIO()
    with Image.open(image_path) as image:
        converted = image.convert("RGB")
        converted.thumbnail((GOOGLE_IMAGE_MAX_SIDE, GOOGLE_IMAGE_MAX_SIDE))
        converted.save(output, format="JPEG", quality=GOOGLE_IMAGE_QUALITY, optimize=True)
    return output.getvalue()


def _delete_google_uploaded_files(google_client: Any, uploaded_file_names: Sequence[str]) -> None:
    for file_name in uploaded_file_names:
        try:
            google_client.files.delete(name=file_name)
        except Exception:  # pragma: no cover - cleanup should not affect captioning
            LOGGER.debug("Failed to delete uploaded Google Gemini file %s.", file_name)


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
