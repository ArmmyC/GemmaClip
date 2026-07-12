from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import httpx

from gemmaclip.leaderboard.config import FireworksLeaderboardConfig
from gemmaclip.leaderboard.validation import (
    CaptionValidationError,
    extract_valid_partial_captions,
    parse_json_object,
)

LOGGER = logging.getLogger("gemmaclip.leaderboard.fireworks")


class FireworksLeaderboardRuntimeError(RuntimeError):
    """Raised before a request when the live competition deadline is unsafe."""


class FireworksLeaderboardRequestError(RuntimeError):
    """Safe, classified request failure; raw provider responses are never logged."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool | None = None,
        fallback_eligible: bool | None = None,
        category: str,
        status_code: int | None = None,
        partial_captions: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        # Keep retryable for compatibility, but use fallback_eligible for
        # model selection. A 404 is not a same-model retry, yet it can safely
        # try the explicitly configured fallback model once.
        self.retryable = bool(retryable) if retryable is not None else bool(fallback_eligible)
        self.fallback_eligible = bool(fallback_eligible) if fallback_eligible is not None else self.retryable
        self.category = category
        self.status_code = status_code
        self.partial_captions = partial_captions or {}


class FireworksLeaderboardClient:
    """Minimal OpenAI-compatible client with one attempt per configured model."""

    def __init__(
        self,
        config: FireworksLeaderboardConfig,
        client: httpx.Client | Any | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(
                connect=10.0,
                read=45.0,
                write=30.0,
                pool=10.0,
            )
        )
        self._clock = clock

    def complete_json(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str,
        temperature: float,
        validator: Callable[[dict[str, Any]], Any],
        remaining_time_fn: Callable[[], float] | None = None,
        minimum_remaining_seconds: float = 65.0,
        operation: str = "generation",
        styles: Sequence[str] = (),
        debug_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> Any:
        self._ensure_attempt_budget(remaining_time_fn, minimum_remaining_seconds)
        started_at = self._clock()
        status_code: int | None = None
        category = "unknown"
        try:
            response = self._client.post(
                f"{self.config.base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": list(messages),
                    "temperature": temperature,
                    "max_tokens": self.config.max_tokens,
                    "response_format": {"type": "json_object"},
                },
            )
            status_code = getattr(response, "status_code", None)
            if status_code is not None and status_code >= 400:
                retryable = _same_model_retryable_status(status_code)
                fallback_eligible = _fallback_eligible_status(status_code)
                category = _status_category(status_code)
                raise FireworksLeaderboardRequestError(
                    "Fireworks request returned an HTTP error",
                    retryable=retryable,
                    fallback_eligible=fallback_eligible,
                    category=category,
                    status_code=status_code,
                )
            payload = response.json()
            response_text = _extract_response_text(payload)
            parsed = parse_json_object(response_text)
            try:
                result = validator(parsed)
            except Exception as exc:
                partial = extract_valid_partial_captions(parsed, styles) if styles else {}
                category = getattr(exc, "category", "validation")
                raise FireworksLeaderboardRequestError(
                    "Fireworks response failed local validation",
                    retryable=True,
                    fallback_eligible=True,
                    category=category,
                    status_code=status_code,
                    partial_captions=partial,
                ) from exc
            _emit_debug(
                debug_callback,
                operation=operation,
                model=model,
                status="success",
                status_code=status_code,
                category="ok",
                elapsed_seconds=self._clock() - started_at,
                result=result,
            )
            LOGGER.info(
                "Fireworks leaderboard operation=%s model=%s status=success status_code=%s elapsed_seconds=%.3f",
                operation,
                model,
                status_code,
                self._clock() - started_at,
            )
            return result
        except FireworksLeaderboardRuntimeError:
            raise
        except FireworksLeaderboardRequestError as exc:
            _emit_debug(
                debug_callback,
                operation=operation,
                model=model,
                status="failed",
                status_code=exc.status_code,
                category=exc.category,
                elapsed_seconds=self._clock() - started_at,
                result=None,
            )
            LOGGER.warning(
                "Fireworks leaderboard operation=%s model=%s status=failed status_code=%s retryable=%s fallback_eligible=%s category=%s elapsed_seconds=%.3f",
                operation,
                model,
                exc.status_code,
                exc.retryable,
                exc.fallback_eligible,
                exc.category,
                self._clock() - started_at,
            )
            raise
        except httpx.TimeoutException as exc:
            _emit_debug(
                debug_callback,
                operation=operation,
                model=model,
                status="failed",
                status_code=status_code,
                category="timeout",
                elapsed_seconds=self._clock() - started_at,
                result=None,
            )
            LOGGER.warning(
                "Fireworks leaderboard operation=%s model=%s status=failed category=timeout elapsed_seconds=%.3f",
                operation,
                model,
                self._clock() - started_at,
            )
            raise FireworksLeaderboardRequestError(
                "Fireworks request timed out",
                retryable=True,
                fallback_eligible=True,
                category="timeout",
                status_code=status_code,
            ) from exc
        except httpx.RequestError as exc:
            _emit_debug(
                debug_callback,
                operation=operation,
                model=model,
                status="failed",
                status_code=status_code,
                category="network",
                elapsed_seconds=self._clock() - started_at,
                result=None,
            )
            LOGGER.warning(
                "Fireworks leaderboard operation=%s model=%s status=failed category=network elapsed_seconds=%.3f",
                operation,
                model,
                self._clock() - started_at,
            )
            raise FireworksLeaderboardRequestError(
                "Fireworks request failed due to a network error",
                retryable=True,
                fallback_eligible=True,
                category="network",
                status_code=status_code,
            ) from exc
        except (CaptionValidationError, ValueError, TypeError, KeyError) as exc:
            category = getattr(exc, "category", "invalid_response")
            _emit_debug(
                debug_callback,
                operation=operation,
                model=model,
                status="failed",
                status_code=status_code,
                category=category,
                elapsed_seconds=self._clock() - started_at,
                result=None,
            )
            LOGGER.warning(
                "Fireworks leaderboard operation=%s model=%s status=failed category=%s elapsed_seconds=%.3f",
                operation,
                model,
                category,
                self._clock() - started_at,
            )
            raise FireworksLeaderboardRequestError(
                "Fireworks response was empty or invalid",
                retryable=True,
                fallback_eligible=True,
                category=category,
                status_code=status_code,
            ) from exc

    @staticmethod
    def _ensure_attempt_budget(
        remaining_time_fn: Callable[[], float] | None,
        minimum_remaining_seconds: float,
    ) -> None:
        if remaining_time_fn is None:
            return
        remaining = max(0.0, float(remaining_time_fn()))
        if remaining < minimum_remaining_seconds:
            raise FireworksLeaderboardRuntimeError(
                f"Only {remaining:.1f}s remain; need {minimum_remaining_seconds:.1f}s before a Fireworks request."
            )


def _extract_response_text(payload: object) -> str:
    if not isinstance(payload, Mapping):
        raise CaptionValidationError("provider response is not an object", category="response_shape")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise CaptionValidationError("provider response has no choices", category="empty_response")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise CaptionValidationError("provider choice is invalid", category="response_shape")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise CaptionValidationError("provider message is invalid", category="response_shape")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise CaptionValidationError("provider response content is empty", category="empty_response")
    return content


def _same_model_retryable_status(status_code: int) -> bool:
    return status_code in {408, 409, 429} or 500 <= status_code <= 599


def _fallback_eligible_status(status_code: int) -> bool:
    if status_code in {401, 403}:
        return False
    if status_code == 404:
        return True
    return _same_model_retryable_status(status_code)


def _status_category(status_code: int) -> str:
    if status_code in {401, 403}:
        return "authentication_or_permission"
    if status_code == 404:
        return "not_found"
    if status_code == 429:
        return "rate_limited"
    if status_code in {408, 409}:
        return "retryable_client_error"
    if 500 <= status_code <= 599:
        return "provider_server_error"
    return "non_retryable_http_error"


def _emit_debug(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    operation: str,
    model: str,
    status: str,
    status_code: int | None,
    category: str,
    elapsed_seconds: float,
    result: Any,
) -> None:
    if callback is None:
        return
    word_counts: dict[str, int] = {}
    if isinstance(result, Mapping):
        captions = result.get("captions") if isinstance(result.get("captions"), Mapping) else result
        if isinstance(captions, Mapping):
            from gemmaclip.leaderboard.validation import caption_word_count

            word_counts = {
                str(style): caption_word_count(value)
                for style, value in captions.items()
                if isinstance(value, str)
            }
    callback(
        {
            "operation": operation,
            "model": model,
            "status": status,
            "status_code": status_code,
            "category": category,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "word_counts": word_counts,
        }
    )
