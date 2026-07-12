from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from typing import Any


LOGGER = logging.getLogger("gemmaclip.web.events")
_SAFE_FIELDS = {
    "run_id",
    "mode",
    "stage",
    "duration_seconds",
    "provider",
    "model",
    "modality",
    "fallback_used",
    "generation_outcome",
    "degraded",
    "error_category",
    "status",
    "artifact_count",
    "remaining_seconds",
}
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_.:/-]+")
_RUN_ID = re.compile(r"^run_[A-Za-z0-9_-]{20,80}$")
_secret_values: tuple[str, ...] = ()
_log_format = "text"


def configure_event_logging(env: Mapping[str, str] | None = None) -> None:
    """Remember runtime secret values so safe lifecycle logs can redact them."""

    values = env if env is not None else os.environ
    candidates = [
        value.strip()
        for key, value in values.items()
        if any(fragment in key.upper() for fragment in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
        and value.strip()
    ]
    global _secret_values
    _secret_values = tuple(sorted(set(candidates), key=len, reverse=True))
    global _log_format
    _log_format = "json" if values.get("GEMMACLIP_LOG_FORMAT", "").strip().lower() == "json" else "text"


def log_event(event: str, *, secrets: Mapping[str, str] | None = None, **fields: Any) -> None:
    """Emit only an allow-listed, secret-safe lifecycle event.

    Callers intentionally pass metadata, never captions, evidence, prompts, or provider
    responses. Unknown fields are dropped before formatting.
    """

    secret_values = (*_secret_values, *(_secret_values_from(secrets) if secrets else ()))
    payload: dict[str, Any] = {"event": _safe_token(event, fallback="unknown_event")}
    for key in sorted(_SAFE_FIELDS):
        if key not in fields:
            continue
        payload[key] = _safe_field(key, fields[key], secret_values)

    if _log_format == "json" or os.environ.get("GEMMACLIP_LOG_FORMAT", "").strip().lower() == "json":
        LOGGER.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return
    rendered = " ".join(f"{key}={_render_text(value)}" for key, value in payload.items())
    LOGGER.info(rendered)


def safe_model_label(model: Any) -> str:
    """Convert provider-specific model IDs into a public log label."""

    value = str(model or "").lower()
    if "12b" in value and "unified" in value:
        return "Gemma 4 12B Unified"
    if "26b" in value and "a4b" in value:
        return "Gemma 4 26B A4B"
    if "31b" in value:
        return "Gemma 4 31B"
    if value:
        return "configured Gemma model"
    return "unknown"


def _safe_field(key: str, value: Any, secret_values: tuple[str, ...]) -> Any:
    if key in {"duration_seconds", "remaining_seconds"}:
        try:
            return round(max(0.0, float(value)), 3)
        except (TypeError, ValueError):
            return 0.0
    if key in {"fallback_used", "degraded"}:
        return bool(value)
    if key == "artifact_count":
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
    if key == "run_id":
        candidate = str(value)
        return candidate if _RUN_ID.fullmatch(candidate) else "redacted"
    if key == "model":
        return safe_model_label(value)
    if key == "error_category":
        return _redact_text(_safe_token(str(value), fallback="unknown"), secret_values)[:80]
    return _redact_text(_safe_token(str(value), fallback="unknown"), secret_values)[:160]


def _safe_token(value: str, *, fallback: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        return fallback
    return _SAFE_TOKEN.sub("_", cleaned)


def _redact_text(value: str, secret_values: tuple[str, ...]) -> str:
    redacted = value
    for secret in secret_values:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    redacted = re.sub(r"(?i)bearer[_:-]?[^\s]+", "bearer_[redacted]", redacted)
    redacted = re.sub(r"(?i)https?://[^\s]+", "[redacted_url]", redacted)
    return redacted


def _secret_values_from(values: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value.strip()
                for key, value in values.items()
                if any(fragment in key.upper() for fragment in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))
                and value.strip()
            },
            key=len,
            reverse=True,
        )
    )


def _render_text(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)
