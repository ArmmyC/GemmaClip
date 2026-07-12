from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

DEFAULT_FIREWORKS_LEADERBOARD_PROVIDER = "fireworks_leaderboard"
DEFAULT_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_MODEL = "accounts/fireworks/models/minimax-m3"
DEFAULT_FIREWORKS_LEADERBOARD_FALLBACK_MODEL = "accounts/fireworks/models/minimax-m3"
DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_FALLBACK_MODEL = DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL
DEFAULT_GENERATION_TEMPERATURE = 0.35
DEFAULT_REPAIR_TEMPERATURE = 0.20
DEFAULT_REVIEW_TEMPERATURE = 0.0
DEFAULT_MIN_GENERATION_REMAINING_SECONDS = 65.0
DEFAULT_MIN_REVIEW_REMAINING_SECONDS = 150.0
DEFAULT_MAX_TOKENS = 2048


@dataclass(frozen=True, slots=True)
class FireworksLeaderboardConfig:
    """Secret-safe runtime configuration for the competition-only provider."""

    api_key: str = field(repr=False)
    base_url: str = field(repr=False)
    generation_model: str = DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL
    review_model: str = DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_MODEL
    fallback_model: str = DEFAULT_FIREWORKS_LEADERBOARD_FALLBACK_MODEL
    review_fallback_model: str = DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_FALLBACK_MODEL
    enable_review: bool = True
    generation_temperature: float = DEFAULT_GENERATION_TEMPERATURE
    repair_temperature: float = DEFAULT_REPAIR_TEMPERATURE
    review_temperature: float = DEFAULT_REVIEW_TEMPERATURE
    min_generation_remaining_seconds: float = DEFAULT_MIN_GENERATION_REMAINING_SECONDS
    min_review_remaining_seconds: float = DEFAULT_MIN_REVIEW_REMAINING_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    provider: str = DEFAULT_FIREWORKS_LEADERBOARD_PROVIDER

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def __repr__(self) -> str:
        return (
            f"FireworksLeaderboardConfig(provider={self.provider!r}, configured={self.is_configured!r}, "
            f"generation_model={self.generation_model!r}, review_model={self.review_model!r})"
        )

    @property
    def model_order(self) -> tuple[str, str]:
        return self.generation_model, self.fallback_model

    @property
    def review_model_order(self) -> tuple[str, str]:
        return self.review_model, self.review_fallback_model


def load_fireworks_leaderboard_config(
    env: Mapping[str, str] | None = None,
) -> FireworksLeaderboardConfig | None:
    """Load only the CLI provider's environment when it is explicitly selected.

    An explicitly selected provider returns an unconfigured object when the
    API key is absent.  This lets the CLI still choose the six-frame path and
    safely produce deterministic output without making a remote request.
    """

    values = env if env is not None else os.environ
    provider = str(values.get("GEMMACLIP_PROVIDER", "")).strip().lower()
    if provider != DEFAULT_FIREWORKS_LEADERBOARD_PROVIDER:
        return None

    return FireworksLeaderboardConfig(
        api_key=str(values.get("FIREWORKS_API_KEY", "")).strip(),
        base_url=str(values.get("FIREWORKS_BASE_URL", "")).strip() or DEFAULT_FIREWORKS_BASE_URL,
        generation_model=_model_value(values, "FIREWORKS_LEADERBOARD_GENERATION_MODEL", DEFAULT_FIREWORKS_LEADERBOARD_GENERATION_MODEL),
        review_model=_model_value(values, "FIREWORKS_LEADERBOARD_REVIEW_MODEL", DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_MODEL),
        fallback_model=_model_value(values, "FIREWORKS_LEADERBOARD_FALLBACK_MODEL", DEFAULT_FIREWORKS_LEADERBOARD_FALLBACK_MODEL),
        review_fallback_model=_model_value(
            values,
            "FIREWORKS_LEADERBOARD_REVIEW_FALLBACK_MODEL",
            DEFAULT_FIREWORKS_LEADERBOARD_REVIEW_FALLBACK_MODEL,
        ),
        enable_review=_parse_bool(values.get("FIREWORKS_LEADERBOARD_ENABLE_REVIEW"), default=True),
        generation_temperature=_parse_temperature(
            values.get("FIREWORKS_LEADERBOARD_GENERATION_TEMPERATURE"), DEFAULT_GENERATION_TEMPERATURE
        ),
        repair_temperature=_parse_temperature(
            values.get("FIREWORKS_LEADERBOARD_REPAIR_TEMPERATURE"), DEFAULT_REPAIR_TEMPERATURE
        ),
        review_temperature=_parse_temperature(
            values.get("FIREWORKS_LEADERBOARD_REVIEW_TEMPERATURE"), DEFAULT_REVIEW_TEMPERATURE
        ),
        min_generation_remaining_seconds=_parse_positive_float(
            values.get("FIREWORKS_LEADERBOARD_MIN_GENERATION_REMAINING_SECONDS"),
            DEFAULT_MIN_GENERATION_REMAINING_SECONDS,
        ),
        min_review_remaining_seconds=_parse_positive_float(
            values.get("FIREWORKS_LEADERBOARD_MIN_REVIEW_REMAINING_SECONDS"),
            DEFAULT_MIN_REVIEW_REMAINING_SECONDS,
        ),
        max_tokens=_parse_positive_int(values.get("GEMMA_MAX_TOKENS"), DEFAULT_MAX_TOKENS),
    )


def _model_value(values: Mapping[str, str], key: str, default: str) -> str:
    value = str(values.get(key, "")).strip()
    return value or default


def _parse_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "t", "yes", "y", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off", "disabled"}:
        return False
    return default


def _parse_temperature(value: object, default: float) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return min(2.0, max(0.0, parsed))


def _parse_positive_float(value: object, default: float) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed) or parsed <= 0:
        return default
    return parsed


def _parse_positive_int(value: object, default: int) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
