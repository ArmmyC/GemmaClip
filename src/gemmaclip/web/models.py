from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from gemmaclip.routed import GenerationOutcome


def _camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=_camel, populate_by_name=True, extra="forbid")


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"


class ConfigResponse(ApiModel):
    maximum_upload_size: int
    supported_video_types: list[str]
    available_caption_styles: list[str]
    gemma_credentials_configured: bool
    audio_mode_available: bool


class PresetRequest(ApiModel):
    preset: Literal["fast", "balanced", "maximum", "custom"] = "balanced"


class RunResponse(ApiModel):
    id: str
    created_at: str
    status: Literal["pending", "processing", "ready", "error"]
    video: dict[str, Any]
    preset: str
    frames: dict[str, Any]
    audio: dict[str, Any]
    evidence: dict[str, Any]
    captions: dict[str, Any]
    experiments: list[dict[str, Any]]
    stages: dict[str, str]
    active_stage: str | None = None
    progress_message: str | None = None
    error: str | None = None
    generation_outcome: GenerationOutcome | None = None
    degraded: bool = False


class StatusResponse(ApiModel):
    id: str
    status: Literal["pending", "processing", "ready", "error"]
    active_stage: str | None = None
    progress_message: str | None = None
    stages: dict[str, str]
    error: str | None = None
    generation_outcome: GenerationOutcome | None = None
    degraded: bool = False


class ApiErrorResponse(ApiModel):
    detail: str = Field(min_length=1)
