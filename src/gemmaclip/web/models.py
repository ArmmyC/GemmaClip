from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gemmaclip.routed import GenerationOutcome


def _camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ApiModel(BaseModel):
    model_config = ConfigDict(alias_generator=_camel, populate_by_name=True, extra="forbid")


class MediaToolsResponse(ApiModel):
    ffmpeg: Literal["available", "unavailable"]
    ffprobe: Literal["available", "unavailable"]


class HealthResponse(ApiModel):
    status: Literal["ok", "degraded", "unavailable"]
    storage: Literal["available", "unavailable"]
    media_tools: MediaToolsResponse
    providers_configured: bool
    job_manager: Literal["available", "unavailable"]
    version: str = "0.1.0"


class ConfigResponse(ApiModel):
    maximum_upload_size: int
    supported_video_types: list[str]
    available_caption_styles: list[str]
    gemma_credentials_configured: bool
    audio_mode_available: bool


class PresetRequest(ApiModel):
    preset: Literal["fast", "balanced", "maximum", "custom"] = "balanced"


class FrameRequest(ApiModel):
    method: Literal["uniform", "aks-lite", "hybrid"] = "hybrid"
    total_frames: int = Field(default=6, ge=6, le=16)
    anchor_count: int = Field(default=4, ge=0, le=16)
    high_change_count: int = Field(default=2, ge=0, le=16)
    min_spacing_sec: float = Field(default=1.0, gt=0, le=5)
    change_sensitivity: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def validate_selection_counts(self) -> "FrameRequest":
        if self.anchor_count + self.high_change_count > self.total_frames:
            raise ValueError("anchorCount and highChangeCount must fit within totalFrames.")
        return self


class FrameSelectionRequest(ApiModel):
    included_frame_ids: list[str] = Field(min_length=6, max_length=16)


class AudioRequest(ApiModel):
    mode: Literal["disabled", "automatic", "always"] = "automatic"
    max_duration_sec: float = Field(default=30, ge=1, le=30)
    sample_rate_hz: int = Field(default=16000, gt=0, le=192000)
    min_rms_energy: float = Field(default=0.01, ge=0, le=1)
    strategy: Literal["highest-energy", "first-non-silent"] = "highest-energy"


class EvidenceRequest(ApiModel):
    route: Literal["auto", "automatic", "gemma-4-26b-a4b", "gemma-4-12b-unified"] = "auto"
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=128, le=8192)
    provider: str = "automatic"
    show_prompt_structure: bool = False
    show_raw_json: bool = True


class CaptionRequest(ApiModel):
    temperature: float = Field(default=0.4, ge=0, le=2)
    min_words: int = Field(default=18, ge=8, le=40)
    max_words: int = Field(default=35, ge=8, le=40)
    strict_grounding: bool = True
    audio_evidence_mode: Literal["ignore", "use-if-present", "require"] = "use-if-present"
    focused_repair: bool = True
    styles: list[Literal["formal", "sarcastic", "humorous-tech", "humorous-non-tech"]] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_word_bounds(self) -> "CaptionRequest":
        if self.min_words > self.max_words:
            raise ValueError("minWords must be less than or equal to maxWords.")
        if not self.strict_grounding:
            raise ValueError("Strict grounding is required for Gemma Lab captions.")
        return self


class ExperimentRequest(ApiModel):
    label: str | None = Field(default=None, max_length=120)
    caption_style: Literal["formal", "sarcastic", "humorous-tech", "humorous-non-tech"] = "formal"


class RunResponse(ApiModel):
    id: str
    created_at: str
    status: Literal["pending", "processing", "ready", "error"]
    mode: Literal["manual", "quick"] = "manual"
    video: dict[str, Any]
    preset: str
    frames: dict[str, Any]
    audio: dict[str, Any]
    evidence: dict[str, Any]
    captions: dict[str, Any]
    experiments: list[dict[str, Any]]
    stages: dict[str, str]
    runtimes: dict[str, float] = {}
    stage_errors: dict[str, str] = {}
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
    runtimes: dict[str, float] = {}
    error: str | None = None
    generation_outcome: GenerationOutcome | None = None
    degraded: bool = False


class ApiErrorResponse(ApiModel):
    detail: str = Field(min_length=1)
