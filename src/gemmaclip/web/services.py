from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemmaclip.audio import has_audio_stream
from gemmaclip.captioner import generate_captions
from gemmaclip.frames import ExtractedFrame, extract_frames
from gemmaclip.gemma_client import create_model_client, load_routed_gemma_config
from gemmaclip.io import Task
from gemmaclip.video import VideoMetadata, probe_video
from gemmaclip.web.adapters import adapt_captions, adapt_evidence, adapt_frames, adapt_video_metadata, selected_route_from_evidence
from gemmaclip.web.storage import RunStorage


DEFAULT_STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")
WEB_JOB_RUNTIME_SECONDS = 570.0


class WebConfigurationError(RuntimeError):
    pass


class WebPipelineError(RuntimeError):
    pass


@dataclass(slots=True)
class PipelineDependencies:
    probe_fn: Callable[..., VideoMetadata] = probe_video
    audio_probe_fn: Callable[..., bool] = has_audio_stream
    frame_extract_fn: Callable[..., list[ExtractedFrame]] = extract_frames
    caption_generate_fn: Callable[..., dict[str, str]] = generate_captions
    client_factory: Callable[[Any], Any] = create_model_client


class WebServices:
    def __init__(
        self,
        storage: RunStorage,
        *,
        env: Mapping[str, str] | None = None,
        dependencies: PipelineDependencies | None = None,
    ) -> None:
        self.storage = storage
        self.env = dict(env if env is not None else os.environ)
        self.dependencies = dependencies or PipelineDependencies()

    def credentials_configured(self) -> bool:
        return load_routed_gemma_config(self._routed_env()).has_credentials

    def ensure_credentials(self) -> None:
        if not self.credentials_configured():
            raise WebConfigurationError("Gemma credentials are not configured for routed captioning.")

    def probe_run_video(self, run_id: str) -> dict[str, Any]:
        video_path = self.storage.input_path(run_id)
        run = self.storage.read_run(run_id)
        try:
            metadata = self.dependencies.probe_fn(video_path)
            audio_present = self.dependencies.audio_probe_fn(video_path)
        except Exception as exc:
            raise WebPipelineError("This video could not be decoded. Try MP4 with H.264 video.") from exc
        adapted = adapt_video_metadata(
            run["video"]["filename"], metadata,
            size_bytes=video_path.stat().st_size,
            has_audio_stream=audio_present,
        )

        def update(payload: dict[str, Any]) -> None:
            payload["video"] = adapted
            payload["audio"]["segment"]["hasAudioStream"] = audio_present
            payload["stages"]["video"] = "complete"
            payload["activeStage"] = "video"
            payload["progressMessage"] = "Inspecting video"

        return self.storage.update_run(run_id, update)

    def apply_preset(self, run_id: str, preset: str) -> dict[str, Any]:
        def update(payload: dict[str, Any]) -> None:
            payload["preset"] = preset
        return self.storage.update_run(run_id, update)

    def run_quick_caption(self, run_id: str, progress: Callable[[str, str], None] | None = None) -> dict[str, Any]:
        self.ensure_credentials()
        started_at = time.monotonic()
        deadline = started_at + WEB_JOB_RUNTIME_SECONDS
        notify = progress or (lambda stage, message: None)

        notify("video", "Inspecting video")
        run = self.probe_run_video(run_id)
        video_path = self.storage.input_path(run_id)
        metadata = self.dependencies.probe_fn(video_path)

        notify("frames", "Selecting important moments")
        processing_root = self.storage.run_dir(run_id) / "_processing_frames"
        try:
            extracted = self.dependencies.frame_extract_fn(
                run_id,
                video_path,
                metadata,
                strategy="aks-lite",
                destination_root=processing_root,
                fireworks_judge=True,
                command_timeout_seconds=15.0,
                env=self._routed_env(),
            )
            if len(extracted) != 6:
                raise WebPipelineError("Frame extraction did not produce the required six-frame Balanced preset.")
            frames_dir = self.storage.clear_frames(run_id)
            persisted_frames: list[ExtractedFrame] = []
            for index, frame in enumerate(extracted, start=1):
                destination = frames_dir / f"frame_{index:03d}.jpg"
                shutil.copy2(frame.path, destination)
                persisted_frames.append(ExtractedFrame(destination, frame.timestamp_seconds, frame.frame_role, frame.change_score))
        except WebPipelineError:
            raise
        except Exception as exc:
            raise WebPipelineError("GemmaClip could not select representative frames from this video.") from exc
        finally:
            shutil.rmtree(processing_root, ignore_errors=True)

        frame_payload = adapt_frames(run_id, persisted_frames)
        self.storage.update_run(run_id, lambda payload: _store_frames(payload, frame_payload))

        def routed_stage(stage: str) -> None:
            mapping = {
                "checking_audio": ("audio", "Checking audio"),
                "building_evidence": ("evidence", "Building grounded evidence"),
                "writing_captions": ("captions", "Writing captions"),
            }
            if stage in mapping:
                active_stage, message = mapping[stage]
                notify(active_stage, message)

        task = Task(run_id, "web-upload", DEFAULT_STYLES)
        debug_dir = self.storage.run_dir(run_id) / "debug"
        try:
            captions = self.dependencies.caption_generate_fn(
                task,
                persisted_frames,
                env=self._routed_env(),
                video_path=video_path,
                debug_dir=debug_dir,
                client_factory=self.dependencies.client_factory,
                remaining_time_fn=lambda: max(0.0, deadline - time.monotonic()),
                stage_callback=routed_stage,
            )
        except Exception as exc:
            raise WebPipelineError("Gemma captioning failed. Check the configured Gemma deployment and try again.") from exc

        evidence = self._load_evidence_debug(run_id)
        selected_route, route_reason = selected_route_from_evidence(evidence)
        adapted_evidence = adapt_evidence(evidence, selected_route=selected_route, route_reason=route_reason)
        adapted_caption_results = adapt_captions(captions)
        self.storage.write_artifact_json(run_id, "debug/evidence.json", evidence)
        self.storage.write_artifact_json(run_id, "debug/captions.json", captions)
        self.storage.write_artifact_json(run_id, "results/captions.json", captions)

        def complete(payload: dict[str, Any]) -> None:
            payload["evidence"]["result"] = adapted_evidence
            payload["captions"]["results"] = adapted_caption_results
            _store_audio_from_evidence(payload, evidence)
            payload["status"] = "ready"
            payload["activeStage"] = "compare"
            payload["progressMessage"] = "Complete"
            payload["error"] = None
            for stage in ("video", "frames", "audio", "evidence", "captions"):
                payload["stages"][stage] = "complete"
            payload["stages"]["compare"] = "waiting"

        notify("captions", "Writing captions")
        return self.storage.update_run(run_id, complete)

    def _load_evidence_debug(self, run_id: str) -> dict[str, Any]:
        debug_dir = self.storage.run_dir(run_id) / "debug"
        candidates = sorted(debug_dir.glob("*_routed_evidence.json"))
        if not candidates:
            return {}
        try:
            payload = json.loads(candidates[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _routed_env(self) -> dict[str, str]:
        values = dict(self.env)
        values["GEMMACLIP_PROVIDER"] = "routed_gemma"
        values.setdefault("GEMMACLIP_AUDIO_MODE", "auto")
        return values


def _store_frames(payload: dict[str, Any], frames: list[dict[str, Any]]) -> None:
    payload["frames"]["frames"] = frames
    payload["frames"]["changeSeries"] = [
        {"t": frame["timestampSec"], "score": frame["changeScore"]}
        for frame in frames
    ]
    payload["stages"]["frames"] = "complete"


def _store_audio_from_evidence(payload: dict[str, Any], evidence: Mapping[str, Any]) -> None:
    audio = evidence.get("audio") if isinstance(evidence.get("audio"), Mapping) else {}
    start = float(audio.get("window_start_seconds") or 0.0)
    duration = float(audio.get("window_duration_seconds") or 0.0)
    status = str(audio.get("status") or "unavailable")
    payload["audio"]["segment"].update(
        {
            "startSec": start,
            "endSec": start + duration,
            "energyCandidateFound": bool(audio.get("available")) and duration > 0,
            "routeExplanation": "The temporary selected audio window was analyzed and then securely removed." if duration > 0 else "No persisted audio artifact is available for this run.",
            "artifactAvailable": False,
            "status": status,
            "waveform": [],
        }
    )
