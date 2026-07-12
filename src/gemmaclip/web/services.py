from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gemmaclip.audio import (
    AudioEvidenceCandidate,
    AudioSettings,
    has_audio_stream,
    prepare_audio_candidate,
    read_pcm16_mono,
    unavailable_audio,
)
from gemmaclip.captioner import generate_captions
from gemmaclip.frames import ExtractedFrame, extract_configured_frames
from gemmaclip.gemma_client import create_model_client, load_routed_gemma_config
from gemmaclip.io import Task
from gemmaclip.routed import (
    EvidenceExecution,
    GenerationOutcome,
    RouteDecision,
    decide_evidence_route,
    generate_routed_captions_from_evidence,
    generate_routed_evidence,
    GENERATION_OUTCOME_DETERMINISTIC_FALLBACK,
    GENERATION_OUTCOME_EVIDENCE_FALLBACK,
    GENERATION_OUTCOME_MODEL,
)
from gemmaclip.routed import caption_evidence_for_mode
from gemmaclip.video import VideoMetadata, probe_video
from gemmaclip.web.adapters import (
    adapt_captions,
    adapt_evidence,
    adapt_frames,
    adapt_video_metadata,
    backend_style_to_frontend,
    frontend_style_to_backend,
    selected_route_from_evidence,
)
from gemmaclip.web.storage import RunStorage


DEFAULT_STYLES = ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")
WEB_JOB_RUNTIME_SECONDS = 570.0
STAGES = ("video", "frames", "audio", "evidence", "captions", "compare")


class WebConfigurationError(RuntimeError):
    pass


class WebPipelineError(RuntimeError):
    pass


@dataclass(slots=True)
class PipelineDependencies:
    probe_fn: Callable[..., VideoMetadata] = probe_video
    audio_probe_fn: Callable[..., bool] = has_audio_stream
    frame_extract_fn: Callable[..., list[ExtractedFrame]] = extract_configured_frames
    audio_prepare_fn: Callable[..., AudioEvidenceCandidate] = prepare_audio_candidate
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
        started = time.monotonic()
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
            payload["status"] = "pending" if payload.get("mode") == "manual" else payload.get("status", "processing")
            payload["stageErrors"].pop("video", None)
            payload["runtimes"]["video"] = round(time.monotonic() - started, 3)

        return self.storage.update_run(run_id, update)

    def apply_preset(self, run_id: str, preset: str) -> dict[str, Any]:
        if preset not in {"fast", "balanced", "maximum", "custom"}:
            raise WebPipelineError("Unsupported processing preset.")

        def update(payload: dict[str, Any]) -> None:
            payload["preset"] = preset
            if preset == "fast":
                payload["frames"]["config"].update({"method": "uniform", "totalFrames": 6, "anchorCount": 0, "highChangeCount": 0})
                payload["audio"]["config"]["mode"] = "disabled"
            elif preset == "balanced":
                payload["frames"]["config"].update({"method": "hybrid", "totalFrames": 6, "anchorCount": 4, "highChangeCount": 2, "minSpacingSec": 1.0, "changeSensitivity": 0.5})
                payload["audio"]["config"]["mode"] = "automatic"
            elif preset == "maximum":
                payload["frames"]["config"].update({"method": "hybrid", "totalFrames": 12, "anchorCount": 4, "highChangeCount": 8, "minSpacingSec": 0.5, "changeSensitivity": 0.5})
                payload["audio"]["config"]["mode"] = "always"
            had_artifacts = bool(payload["frames"]["frames"] or payload["evidence"]["result"] or payload["captions"]["results"] or payload["audio"]["segment"].get("energyCandidateFound"))
            if had_artifacts:
                _invalidate(payload, "video")
            payload["stages"]["video"] = "complete"
            payload["status"] = "processing" if payload.get("mode") == "quick" else "pending"
            payload["activeStage"] = "video"
            payload["progressMessage"] = "Preset updated"

        return self.storage.update_run(run_id, update)

    def extract_run_frames(self, run_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
        values = _validate_frame_config(config)
        run = self.storage.read_run(run_id)
        if run["stages"].get("video") != "complete":
            raise WebPipelineError("Complete video metadata is required before extracting frames.")
        video_path = self.storage.input_path(run_id)
        metadata = _metadata_from_run(run["video"])
        started = time.monotonic()
        processing_root = self.storage.run_dir(run_id) / "_processing_frames"
        shutil.rmtree(processing_root, ignore_errors=True)
        try:
            extracted = self.dependencies.frame_extract_fn(
                run_id,
                video_path,
                metadata,
                strategy=values["method"],
                total_frames=values["totalFrames"],
                anchor_count=values["anchorCount"],
                high_change_count=values["highChangeCount"],
                min_spacing_seconds=values["minSpacingSec"],
                change_sensitivity=values["changeSensitivity"],
                destination_root=processing_root,
                command_timeout_seconds=15.0,
                env=self._routed_env(),
            )
            if len(extracted) < 6 or len(extracted) > 16:
                raise WebPipelineError("Frame extraction did not produce the required six-frame evidence set.")
            if any(not frame.path.is_file() for frame in extracted):
                raise WebPipelineError("Frame extraction produced an incomplete artifact.")
            ordered = sorted(extracted, key=lambda frame: frame.timestamp_seconds)
            staged_frames = adapt_frames(run_id, ordered)
            _replace_frames_directory(self.storage.run_dir(run_id) / "frames", [frame.path for frame in ordered])
        except WebPipelineError:
            raise
        except Exception as exc:
            raise WebPipelineError("GemmaClip could not select representative frames from this video.") from exc
        finally:
            shutil.rmtree(processing_root, ignore_errors=True)

        def complete(payload: dict[str, Any]) -> None:
            payload["frames"]["config"] = values
            payload["frames"]["frames"] = staged_frames
            payload["frames"]["changeSeries"] = [
                {"t": frame["timestampSec"], "score": frame["changeScore"]}
                for frame in staged_frames
            ]
            payload["stages"]["frames"] = "complete"
            _invalidate(payload, "frames")
            payload["stages"]["frames"] = "complete"
            _mark_stage_ready(payload, "frames", "Frames updated", time.monotonic() - started)

        return self.storage.update_run(run_id, complete)

    def update_frame_selection(self, run_id: str, included_ids: Sequence[str]) -> dict[str, Any]:
        run = self.storage.read_run(run_id)
        if run["stages"].get("frames") != "complete":
            raise WebPipelineError("Complete frames are required before changing selection.")
        known = {frame["id"] for frame in run["frames"]["frames"]}
        selected = list(dict.fromkeys(str(item) for item in included_ids))
        if any(item not in known for item in selected):
            raise WebPipelineError("The frame selection contains an unknown frame.")
        if len(selected) < 6:
            raise WebPipelineError("Select at least six frames for evidence generation.")

        def update(payload: dict[str, Any]) -> None:
            for frame in payload["frames"]["frames"]:
                frame["included"] = frame["id"] in selected
            _invalidate(payload, "frames")
            payload["stages"]["frames"] = "complete"
            _mark_stage_ready(payload, "frames", "Frame selection updated", 0.0)

        return self.storage.update_run(run_id, update)

    def analyze_run_audio(self, run_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
        values = _validate_audio_config(config)
        run = self.storage.read_run(run_id)
        if run["stages"].get("video") != "complete":
            raise WebPipelineError("Complete video metadata is required before analyzing audio.")
        started = time.monotonic()
        processing_root = self.storage.run_dir(run_id) / "_processing_audio"
        shutil.rmtree(processing_root, ignore_errors=True)
        candidate: AudioEvidenceCandidate | None = None
        try:
            settings = AudioSettings(
                mode=values["mode"],
                max_seconds=values["maxDurationSec"],
                sample_rate=values["sampleRateHz"],
                min_rms=values["minRmsEnergy"],
                strategy=values["strategy"],
            )
            try:
                candidate = self.dependencies.audio_prepare_fn(self.storage.input_path(run_id), processing_root, settings=settings)
            except Exception:
                candidate = unavailable_audio(settings.sample_rate, "audio extraction failed; visual-only continuation is available")
            waveform = _waveform(candidate.path) if candidate.path else []
            status = "usable" if candidate.energy_candidate else "silent" if candidate.silent else "unavailable"
            segment = {
                "startSec": candidate.start_seconds,
                "endSec": candidate.start_seconds + candidate.duration_seconds,
                "rms": candidate.rms,
                "waveform": waveform,
                "hasAudioStream": bool(run["video"].get("hasAudioStream")),
                "energyCandidateFound": candidate.energy_candidate,
                "routeExplanation": candidate.reason,
                "artifactAvailable": False,
                "status": status,
            }
        except Exception as exc:
            raise WebPipelineError("Audio analysis failed safely. The visual-only route remains available.") from exc
        finally:
            if candidate is not None:
                _cleanup_candidate(candidate)
            shutil.rmtree(processing_root, ignore_errors=True)

        def complete(payload: dict[str, Any]) -> None:
            payload["audio"]["config"] = values
            payload["audio"]["segment"] = segment
            _invalidate(payload, "audio")
            payload["stages"]["audio"] = "complete"
            _mark_stage_ready(payload, "audio", "Audio analyzed", time.monotonic() - started)

        return self.storage.update_run(run_id, complete)

    def generate_run_evidence(self, run_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
        values = _validate_evidence_config(config)
        run = self.storage.read_run(run_id)
        if run["stages"].get("frames") != "complete":
            raise WebPipelineError("Run Frames before generating evidence.")
        included = [frame for frame in run["frames"]["frames"] if frame.get("included", True)]
        if len(included) < 6:
            raise WebPipelineError("Evidence generation requires at least six included frames.")
        self.ensure_credentials()
        frames = _frames_from_run(self.storage, run_id, included)
        candidate: AudioEvidenceCandidate | None = None
        started = time.monotonic()
        try:
            audio_config = _audio_settings_from_run(run)
            if values["route"] == "gemma-4-12b-unified":
                audio_config = AudioSettings(
                    mode="always",
                    max_seconds=audio_config.max_seconds,
                    sample_rate=audio_config.sample_rate,
                    min_rms=audio_config.min_rms,
                    strategy=audio_config.strategy,
                    min_remaining_seconds=audio_config.min_remaining_seconds,
                    command_timeout_seconds=audio_config.command_timeout_seconds,
                )
            try:
                candidate = self.dependencies.audio_prepare_fn(
                    self.storage.input_path(run_id),
                    self.storage.run_dir(run_id) / "_processing_evidence_audio",
                    settings=audio_config,
                )
            except Exception:
                candidate = unavailable_audio(audio_config.sample_rate, "audio extraction failed; visual-only evidence fallback is available")
            if values["route"] == "gemma-4-26b-a4b":
                decision = RouteDecision("visual", False, "visual-only route selected by the user")
            elif values["route"] == "gemma-4-12b-unified":
                decision = RouteDecision("audio_visual", bool(candidate.path), "audio-visual route selected by the user")
            else:
                decision = decide_evidence_route(audio_config, candidate, WEB_JOB_RUNTIME_SECONDS)
            routed = load_routed_gemma_config(self._routed_env())
            raw_evidence, execution = generate_routed_evidence(
                run_id,
                frames,
                candidate,
                decision,
                config=routed,
                client_factory=self.dependencies.client_factory,
                remaining_time_fn=lambda: WEB_JOB_RUNTIME_SECONDS,
                temperature=values["temperature"],
                max_tokens=values["maxTokens"],
            )
        except WebConfigurationError:
            raise
        except Exception as exc:
            raise WebPipelineError("Evidence generation failed safely. You can retry or choose a visual route.") from exc
        finally:
            if candidate is not None:
                _cleanup_candidate(candidate)
            shutil.rmtree(self.storage.run_dir(run_id) / "_processing_evidence_audio", ignore_errors=True)

        selected_route, route_reason = selected_route_from_evidence(raw_evidence, execution)
        adapted = adapt_evidence(raw_evidence, selected_route=selected_route, route_reason=route_reason, execution=execution)
        self.storage.write_artifact_json(run_id, "results/evidence.json", raw_evidence)

        def complete(payload: dict[str, Any]) -> None:
            payload["evidence"]["config"] = values
            payload["evidence"]["result"] = adapted
            _invalidate(payload, "evidence")
            payload["stages"]["evidence"] = "complete"
            _mark_stage_ready(payload, "evidence", "Evidence updated", time.monotonic() - started)

        return self.storage.update_run(run_id, complete)

    def generate_run_captions(self, run_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
        values = _validate_caption_config(config)
        started = time.monotonic()
        run = self.storage.read_run(run_id)
        if run["stages"].get("evidence") != "complete":
            raise WebPipelineError("Build current evidence before generating captions.")
        included = [frame for frame in run["frames"]["frames"] if frame.get("included", True)]
        if len(included) < 6:
            raise WebPipelineError("Caption generation requires at least six included frames.")
        self.ensure_credentials()
        try:
            evidence = self.storage.read_artifact_json(run_id, "results/evidence.json")
        except Exception as exc:
            raise WebPipelineError("Current evidence is unavailable. Rebuild evidence before captions.") from exc
        frames = _frames_from_run(self.storage, run_id, included)
        backend_styles = tuple(frontend_style_to_backend(style) for style in values["styles"])
        task = Task(run_id, "web-upload", backend_styles)
        outcome: GenerationOutcome | None = None

        def capture(value: GenerationOutcome) -> None:
            nonlocal outcome
            outcome = value

        try:
            captions = generate_routed_captions_from_evidence(
                task,
                frames,
                evidence,
                config=load_routed_gemma_config(self._routed_env()),
                client_factory=self.dependencies.client_factory,
                remaining_time_fn=lambda: WEB_JOB_RUNTIME_SECONDS,
                temperature=values["temperature"],
                repair_temperature=0.25,
                min_words=values["minWords"],
                max_words=values["maxWords"],
                audio_evidence_mode=values["audioEvidenceMode"],
                focused_repair=values["focusedRepair"],
                strict_grounding=values["strictGrounding"],
                outcome_callback=capture,
            )
        except Exception as exc:
            raise WebPipelineError("Caption generation failed safely. Please retry after rebuilding evidence.") from exc
        if outcome == GENERATION_OUTCOME_DETERMINISTIC_FALLBACK:
            raise WebPipelineError("Gemma could not produce grounded captions for this evidence.")
        adapted = adapt_captions(captions, caption_evidence_for_mode(evidence, values["audioEvidenceMode"]))
        self.storage.write_artifact_json(run_id, "results/captions.json", captions)

        def complete(payload: dict[str, Any]) -> None:
            payload["captions"]["config"] = values
            payload["captions"]["results"] = adapted
            _invalidate(payload, "captions")
            payload["stages"]["captions"] = "complete"
            _store_generation_outcome(payload, outcome or GENERATION_OUTCOME_MODEL, (outcome == GENERATION_OUTCOME_EVIDENCE_FALLBACK))
            _mark_stage_ready(payload, "captions", "Captions generated", time.monotonic() - started)

        return self.storage.update_run(run_id, complete)

    def create_run_experiment(self, run_id: str, label: str | None = None, caption_style: str = "formal") -> dict[str, Any]:
        run = self.storage.read_run(run_id)
        if run["stages"].get("captions") != "complete":
            raise WebPipelineError("Generate current captions before saving an experiment.")
        styles = {item["style"]: item for item in run["captions"]["results"]}
        if caption_style not in styles:
            raise WebPipelineError("The requested experiment caption style is not available.")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        selected = [frame for frame in run["frames"]["frames"] if frame.get("included", True)]
        evidence = run["evidence"]["result"]
        caption = styles[caption_style]
        experiment = {
            "id": f"experiment_{os.urandom(8).hex()}",
            "label": (label or f"Experiment {len(run['experiments']) + 1}")[:120],
            "createdAt": now,
            "frameMethod": run["frames"]["config"]["method"],
            "frameCount": len(selected),
            "includedFrameIds": [frame["id"] for frame in selected],
            "timestamps": [frame["timestampSec"] for frame in selected],
            "audioMode": run["audio"]["config"]["mode"],
            "audioStatus": run["audio"]["segment"]["status"],
            "evidenceRoute": evidence.get("selectedRoute"),
            "evidenceProvider": evidence.get("routeProvider"),
            "evidenceModel": evidence.get("routeModel"),
            "evidenceModality": evidence.get("routeModality"),
            "audioFallbackOccurred": bool(evidence.get("audioFallbackOccurred")),
            "evidenceTemperature": run["evidence"]["config"]["temperature"],
            "captionTemperature": run["captions"]["config"]["temperature"],
            "runtimeMs": round(sum(float(value) for value in run.get("runtimes", {}).values()) * 1000),
            "caption": caption["text"],
            "style": caption_style,
            "generationOutcome": run.get("generationOutcome"),
            "degraded": bool(run.get("degraded", False)),
        }
        self.storage.write_artifact_json(run_id, f"snapshots/{experiment['id']}.json", experiment)
        return self.storage.update_run(run_id, lambda payload: payload["experiments"].append(experiment) or payload)

    def compare_experiments(self, run_id: str, left_id: str, right_id: str) -> dict[str, Any]:
        run = self.storage.read_run(run_id)
        snapshots = {item["id"]: item for item in run["experiments"]}
        if left_id not in snapshots or right_id not in snapshots or left_id == right_id:
            raise WebPipelineError("Choose two different saved experiments from this run.")
        left, right = snapshots[left_id], snapshots[right_id]
        fields = ("frameMethod", "frameCount", "audioMode", "evidenceRoute", "evidenceProvider", "evidenceModel", "evidenceModality", "evidenceTemperature", "captionTemperature", "runtimeMs", "generationOutcome", "degraded")
        return {"left": left, "right": right, "differences": {field: {"left": left.get(field), "right": right.get(field)} for field in fields if left.get(field) != right.get(field)}}

    def run_quick_caption(self, run_id: str, progress: Callable[[str, str], None] | None = None) -> dict[str, Any]:
        self.ensure_credentials()
        notify = progress or (lambda stage, message: None)
        self.storage.update_run(run_id, lambda payload: payload.update(mode="quick"))
        notify("video", "Inspecting video")
        self.probe_run_video(run_id)
        balanced = {"method": "hybrid", "totalFrames": 6, "anchorCount": 4, "highChangeCount": 2, "minSpacingSec": 1.0, "changeSensitivity": 0.5}
        notify("frames", "Selecting important moments")
        self.extract_run_frames(run_id, balanced)
        notify("audio", "Checking audio")
        self.analyze_run_audio(run_id, {"mode": "automatic", "maxDurationSec": 30, "sampleRateHz": 16000, "minRmsEnergy": 0.01, "strategy": "highest-energy"})

        # Test and embedding callers may inject the existing caption function.
        # The production path uses the same individual stage services as Lab.
        if self.dependencies.caption_generate_fn is not generate_captions:
            run = self.storage.read_run(run_id)
            frames = _frames_from_run(self.storage, run_id, run["frames"]["frames"])
            debug_dir = self.storage.run_dir(run_id) / "debug"
            outcome: GenerationOutcome | None = None
            execution: EvidenceExecution | None = None
            def capture(value: GenerationOutcome) -> None:
                nonlocal outcome
                outcome = value
            def capture_execution(value: EvidenceExecution) -> None:
                nonlocal execution
                execution = value
            captions = self.dependencies.caption_generate_fn(
                Task(run_id, "web-upload", DEFAULT_STYLES), frames,
                env=self._routed_env(), video_path=self.storage.input_path(run_id), debug_dir=debug_dir,
                client_factory=self.dependencies.client_factory, remaining_time_fn=lambda: WEB_JOB_RUNTIME_SECONDS,
                stage_callback=lambda stage: notify(stage, {"building_evidence": "Building grounded evidence", "writing_captions": "Writing captions"}.get(stage, stage)),
                outcome_callback=capture, evidence_execution_callback=capture_execution,
            )
            evidence = self._load_evidence_debug(run_id)
            if outcome == GENERATION_OUTCOME_EVIDENCE_FALLBACK and not evidence:
                raise WebPipelineError("Gemma captioning could not preserve grounded evidence. Please retry.")
            if outcome == GENERATION_OUTCOME_DETERMINISTIC_FALLBACK or not evidence:
                self.storage.update_run(run_id, lambda payload: _store_generation_outcome(payload, GENERATION_OUTCOME_DETERMINISTIC_FALLBACK, False))
                raise WebPipelineError("Gemma could not produce grounded evidence for this video. Please retry or check the configured deployment.")
            if set(captions) != set(DEFAULT_STYLES) or any(not str(captions.get(style, "")).strip() for style in DEFAULT_STYLES):
                raise WebPipelineError("Gemma captioning did not produce all required caption styles. Please retry.")
            execution = execution or EvidenceExecution("unknown", "unknown", "visual", False, False, False, None)
            selected_route, route_reason = selected_route_from_evidence(evidence, execution)
            adapted_evidence = adapt_evidence(evidence, selected_route=selected_route, route_reason=route_reason, execution=execution)
            adapted_captions = adapt_captions(captions, evidence)
            self.storage.write_artifact_json(run_id, "results/evidence.json", evidence)
            self.storage.write_artifact_json(run_id, "results/captions.json", captions)
            def complete(payload: dict[str, Any]) -> None:
                payload["evidence"]["result"] = adapted_evidence
                payload["captions"]["results"] = adapted_captions
                for stage in ("video", "frames", "audio", "evidence", "captions"):
                    payload["stages"][stage] = "complete"
                payload["stages"]["compare"] = "waiting"
                _store_generation_outcome(payload, outcome or GENERATION_OUTCOME_MODEL, outcome == GENERATION_OUTCOME_EVIDENCE_FALLBACK)
                payload["status"] = "ready"
                payload["activeStage"] = "compare"
                payload["progressMessage"] = "Complete"
            return self.storage.update_run(run_id, complete)

        notify("evidence", "Building grounded evidence")
        self.generate_run_evidence(run_id, {"route": "auto", "temperature": 0.0, "maxTokens": 2048, "provider": "automatic", "showPromptStructure": False, "showRawJson": True})
        notify("captions", "Writing captions")
        result = self.generate_run_captions(run_id, {"model": "gemma-4-31b", "temperature": 0.4, "minWords": 18, "maxWords": 35, "strictGrounding": True, "audioEvidenceMode": "use-if-present", "focusedRepair": True, "styles": ["formal", "sarcastic", "humorous-tech", "humorous-non-tech"]})
        self.storage.update_run(run_id, lambda payload: payload.update(status="ready", activeStage="compare", progressMessage="Complete"))
        return self.storage.public_run(self.storage.read_run(run_id))

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


def invalidate_downstream(payload: dict[str, Any], source_stage: str) -> None:
    _invalidate(payload, source_stage)


def _invalidate(payload: dict[str, Any], source_stage: str) -> None:
    dependents = {
        "video": ("frames", "audio", "evidence", "captions", "compare"),
        "frames": ("evidence", "captions", "compare"),
        "audio": ("evidence", "captions", "compare"),
        "evidence": ("captions", "compare"),
        "captions": ("compare",),
    }.get(source_stage, ())
    for stage in dependents:
        payload["stages"][stage] = "invalidated"
        payload["stageErrors"].pop(stage, None)
    # Keep prior evidence as a clearly stale artifact so the Lab remains
    # inspectable while a user prepares a replacement run.
    if "captions" in dependents:
        payload["captions"]["results"] = []
    if dependents:
        payload["generationOutcome"] = None
        payload["degraded"] = False
        payload["error"] = None


def _mark_stage_ready(payload: dict[str, Any], stage: str, message: str, elapsed: float) -> None:
    payload["status"] = "processing" if payload.get("mode") == "quick" else "ready"
    payload["activeStage"] = stage
    payload["progressMessage"] = message
    payload["error"] = None
    payload["stageErrors"].pop(stage, None)
    payload["runtimes"][stage] = round(float(elapsed), 3)


def _store_generation_outcome(payload: dict[str, Any], outcome: GenerationOutcome, degraded: bool) -> None:
    payload["generationOutcome"] = outcome
    payload["degraded"] = degraded


def _metadata_from_run(video: Mapping[str, Any]) -> VideoMetadata:
    return VideoMetadata(float(video.get("durationSec", 0)), float(video.get("fps", 0)), int(video.get("width", 0)), int(video.get("height", 0)), 0, str(video.get("codec", "unknown")))


def _frames_from_run(storage: RunStorage, run_id: str, frames: Sequence[Mapping[str, Any]]) -> list[ExtractedFrame]:
    result: list[ExtractedFrame] = []
    for frame in sorted(frames, key=lambda item: float(item.get("timestampSec", 0))):
        frame_id = str(frame.get("id", ""))
        path = storage.frame_path(run_id, frame_id)
        result.append(ExtractedFrame(path, float(frame.get("timestampSec", 0)), str(frame.get("reason", "uniform")), float(frame.get("changeScore", 0))))
    return result


def _replace_frames_directory(frames_dir: Path, paths: Sequence[Path]) -> None:
    run_dir = frames_dir.parent
    staging = run_dir / "frames.next"
    previous = run_dir / "frames.previous"
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(previous, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sorted(paths, key=lambda path: path.name), start=1):
        shutil.copy2(source, staging / f"frame_{index:03d}.jpg")
    if frames_dir.exists():
        frames_dir.replace(previous)
    staging.replace(frames_dir)
    shutil.rmtree(previous, ignore_errors=True)


def _waveform(path: Path, count: int = 128) -> list[float]:
    samples, _ = read_pcm16_mono(path)
    if not samples:
        return []
    step = max(1, len(samples) // count)
    return [min(1.0, max(0.0, (sum(abs(value) for value in samples[index:index + step]) / max(1, len(samples[index:index + step]))) / 32768.0)) for index in range(0, len(samples), step)][:count]


def _cleanup_candidate(candidate: AudioEvidenceCandidate) -> None:
    from gemmaclip.audio import cleanup_audio_candidate
    cleanup_audio_candidate(candidate)


def _validate_frame_config(config: Mapping[str, Any]) -> dict[str, Any]:
    method = str(config.get("method", "hybrid"))
    values = {
        "method": method,
        "totalFrames": int(config.get("totalFrames", 6)),
        "anchorCount": int(config.get("anchorCount", 4)),
        "highChangeCount": int(config.get("highChangeCount", 2)),
        "minSpacingSec": float(config.get("minSpacingSec", 1.0)),
        "changeSensitivity": float(config.get("changeSensitivity", 0.5)),
    }
    if method not in {"uniform", "aks-lite", "hybrid"} or not 6 <= values["totalFrames"] <= 16:
        raise WebPipelineError("Frame configuration is outside the supported range.")
    if min(values["anchorCount"], values["highChangeCount"]) < 0 or values["anchorCount"] + values["highChangeCount"] > values["totalFrames"]:
        raise WebPipelineError("Anchor and high-change counts must fit within total frames.")
    if not 0 < values["minSpacingSec"] <= 5 or not 0 <= values["changeSensitivity"] <= 1:
        raise WebPipelineError("Frame spacing and sensitivity are outside the supported range.")
    return values


def _validate_audio_config(config: Mapping[str, Any]) -> dict[str, Any]:
    mode = str(config.get("mode", "automatic"))
    if mode not in {"disabled", "automatic", "always", "off", "auto"}:
        raise WebPipelineError("Unsupported audio mode.")
    mode = {"disabled": "disabled", "automatic": "automatic", "always": "always", "off": "disabled", "auto": "automatic"}[mode]
    values = {"mode": mode, "maxDurationSec": float(config.get("maxDurationSec", 30)), "sampleRateHz": int(config.get("sampleRateHz", 16000)), "minRmsEnergy": float(config.get("minRmsEnergy", 0.01)), "strategy": str(config.get("strategy", "highest-energy"))}
    if not 1 <= values["maxDurationSec"] <= 30 or values["sampleRateHz"] <= 0 or not 0 <= values["minRmsEnergy"] <= 1 or values["strategy"] not in {"highest-energy", "first-non-silent"}:
        raise WebPipelineError("Audio configuration is outside the supported range.")
    return values


def _validate_evidence_config(config: Mapping[str, Any]) -> dict[str, Any]:
    route = str(config.get("route", "auto"))
    if route == "automatic":
        route = "auto"
    if route not in {"auto", "gemma-4-26b-a4b", "gemma-4-12b-unified"}:
        raise WebPipelineError("Unsupported evidence route.")
    temperature = float(config.get("temperature", 0.0)); max_tokens = int(config.get("maxTokens", 2048))
    if not 0 <= temperature <= 2 or not 128 <= max_tokens <= 8192:
        raise WebPipelineError("Evidence configuration is outside the supported range.")
    return {"route": route, "temperature": temperature, "maxTokens": max_tokens, "provider": str(config.get("provider", "automatic")), "showPromptStructure": bool(config.get("showPromptStructure", False)), "showRawJson": bool(config.get("showRawJson", True))}


def _validate_caption_config(config: Mapping[str, Any]) -> dict[str, Any]:
    styles = list(config.get("styles", DEFAULT_STYLES))
    if not styles or len(set(styles)) != len(styles) or any(style not in {"formal", "sarcastic", "humorous-tech", "humorous-non-tech"} for style in styles):
        raise WebPipelineError("Choose at least one supported caption style.")
    values = {"model": "gemma-4-31b", "temperature": float(config.get("temperature", 0.4)), "minWords": int(config.get("minWords", 18)), "maxWords": int(config.get("maxWords", 35)), "strictGrounding": bool(config.get("strictGrounding", True)), "audioEvidenceMode": str(config.get("audioEvidenceMode", "use-if-present")), "focusedRepair": bool(config.get("focusedRepair", True)), "styles": styles}
    if not 0 <= values["temperature"] <= 2 or not 1 <= values["minWords"] <= values["maxWords"] <= 120 or values["audioEvidenceMode"] not in {"ignore", "use-if-present", "require"} or not values["strictGrounding"]:
        raise WebPipelineError("Caption configuration is outside the supported range.")
    return values


def _audio_settings_from_run(run: Mapping[str, Any]) -> AudioSettings:
    config = run["audio"]["config"]
    mode = {"disabled": "off", "automatic": "auto", "always": "always"}.get(str(config.get("mode", "automatic")), "auto")
    return AudioSettings(mode=mode, max_seconds=float(config.get("maxDurationSec", 30)), sample_rate=int(config.get("sampleRateHz", 16000)), min_rms=float(config.get("minRmsEnergy", 0.01)), strategy=str(config.get("strategy", "highest-energy")))
