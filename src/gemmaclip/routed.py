from __future__ import annotations

import base64
import json
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from gemmaclip.audio import AudioEvidenceCandidate, AudioSettings, DEFAULT_AUDIO_MIN_REMAINING_SECONDS, cleanup_audio_candidate, load_audio_settings, prepare_audio_candidate, unavailable_audio
from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import DEFAULT_PROVIDER_GOOGLE, RoutedGemmaConfig
from gemmaclip.io import Task, safe_task_id


LOGGER = logging.getLogger("gemmaclip.routed")
AUDIO_ROUTE_MIN_SECONDS = DEFAULT_AUDIO_MIN_REMAINING_SECONDS
TWO_CALL_VISUAL_MIN_SECONDS = 130.0
SINGLE_CALL_MIN_SECONDS = 70.0
EVIDENCE_ATTEMPT_MIN_SECONDS = 95.0
FINAL_SYNTHESIS_MIN_SECONDS = 70.0
FOCUSED_REPAIR_MIN_SECONDS = 70.0
SINGLE_CALL_ATTEMPT_MIN_SECONDS = 70.0
DEFAULT_EVIDENCE_TEMPERATURE = 0.0
DEFAULT_CAPTION_TEMPERATURE = 0.4
DEFAULT_REPAIR_TEMPERATURE = 0.25
DEFAULT_SINGLE_CALL_TEMPERATURE = 0.4
VALID_AUDIO_STATUSES = {"usable", "uncertain", "silent", "unavailable", "failed"}
ALWAYS_UNSUPPORTED_CLAIMS = ["motive", "destination", "occupation", "deadline", "relationship", "off-camera event"]
CONDITIONAL_AUDIO_CLAIMS = ["sound", "dialogue", "speech", "music", "noise"]
GENERATION_OUTCOME_MODEL = "model_generated"
GENERATION_OUTCOME_EVIDENCE_FALLBACK = "evidence_fallback"
GENERATION_OUTCOME_DETERMINISTIC_FALLBACK = "deterministic_fallback"
GenerationOutcome = Literal["model_generated", "evidence_fallback", "deterministic_fallback"]


class RoutedRuntimeBudgetError(RuntimeError):
    """Raised before a routed provider attempt when the live budget is unsafe."""


@dataclass(frozen=True, slots=True)
class RoutedStageSettings:
    evidence_temperature: float = DEFAULT_EVIDENCE_TEMPERATURE
    caption_temperature: float = DEFAULT_CAPTION_TEMPERATURE
    repair_temperature: float = DEFAULT_REPAIR_TEMPERATURE
    single_call_temperature: float = DEFAULT_SINGLE_CALL_TEMPERATURE


def load_routed_stage_settings(env: Mapping[str, str]) -> RoutedStageSettings:
    return RoutedStageSettings(
        evidence_temperature=_parse_temperature(env.get("GEMMACLIP_ROUTED_EVIDENCE_TEMPERATURE"), DEFAULT_EVIDENCE_TEMPERATURE),
        caption_temperature=_parse_temperature(env.get("GEMMACLIP_ROUTED_CAPTION_TEMPERATURE"), DEFAULT_CAPTION_TEMPERATURE),
        repair_temperature=_parse_temperature(env.get("GEMMACLIP_ROUTED_REPAIR_TEMPERATURE"), DEFAULT_REPAIR_TEMPERATURE),
        single_call_temperature=_parse_temperature(env.get("GEMMACLIP_ROUTED_SINGLE_CALL_TEMPERATURE"), DEFAULT_SINGLE_CALL_TEMPERATURE),
    )


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    use_audio: bool
    reason: str


@dataclass(frozen=True, slots=True)
class EvidenceExecution:
    provider: str
    model: str
    modality: Literal["visual", "audio_visual"]
    audio_attempted: bool
    audio_used: bool
    fallback_used: bool
    fallback_reason: str | None


def decide_evidence_route(settings: AudioSettings, audio: AudioEvidenceCandidate, remaining_seconds: float) -> RouteDecision:
    if settings.mode == "off":
        return RouteDecision("visual", False, "audio mode is off")
    if remaining_seconds < settings.min_remaining_seconds:
        return RouteDecision("visual", False, "insufficient runtime for audio-visual evidence")
    useful = audio.available and audio.energy_candidate and not audio.silent and audio.duration_seconds > 0 and audio.path is not None
    if useful:
        return RouteDecision("audio_visual", True, f"useful audio selected in {settings.mode} mode")
    return RouteDecision("visual", False, audio.reason)


def generate_routed_evidence(
    task_id: str,
    frames: Sequence[ExtractedFrame],
    audio: AudioEvidenceCandidate,
    decision: RouteDecision,
    *,
    config: RoutedGemmaConfig,
    client_factory: Callable[[Any], Any],
    remaining_time_fn: Callable[[], float],
    temperature: float = DEFAULT_EVIDENCE_TEMPERATURE,
    max_tokens: int | None = None,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, Any], EvidenceExecution]:
    """Run one validated evidence stage for the interactive Lab.

    The provider-attempt loop is shared with the competition routed pipeline;
    this wrapper only makes the evidence half available without also writing
    captions.  Audio candidates are owned by the caller and are never retained
    by this function.
    """
    return _generate_validated_evidence(
        task_id,
        frames,
        audio,
        decision,
        config,
        client_factory,
        temperature=temperature,
        max_tokens=max_tokens,
        logger=logger or LOGGER,
        remaining_time_fn=remaining_time_fn,
    )


def generate_routed_captions_from_evidence(
    task: Task,
    frames: Sequence[ExtractedFrame],
    evidence: Mapping[str, Any],
    *,
    config: RoutedGemmaConfig,
    client_factory: Callable[[Any], Any],
    remaining_time_fn: Callable[[], float],
    temperature: float = DEFAULT_CAPTION_TEMPERATURE,
    repair_temperature: float = DEFAULT_REPAIR_TEMPERATURE,
    min_words: int = 18,
    max_words: int = 35,
    audio_evidence_mode: str = "use-if-present",
    focused_repair: bool = True,
    strict_grounding: bool = True,
    logger: logging.Logger | None = None,
    outcome_callback: Callable[[GenerationOutcome], None] | None = None,
) -> dict[str, str]:
    """Synthesize captions from persisted evidence without rerunning upstream stages."""
    from gemmaclip.captioner import build_evidence_based_captions, cleanup_caption

    active_logger = logger or LOGGER
    audio = unavailable_audio(16_000, "using persisted evidence")
    audio_raw = evidence.get("audio") if isinstance(evidence.get("audio"), Mapping) else {}
    audio_selected = bool(audio_raw.get("available"))
    if not strict_grounding:
        raise ValueError("Strict grounding is required for Gemma Lab captions.")
    if audio_evidence_mode == "require" and not _usable_caption_audio(audio_raw):
        raise ValueError("Caption configuration requires usable audio evidence.")
    caption_evidence = caption_evidence_for_mode(evidence, audio_evidence_mode)
    try:
        captions = _call_caption_with_fallback(
            task.task_id,
            task.styles,
            caption_evidence,
            config,
            client_factory,
            lambda provider, missing: build_final_caption_messages(
                Task(task.task_id, task.video_url, tuple(missing)),
                frames,
                caption_evidence,
                google=provider == DEFAULT_PROVIDER_GOOGLE,
                min_words=min_words,
                max_words=max_words,
                audio_evidence_mode=audio_evidence_mode,
            ),
            temperature=temperature,
            logger=active_logger,
            remaining_time_fn=remaining_time_fn,
            minimum_remaining_seconds=FINAL_SYNTHESIS_MIN_SECONDS,
            operation="caption",
            route="caption",
            audio=audio,
            audio_selected=audio_selected,
            min_words=min_words,
            max_words=max_words,
        )
    except RoutedRuntimeBudgetError:
        captions = build_evidence_based_captions(task.styles, caption_evidence)
        return _return_with_outcome(captions, outcome_callback, GENERATION_OUTCOME_EVIDENCE_FALLBACK)
    except Exception:
        captions = build_evidence_based_captions(task.styles, caption_evidence)
        return _return_with_outcome(captions, outcome_callback, GENERATION_OUTCOME_EVIDENCE_FALLBACK)

    model_generated_styles = set(captions)
    missing_styles = [style for style in task.styles if style not in captions]
    if focused_repair and missing_styles and remaining_time_fn() >= FOCUSED_REPAIR_MIN_SECONDS:
        try:
            repaired = _call_caption_with_fallback(
                task.task_id,
                missing_styles,
                caption_evidence,
                config,
                client_factory,
                lambda provider, missing: build_focused_repair_messages(
                    task, missing, captions, frames, caption_evidence,
                    google=provider == DEFAULT_PROVIDER_GOOGLE,
                    min_words=min_words,
                    max_words=max_words,
                    audio_evidence_mode=audio_evidence_mode,
                ),
                temperature=repair_temperature,
                logger=active_logger,
                remaining_time_fn=remaining_time_fn,
                minimum_remaining_seconds=FOCUSED_REPAIR_MIN_SECONDS,
                operation="focused_repair",
                route="caption",
                audio=audio,
                audio_selected=audio_selected,
                min_words=min_words,
                max_words=max_words,
            )
            captions.update(repaired)
            model_generated_styles.update(repaired)
        except Exception:
            pass
    still_missing = [style for style in task.styles if style not in captions]
    if still_missing:
        captions.update(build_evidence_based_captions(still_missing, caption_evidence))
    normalized = {style: cleanup_caption(captions[style], style, dict(caption_evidence), max_words=max_words) for style in task.styles}
    outcome = GENERATION_OUTCOME_MODEL if model_generated_styles else GENERATION_OUTCOME_EVIDENCE_FALLBACK
    return _return_with_outcome(normalized, outcome_callback, outcome)


def generate_routed_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    video_path: str | Path,
    *,
    config: RoutedGemmaConfig,
    env: Mapping[str, str],
    client_factory: Callable[[Any], Any],
    remaining_time_fn: Callable[[], float],
    debug_dir: str | Path | None = None,
    logger: logging.Logger | None = None,
    stage_callback: Callable[[str], None] | None = None,
    outcome_callback: Callable[[GenerationOutcome], None] | None = None,
    evidence_execution_callback: Callable[[EvidenceExecution], None] | None = None,
) -> dict[str, str]:
    from gemmaclip.captioner import build_evidence_based_captions, build_fallback_captions, cleanup_caption

    active_logger = logger or LOGGER
    if not config.has_credentials:
        active_logger.warning("task_id=%s operation=config route=fallback provider=routed_gemma model=none attempt=0 status=missing_credentials elapsed_seconds=0 remaining_seconds=%.3f minimum_remaining_seconds=0 fallback_used=true audio_available=false audio_selected=false audio_window_seconds=0", task.task_id, remaining_time_fn())
        return _return_with_outcome(build_fallback_captions(task.styles), outcome_callback, GENERATION_OUTCOME_DETERMINISTIC_FALLBACK)
    selected_frames = list(frames)
    if len(selected_frames) != 6:
        active_logger.warning("task_id=%s operation=frames route=fallback provider=routed_gemma model=none attempt=0 status=invalid_output elapsed_seconds=0 remaining_seconds=%.3f minimum_remaining_seconds=0 fallback_used=true audio_available=false audio_selected=false audio_window_seconds=0", task.task_id, remaining_time_fn())
        return _return_with_outcome(build_fallback_captions(task.styles), outcome_callback, GENERATION_OUTCOME_DETERMINISTIC_FALLBACK)

    stage_settings = load_routed_stage_settings(env)
    remaining = remaining_time_fn()
    if remaining < SINGLE_CALL_MIN_SECONDS:
        _log_runtime_stage_skip(
            active_logger, task.task_id, "single_call", "visual", remaining,
            SINGLE_CALL_MIN_SECONDS, unavailable_audio(16_000, "runtime unsafe"), False,
        )
        return _return_with_outcome(build_fallback_captions(task.styles), outcome_callback, GENERATION_OUTCOME_DETERMINISTIC_FALLBACK)
    if remaining < TWO_CALL_VISUAL_MIN_SECONDS:
        _notify_stage(stage_callback, "writing_captions")
        return _single_visual_caption_call(
            task, selected_frames, config, client_factory, remaining_time_fn,
            active_logger, temperature=stage_settings.single_call_temperature, outcome_callback=outcome_callback,
        )

    settings = load_audio_settings(env)
    _notify_stage(stage_callback, "checking_audio")
    remaining_before_audio = remaining_time_fn()
    if settings.mode != "off" and remaining_before_audio >= settings.min_remaining_seconds:
        audio = prepare_audio_candidate(video_path, Path(video_path).parent.parent / "audio" / safe_task_id(task.task_id), settings=settings)
    elif settings.mode == "off":
        audio = unavailable_audio(settings.sample_rate, "audio mode is off")
    else:
        audio = unavailable_audio(
            settings.sample_rate,
            "audio preprocessing skipped because runtime is below threshold",
        )
    remaining_after_audio = remaining_time_fn()
    if remaining_after_audio < SINGLE_CALL_MIN_SECONDS:
        _log_runtime_stage_skip(
            active_logger, task.task_id, "single_call", "visual",
            remaining_after_audio, SINGLE_CALL_MIN_SECONDS, audio, False,
        )
        cleanup_audio_candidate(audio)
        return _return_with_outcome(build_fallback_captions(task.styles), outcome_callback, GENERATION_OUTCOME_DETERMINISTIC_FALLBACK)
    if remaining_after_audio < TWO_CALL_VISUAL_MIN_SECONDS:
        cleanup_audio_candidate(audio)
        _notify_stage(stage_callback, "writing_captions")
        return _single_visual_caption_call(
            task, selected_frames, config, client_factory, remaining_time_fn,
            active_logger, temperature=stage_settings.single_call_temperature, outcome_callback=outcome_callback,
        )
    decision = decide_evidence_route(settings, audio, remaining_after_audio)
    _log_route(active_logger, task.task_id, decision, audio, remaining_after_audio, settings.min_remaining_seconds)

    try:
        _notify_stage(stage_callback, "building_evidence")
        evidence, execution = _generate_validated_evidence(
            task.task_id, selected_frames, audio, decision, config, client_factory,
            temperature=stage_settings.evidence_temperature,
            logger=active_logger,
            remaining_time_fn=remaining_time_fn,
        )
        _notify_evidence_execution(evidence_execution_callback, execution)
        if debug_dir is not None:
            _write_debug(debug_dir, task.task_id, "evidence", evidence)
    except RoutedRuntimeBudgetError:
        return _return_with_outcome(build_fallback_captions(task.styles), outcome_callback, GENERATION_OUTCOME_DETERMINISTIC_FALLBACK)
    except Exception as exc:
        status = "invalid_output" if isinstance(exc, ValueError) else "failed"
        active_logger.warning("task_id=%s operation=evidence route=%s provider=routed_gemma model=role_configured attempt=0 status=%s elapsed_seconds=0 remaining_seconds=%.3f minimum_remaining_seconds=%.3f fallback_used=true audio_available=%s audio_selected=%s audio_window_seconds=%.3f error=%s", task.task_id, decision.route, status, remaining_time_fn(), EVIDENCE_ATTEMPT_MIN_SECONDS, str(audio.available).lower(), str(decision.use_audio).lower(), audio.duration_seconds, type(exc).__name__)
        return _return_with_outcome(build_fallback_captions(task.styles), outcome_callback, GENERATION_OUTCOME_DETERMINISTIC_FALLBACK)
    finally:
        cleanup_audio_candidate(audio)

    remaining_before_final = remaining_time_fn()
    if remaining_before_final < FINAL_SYNTHESIS_MIN_SECONDS:
        _log_runtime_stage_skip(
            active_logger, task.task_id, "caption", "caption",
            remaining_before_final, FINAL_SYNTHESIS_MIN_SECONDS, audio, decision.use_audio,
        )
        return _return_with_outcome(build_evidence_based_captions(task.styles, evidence), outcome_callback, GENERATION_OUTCOME_EVIDENCE_FALLBACK)

    try:
        _notify_stage(stage_callback, "writing_captions")
        captions = _call_caption_with_fallback(
            task.task_id, task.styles, evidence, config, client_factory,
            lambda provider, missing: build_final_caption_messages(
                Task(task.task_id, task.video_url, tuple(missing)), selected_frames, evidence,
                google=provider == DEFAULT_PROVIDER_GOOGLE,
            ),
            temperature=stage_settings.caption_temperature,
            logger=active_logger,
            remaining_time_fn=remaining_time_fn,
            minimum_remaining_seconds=FINAL_SYNTHESIS_MIN_SECONDS,
            operation="caption",
            route="caption",
            audio=audio,
            audio_selected=execution.audio_used,
        )
    except RoutedRuntimeBudgetError:
        return _return_with_outcome(build_evidence_based_captions(task.styles, evidence), outcome_callback, GENERATION_OUTCOME_EVIDENCE_FALLBACK)
    except Exception as exc:
        active_logger.warning("task_id=%s operation=caption route=caption provider=routed_gemma model=role_configured attempt=0 status=failed elapsed_seconds=0 remaining_seconds=%.3f minimum_remaining_seconds=%.3f fallback_used=true audio_available=%s audio_selected=%s audio_window_seconds=%.3f error=%s", task.task_id, remaining_time_fn(), FINAL_SYNTHESIS_MIN_SECONDS, str(audio.available).lower(), str(decision.use_audio).lower(), audio.duration_seconds, type(exc).__name__)
        return _return_with_outcome(build_evidence_based_captions(task.styles, evidence), outcome_callback, GENERATION_OUTCOME_EVIDENCE_FALLBACK)

    model_generated_styles = set(captions)
    missing_styles = [style for style in task.styles if style not in captions]
    if missing_styles:
        _log_invalid_output(
            active_logger, task.task_id, "caption", remaining_time_fn(),
            FINAL_SYNTHESIS_MIN_SECONDS, audio, decision.use_audio,
        )
        remaining_before_repair = remaining_time_fn()
        if remaining_before_repair >= FOCUSED_REPAIR_MIN_SECONDS:
            try:
                repaired_captions = _call_caption_with_fallback(
                    task.task_id, missing_styles, evidence, config, client_factory,
                    lambda provider, missing: build_focused_repair_messages(
                        task, missing, captions, selected_frames, evidence,
                        google=provider == DEFAULT_PROVIDER_GOOGLE,
                    ),
                    temperature=stage_settings.repair_temperature,
                    logger=active_logger,
                    remaining_time_fn=remaining_time_fn,
                    minimum_remaining_seconds=FOCUSED_REPAIR_MIN_SECONDS,
                    operation="focused_repair",
                    route="caption",
                    audio=audio,
                    audio_selected=execution.audio_used,
                )
                captions.update(repaired_captions)
                model_generated_styles.update(repaired_captions)
                if any(style not in repaired_captions for style in missing_styles):
                    _log_invalid_output(
                        active_logger, task.task_id, "focused_repair", remaining_time_fn(),
                        FOCUSED_REPAIR_MIN_SECONDS, audio, decision.use_audio,
                    )
            except Exception:
                pass
        else:
            _log_runtime_stage_skip(
                active_logger, task.task_id, "focused_repair", "caption",
                remaining_before_repair, FOCUSED_REPAIR_MIN_SECONDS, audio, decision.use_audio,
            )
        still_missing = [style for style in task.styles if style not in captions]
        if still_missing:
            captions.update(build_evidence_based_captions(still_missing, evidence))

    captions = {style: cleanup_caption(captions[style], style, evidence) for style in task.styles}
    if debug_dir is not None:
        _write_debug(debug_dir, task.task_id, "captions", captions)
    outcome = GENERATION_OUTCOME_MODEL if model_generated_styles else GENERATION_OUTCOME_EVIDENCE_FALLBACK
    return _return_with_outcome(captions, outcome_callback, outcome)


def build_evidence_messages(task_id: str, frames: Sequence[ExtractedFrame], audio: AudioEvidenceCandidate, decision: RouteDecision, *, google: bool) -> list[dict[str, Any]]:
    content = [_image_part(frame.path, google=google) for frame in frames]
    content.append({"type": "text", "text": build_evidence_prompt(task_id, frames, audio, decision)})
    if decision.use_audio and audio.path is not None:
        if google:
            content.append({"type": "audio_file", "path": str(audio.path), "mime_type": "audio/wav"})
        else:
            content.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(audio.path.read_bytes()).decode("ascii"), "format": "wav"}})
    return [{"role": "system", "content": "Return JSON only. Report visible or audible evidence, never hidden reasoning."}, {"role": "user", "content": content}]


def build_final_caption_messages(
    task: Task,
    frames: Sequence[ExtractedFrame],
    evidence: Mapping[str, Any],
    *,
    google: bool,
    min_words: int = 18,
    max_words: int = 35,
    audio_evidence_mode: str = "use-if-present",
) -> list[dict[str, Any]]:
    content = [_image_part(frame.path, google=google) for frame in frames]
    content.append({"type": "text", "text": build_final_caption_prompt(task, frames, evidence, min_words=min_words, max_words=max_words, audio_evidence_mode=audio_evidence_mode)})
    return [{"role": "system", "content": final_caption_system_prompt()}, {"role": "user", "content": content}]


def build_focused_repair_messages(
    task: Task,
    missing_styles: Sequence[str],
    valid_captions: Mapping[str, str],
    frames: Sequence[ExtractedFrame],
    evidence: Mapping[str, Any],
    *,
    google: bool,
    min_words: int = 18,
    max_words: int = 35,
    audio_evidence_mode: str = "use-if-present",
) -> list[dict[str, Any]]:
    content = [_image_part(frame.path, google=google) for frame in frames]
    schema = {style: f"<{min_words}-{max_words} word caption>" for style in missing_styles}
    content.append({"type": "text", "text": (
        f"Task ID: {task.task_id}\nOnly repair these missing or invalid styles: {', '.join(missing_styles)}. "
        "Do not return or rewrite valid styles. Follow all grounding and audio rules from the system instruction.\n"
        f"Retained captions: {json.dumps(dict(valid_captions))}\nEvidence: {json.dumps(evidence, separators=(',', ':'))}\n"
        f"Each caption must contain {min_words}-{max_words} words. Audio policy: {audio_evidence_mode}. "
        f"Return only this exact JSON shape: {json.dumps(schema)}"
    )})
    return [{"role": "system", "content": final_caption_system_prompt()}, {"role": "user", "content": content}]


def build_evidence_prompt(task_id: str, frames: Sequence[ExtractedFrame], audio: AudioEvidenceCandidate, decision: RouteDecision) -> str:
    timestamps = ", ".join(f"{frame.timestamp_seconds:.3f}" for frame in frames)
    audio_instruction = (
        f"A selected audio window follows: start={audio.start_seconds:.3f}s duration={audio.duration_seconds:.3f}s. Distinguish directly audible speech, uncertain speech, non-speech audio, and unsupported interpretations."
        if decision.use_audio else
        "No audio was provided. Set audio.available=false and audio.status=unavailable. Never infer sound, speech, dialogue, music, or noise from images."
    )
    return (
        f"Task ID: {task_id}\nSix chronological frame timestamps: {timestamps}\n{audio_instruction}\n"
        "Return only one JSON object matching the supplied schema. Report only visible or audible evidence. Do not invent speaker identity, relationships, motives, occupation, destination, or events outside the sampled frames and selected audio window. Put uncertain transcript content in an uncertain state. Only place audio facts safe for final captions in audio.allowed_caption_facts. Set audio.visual_consistency to consistent, contradictory, or unknown. Do not provide chain-of-thought or detailed reasoning.\n"
        f"Schema: {json.dumps(empty_evidence(), separators=(',', ':'))}"
    )


def final_caption_system_prompt() -> str:
    from gemmaclip.prompts import build_fireworks_judge_generation_system_prompt
    humor_prompt = build_fireworks_judge_generation_system_prompt().replace(
        "Use only the six separate chronological frames provided.",
        "Use only the six separate chronological frames and structured evidence provided.",
    )
    return humor_prompt + (
        "\n\nAudio evidence rule: use an audio fact only when audio.status is usable, it is listed in "
        "audio.allowed_caption_facts, and it agrees with the visible evidence. Otherwise do not mention or quote "
        "speech, music, noise, sound, or speaker intent."
    )


def build_final_caption_prompt(
    task: Task,
    frames: Sequence[ExtractedFrame],
    evidence: Mapping[str, Any],
    *,
    min_words: int = 18,
    max_words: int = 35,
    audio_evidence_mode: str = "use-if-present",
) -> str:
    schema = {style: f"<{min_words}-{max_words} word caption>" for style in task.styles}
    audio_rule = {
        "ignore": "Ignore all audio evidence and do not mention sound, speech, music, noise, or dialogue.",
        "require": "Use audio only when audio.status is usable and caption-safe facts are present; otherwise this request should not be generated.",
        "use-if-present": "Use audio-derived facts only when audio.status is usable, the fact appears in audio.allowed_caption_facts, and it is visually consistent.",
    }.get(audio_evidence_mode, "Use only grounded visual evidence.")
    return (
        f"Task ID: {task.task_id}\nRequested styles: {', '.join(task.styles)}\n"
        f"Six chronological timestamps: {', '.join(f'{frame.timestamp_seconds:.3f}' for frame in frames)}\n"
        f"Each caption must contain {min_words}-{max_words} words. Strict visual grounding is required.\n"
        f"{audio_rule}\nStructured evidence JSON: {json.dumps(evidence, separators=(',', ':'))}\n"
        "Do not use a transcript verbatim unless short, clearly audible, relevant, and explicitly allowed.\n"
        f"Return exactly this dynamic JSON object and no extra prose: {json.dumps(schema)}"
    )


def normalize_routed_evidence(payload: Mapping[str, Any], audio_candidate: AudioEvidenceCandidate, decision: RouteDecision) -> dict[str, Any]:
    default = empty_evidence()
    normalized: dict[str, Any] = {}
    for key, fallback in default.items():
        value = payload.get(key, fallback)
        if key == "audio":
            continue
        if isinstance(fallback, list):
            normalized[key] = [str(item).strip() for item in value if str(item).strip()] if isinstance(value, list) else list(fallback)
        elif isinstance(fallback, dict):
            source = value if isinstance(value, Mapping) else {}
            normalized[key] = {nested: str(source.get(nested, nested_default)).strip() for nested, nested_default in fallback.items()}
        else:
            normalized[key] = str(value).strip() if value is not None else ""
    source_audio = payload.get("audio") if isinstance(payload.get("audio"), Mapping) else {}
    status = str(source_audio.get("status", "unavailable")).strip().lower()
    if status not in VALID_AUDIO_STATUSES:
        status = "uncertain" if decision.use_audio else "unavailable"
    if not decision.use_audio:
        status = "unavailable"
    allowed = source_audio.get("allowed_caption_facts", [])
    visual_consistency = str(source_audio.get("visual_consistency", "unknown")).strip().lower()
    if visual_consistency not in {"consistent", "contradictory", "unknown"}:
        visual_consistency = "unknown"
    safe_audio_facts = (
        [str(item).strip() for item in allowed if str(item).strip()]
        if status == "usable" and visual_consistency != "contradictory" and isinstance(allowed, list)
        else []
    )
    normalized["audio"] = {
        "available": bool(decision.use_audio),
        "window_start_seconds": audio_candidate.start_seconds if decision.use_audio else 0.0,
        "window_duration_seconds": audio_candidate.duration_seconds if decision.use_audio else 0.0,
        "speech_present": bool(source_audio.get("speech_present", False)) if status == "usable" else False,
        "language": str(source_audio.get("language", "")).strip() if decision.use_audio else "",
        "transcript": str(source_audio.get("transcript", "")).strip() if decision.use_audio else "",
        "status": status,
        "visual_consistency": visual_consistency,
        "allowed_caption_facts": safe_audio_facts,
    }
    reported_unsupported = [
        item for item in normalized["unsupported_claim_types"]
        if item not in CONDITIONAL_AUDIO_CLAIMS
    ]
    unsupported = [*reported_unsupported, *ALWAYS_UNSUPPORTED_CLAIMS]
    if status != "usable" or visual_consistency == "contradictory":
        unsupported.extend(CONDITIONAL_AUDIO_CLAIMS)
    normalized["unsupported_claim_types"] = list(dict.fromkeys(unsupported))
    return normalized


def empty_evidence() -> dict[str, Any]:
    return {
        "scene": "", "main_subjects": [], "actions": [], "setting": "", "visible_objects": [],
        "mood": "", "camera_notes": "", "temporal_progression": "", "caption_focus": "",
        "verified_description": "", "possible_misreads_to_avoid": [],
        "unsupported_claim_types": [*ALWAYS_UNSUPPORTED_CLAIMS, *CONDITIONAL_AUDIO_CLAIMS],
        "audio": {"available": False, "window_start_seconds": 0, "window_duration_seconds": 0, "speech_present": False, "language": "", "transcript": "", "status": "unavailable", "visual_consistency": "unknown", "allowed_caption_facts": []},
        "style_hooks": {"sarcastic": "", "humorous_tech": "", "humorous_non_tech": ""},
    }


def _single_visual_caption_call(
    task, frames, config, client_factory, remaining_time_fn, logger, *, temperature: float,
    outcome_callback: Callable[[GenerationOutcome], None] | None = None,
):
    from gemmaclip.captioner import build_fallback_captions, normalize_captions
    evidence = empty_evidence()
    try:
        captions = _call_caption_with_fallback(
            task.task_id, task.styles, evidence, config, client_factory,
            lambda provider, missing: build_final_caption_messages(
                Task(task.task_id, task.video_url, tuple(missing)), frames, evidence,
                google=provider == DEFAULT_PROVIDER_GOOGLE,
            ),
            temperature=temperature,
            logger=logger,
            remaining_time_fn=remaining_time_fn,
            minimum_remaining_seconds=SINGLE_CALL_ATTEMPT_MIN_SECONDS,
            operation="single_call",
            route="visual",
            audio=unavailable_audio(16_000, "single-call visual path"),
            audio_selected=False,
        )
        captions = normalize_captions(captions, task.styles, evidence)
        return _return_with_outcome(captions, outcome_callback, GENERATION_OUTCOME_MODEL)
    except Exception:
        return _return_with_outcome(build_fallback_captions(task.styles), outcome_callback, GENERATION_OUTCOME_DETERMINISTIC_FALLBACK)


def _generate_validated_evidence(
    task_id: str,
    frames: Sequence[ExtractedFrame],
    audio: AudioEvidenceCandidate,
    decision: RouteDecision,
    config: RoutedGemmaConfig,
    client_factory: Callable[[Any], Any],
    *,
    temperature: float,
    max_tokens: int | None = None,
    logger: logging.Logger,
    remaining_time_fn: Callable[[], float],
) -> tuple[dict[str, Any], EvidenceExecution]:
    from gemmaclip.captioner import request_model_text

    visual_decision = RouteDecision(
        "visual", False,
        "audio was not supplied to the Google visual fallback" if decision.use_audio else decision.reason,
    )
    no_audio = unavailable_audio(audio.sample_rate, "audio omitted from visual fallback")
    attempts: list[tuple[Any, RouteDecision, AudioEvidenceCandidate, bool]] = []
    if decision.use_audio:
        attempts.extend(
            (model_config, decision, audio, False)
            for model_config in config.role_configs("audio_visual")
            if model_config.provider != DEFAULT_PROVIDER_GOOGLE
        )
        attempts.extend(
            (model_config, visual_decision, no_audio, True)
            for model_config in config.role_configs("visual")
            if model_config.provider == DEFAULT_PROVIDER_GOOGLE
        )
    else:
        attempts.extend(
            (model_config, decision, audio, model_config.provider == DEFAULT_PROVIDER_GOOGLE and bool(config.fireworks_api_key))
            for model_config in config.role_configs("visual")
        )

    failures: list[str] = []
    for attempt, (model_config, attempt_decision, attempt_audio, fallback_used) in enumerate(attempts, start=1):
        remaining_seconds = remaining_time_fn()
        if remaining_seconds < EVIDENCE_ATTEMPT_MIN_SECONDS:
            _log_remote_attempt(
                logger, task_id, "evidence", attempt_decision.route, model_config.provider,
                model_config.model, attempt, "skipped_runtime", 0.0, remaining_seconds,
                EVIDENCE_ATTEMPT_MIN_SECONDS, fallback_used, attempt_audio, attempt_decision.use_audio,
            )
            raise RoutedRuntimeBudgetError(
                f"Only {remaining_seconds:.1f}s remain; need {EVIDENCE_ATTEMPT_MIN_SECONDS:.1f}s before evidence."
            )
        started = time.monotonic()
        try:
            if decision.use_audio and model_config.provider == DEFAULT_PROVIDER_GOOGLE:
                cleanup_audio_candidate(audio)
            messages = build_evidence_messages(
                task_id, frames, attempt_audio, attempt_decision,
                google=model_config.provider == DEFAULT_PROVIDER_GOOGLE,
            )
            text = request_model_text(
                client_factory(model_config), messages,
                temperature=temperature, use_response_format=False,
                max_tokens=max_tokens,
            )
            evidence = normalize_routed_evidence(_extract_object(text), attempt_audio, attempt_decision)
            if not _has_useful_evidence(evidence):
                raise ValueError("Evidence did not contain grounded visual observations.")
            actual_fallback = fallback_used or attempt > 1 or (decision.use_audio and not config.fireworks_api_key)
            reason = None
            if actual_fallback and decision.use_audio:
                reason = "The Fireworks audio-visual model was unavailable, so GemmaClip continued with Google Gemma 4 31B using frames only."
            elif actual_fallback:
                reason = "The Fireworks visual model was unavailable, so GemmaClip continued with Google Gemma 4 31B using frames only."
            execution = EvidenceExecution(
                provider=model_config.provider,
                model=model_config.model,
                modality="audio_visual" if attempt_decision.use_audio else "visual",
                audio_attempted=decision.use_audio,
                audio_used=attempt_decision.use_audio,
                fallback_used=actual_fallback,
                fallback_reason=reason,
            )
            _log_remote_attempt(
                logger, task_id, "evidence", attempt_decision.route, model_config.provider,
                model_config.model, attempt, "success", time.monotonic() - started,
                remaining_seconds, EVIDENCE_ATTEMPT_MIN_SECONDS, actual_fallback,
                attempt_audio, attempt_decision.use_audio,
            )
            return evidence, execution
        except Exception as exc:
            failures.append(type(exc).__name__)
            _log_remote_attempt(
                logger, task_id, "evidence", attempt_decision.route, model_config.provider,
                model_config.model, attempt, "invalid_output" if isinstance(exc, ValueError) else "failed",
                time.monotonic() - started, remaining_seconds, EVIDENCE_ATTEMPT_MIN_SECONDS,
                fallback_used, attempt_audio, attempt_decision.use_audio,
            )
    raise RuntimeError(f"All allowed evidence providers failed: {', '.join(failures) or 'no credentials'}")


def _call_caption_with_fallback(
    task_id: str,
    styles: Sequence[str],
    evidence: Mapping[str, Any],
    config: RoutedGemmaConfig,
    client_factory: Callable[[Any], Any],
    message_builder: Callable[[str, Sequence[str]], Sequence[Mapping[str, Any]]],
    *,
    temperature: float,
    logger: logging.Logger,
    remaining_time_fn: Callable[[], float],
    minimum_remaining_seconds: float,
    operation: str,
    route: str,
    audio: AudioEvidenceCandidate,
    audio_selected: bool,
    min_words: int = 18,
    max_words: int = 35,
) -> dict[str, str]:
    from gemmaclip.captioner import request_model_text

    failures: list[str] = []
    collected: dict[str, str] = {}
    for attempt, model_config in enumerate(config.role_configs("caption"), start=1):
        missing = [style for style in styles if style not in collected]
        if not missing:
            break
        remaining_seconds = remaining_time_fn()
        if remaining_seconds < minimum_remaining_seconds:
            _log_remote_attempt(logger, task_id, operation, route, model_config.provider, model_config.model, attempt, "skipped_runtime", 0.0, remaining_seconds, minimum_remaining_seconds, attempt > 1, audio, audio_selected)
            if collected:
                break
            raise RoutedRuntimeBudgetError(f"Only {remaining_seconds:.1f}s remain; need {minimum_remaining_seconds:.1f}s before {operation}.")
        started = time.monotonic()
        try:
            text = request_model_text(client_factory(model_config), message_builder(model_config.provider, missing), temperature=temperature, use_response_format=False)
            captions = _extract_valid_partial_captions(text, missing, evidence, min_words=min_words, max_words=max_words)
            if not captions:
                raise ValueError("Model did not return any valid requested captions.")
            collected.update(captions)
            _log_remote_attempt(logger, task_id, operation, route, model_config.provider, model_config.model, attempt, "success", time.monotonic() - started, remaining_seconds, minimum_remaining_seconds, attempt > 1, audio, audio_selected)
        except Exception as exc:
            failures.append(type(exc).__name__)
            _log_remote_attempt(logger, task_id, operation, route, model_config.provider, model_config.model, attempt, "invalid_output" if isinstance(exc, ValueError) else "failed", time.monotonic() - started, remaining_seconds, minimum_remaining_seconds, attempt > 1, audio, audio_selected)
    if collected:
        return collected
    raise RuntimeError(f"All caption providers failed: {', '.join(failures) or 'no credentials'}")


def _has_useful_evidence(evidence: Mapping[str, Any]) -> bool:
    for key in ("scene", "main_subjects", "actions", "setting", "visible_objects", "camera_notes", "temporal_progression", "verified_description", "caption_focus"):
        value = evidence.get(key)
        if isinstance(value, list) and any(str(item).strip() for item in value):
            return True
        if not isinstance(value, (list, Mapping)) and str(value or "").strip():
            return True
    return False


def _call_role_with_fallback(
    task_id: str,
    role: str,
    config: RoutedGemmaConfig,
    client_factory: Callable[[Any], Any],
    message_builder: Callable[[str], Sequence[Mapping[str, Any]]],
    *,
    temperature: float,
    logger: logging.Logger,
    remaining_time_fn: Callable[[], float],
    minimum_remaining_seconds: float,
    operation: str | None = None,
    route: str | None = None,
    audio: AudioEvidenceCandidate | None = None,
    audio_selected: bool | None = None,
) -> str:
    errors = []
    operation_name = operation or role
    route_name = route or role
    audio_candidate = audio or unavailable_audio(16_000, "audio metadata unavailable")
    selected_audio = audio_candidate.path is not None if audio_selected is None else audio_selected
    for attempt, model_config in enumerate(config.role_configs(role), start=1):
        remaining_seconds = remaining_time_fn()
        if remaining_seconds < minimum_remaining_seconds:
            _log_remote_attempt(
                logger, task_id, operation_name, route_name, model_config.provider,
                model_config.model, attempt, "skipped_runtime", 0.0,
                remaining_seconds, minimum_remaining_seconds, attempt > 1, audio_candidate, selected_audio,
            )
            raise RoutedRuntimeBudgetError(
                f"Only {remaining_seconds:.1f}s remain; need {minimum_remaining_seconds:.1f}s before {operation_name}."
            )
        started = time.monotonic()
        try:
            from gemmaclip.captioner import request_model_text
            text = request_model_text(client_factory(model_config), message_builder(model_config.provider), temperature=temperature, use_response_format=False)
            _log_remote_attempt(
                logger, task_id, operation_name, route_name, model_config.provider,
                model_config.model, attempt, "success", time.monotonic() - started,
                remaining_seconds, minimum_remaining_seconds, attempt > 1, audio_candidate, selected_audio,
            )
            return text
        except Exception as exc:
            errors.append(type(exc).__name__)
            _log_remote_attempt(
                logger, task_id, operation_name, route_name, model_config.provider,
                model_config.model, attempt, "failed", time.monotonic() - started,
                remaining_seconds, minimum_remaining_seconds, attempt > 1, audio_candidate, selected_audio,
            )
    raise RuntimeError(f"All same-role Gemma providers failed: {', '.join(errors) or 'no credentials'}")


def _image_part(path: Path, *, google: bool) -> dict[str, Any]:
    if google:
        return {"type": "image_file", "path": str(path), "mime_type": "image/jpeg"}
    from gemmaclip.captioner import make_jpeg_data_url
    return {"type": "image_url", "image_url": {"url": make_jpeg_data_url(path)}}


def _extract_object(text: str) -> dict[str, Any]:
    from gemmaclip.gemma_client import extract_json_objects
    objects = extract_json_objects(text)
    if not objects:
        raise ValueError("Model did not return evidence JSON.")
    return objects[-1]


def _usable_caption_audio(audio: Mapping[str, Any]) -> bool:
    facts = audio.get("allowed_caption_facts")
    return (
        str(audio.get("status", "")).lower() == "usable"
        and isinstance(facts, list)
        and any(str(item).strip() for item in facts)
        and str(audio.get("visual_consistency", "unknown")).lower() != "contradictory"
    )


def caption_evidence_for_mode(evidence: Mapping[str, Any], mode: str) -> dict[str, Any]:
    if mode not in {"ignore", "use-if-present", "require"}:
        raise ValueError("Unsupported audio evidence policy.")
    copied = json.loads(json.dumps(dict(evidence)))
    if mode != "ignore":
        return copied
    audio = copied.get("audio")
    if not isinstance(audio, dict):
        audio = {}
        copied["audio"] = audio
    audio.update({
        "available": False,
        "status": "unavailable",
        "transcript": "",
        "allowed_caption_facts": [],
        "visual_consistency": "unknown",
    })
    unsupported = list(copied.get("unsupported_claim_types", []))
    copied["unsupported_claim_types"] = list(dict.fromkeys([*unsupported, *CONDITIONAL_AUDIO_CLAIMS]))
    return copied


def _extract_valid_partial_captions(
    text: str,
    styles: Sequence[str],
    evidence: Mapping[str, Any],
    *,
    min_words: int = 18,
    max_words: int = 35,
) -> dict[str, str]:
    from gemmaclip.captioner import cleanup_caption
    from gemmaclip.gemma_client import extract_json_objects
    objects = extract_json_objects(text)
    if not objects:
        return {}
    payload = objects[-1]
    result: dict[str, str] = {}
    for style in styles:
        value = payload.get(style)
        word_count = len(value.split()) if isinstance(value, str) else 0
        if not isinstance(value, str) or sum(character.isalpha() for character in value) < 5 or not min_words <= word_count <= max_words:
            continue
        result[style] = cleanup_caption(value.strip(), style, dict(evidence), max_words=max_words)
    return result


def _write_debug(debug_dir, task_id, suffix, payload):
    path = Path(debug_dir) / f"{safe_task_id(task_id)}_routed_{suffix}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_temperature(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if not math.isfinite(parsed):
        return default
    return min(2.0, max(0.0, parsed))


def _log_remote_attempt(
    logger,
    task_id,
    operation,
    route,
    provider,
    model,
    attempt,
    status,
    elapsed_seconds,
    remaining_seconds,
    minimum_remaining_seconds,
    fallback_used,
    audio,
    audio_selected,
):
    log_method = logger.info if status == "success" else logger.warning
    log_method(
        "task_id=%s operation=%s route=%s provider=%s model=%s attempt=%s status=%s "
        "elapsed_seconds=%.3f remaining_seconds=%.3f minimum_remaining_seconds=%.3f "
        "fallback_used=%s audio_available=%s audio_selected=%s audio_window_seconds=%.3f",
        task_id, operation, route, provider, model, attempt, status,
        elapsed_seconds, remaining_seconds, minimum_remaining_seconds,
        str(fallback_used).lower(), str(audio.available).lower(),
        str(audio_selected).lower(), audio.duration_seconds,
    )


def _log_runtime_stage_skip(logger, task_id, operation, route, remaining_seconds, minimum_remaining_seconds, audio, audio_selected):
    logger.warning(
        "task_id=%s operation=%s route=%s provider=routed_gemma model=role_configured attempt=0 "
        "status=skipped_runtime elapsed_seconds=0 remaining_seconds=%.3f minimum_remaining_seconds=%.3f "
        "fallback_used=true audio_available=%s audio_selected=%s audio_window_seconds=%.3f",
        task_id, operation, route, remaining_seconds, minimum_remaining_seconds,
        str(audio.available).lower(), str(audio_selected).lower(), audio.duration_seconds,
    )


def _log_invalid_output(logger, task_id, operation, remaining_seconds, minimum_remaining_seconds, audio, audio_selected):
    logger.warning(
        "task_id=%s operation=%s route=caption provider=routed_gemma model=role_configured attempt=0 "
        "status=invalid_output elapsed_seconds=0 remaining_seconds=%.3f minimum_remaining_seconds=%.3f "
        "fallback_used=false audio_available=%s audio_selected=%s audio_window_seconds=%.3f",
        task_id, operation, remaining_seconds, minimum_remaining_seconds,
        str(audio.available).lower(), str(audio_selected).lower(), audio.duration_seconds,
    )


def _log_route(logger, task_id, decision, audio, remaining_seconds, minimum_remaining_seconds):
    logger.info("task_id=%s operation=routing route=%s provider=routed_gemma model=role_configured attempt=0 status=success elapsed_seconds=0 remaining_seconds=%.3f minimum_remaining_seconds=%.3f fallback_used=false audio_available=%s audio_selected=%s audio_window_seconds=%.3f", task_id, decision.route, remaining_seconds, minimum_remaining_seconds, str(audio.available).lower(), str(decision.use_audio).lower(), audio.duration_seconds)


def _notify_stage(callback: Callable[[str], None] | None, stage: str) -> None:
    if callback is not None:
        callback(stage)


def _notify_evidence_execution(
    callback: Callable[[EvidenceExecution], None] | None,
    execution: EvidenceExecution,
) -> None:
    if callback is not None:
        try:
            callback(execution)
        except Exception as exc:
            LOGGER.warning("operation=evidence_execution_callback status=failed error=%s", type(exc).__name__)


def _return_with_outcome(
    captions: dict[str, str],
    callback: Callable[[GenerationOutcome], None] | None,
    outcome: GenerationOutcome,
) -> dict[str, str]:
    if callback is not None:
        try:
            callback(outcome)
        except Exception as exc:
            LOGGER.warning("operation=outcome_callback status=failed error=%s", type(exc).__name__)
    return captions
