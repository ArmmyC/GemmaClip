from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from gemmaclip.frames import ExtractedFrame
from gemmaclip.video import VideoMetadata


FRONTEND_TO_BACKEND_STYLE = {
    "formal": "formal",
    "sarcastic": "sarcastic",
    "humorous-tech": "humorous_tech",
    "humorous-non-tech": "humorous_non_tech",
}
BACKEND_TO_FRONTEND_STYLE = {value: key for key, value in FRONTEND_TO_BACKEND_STYLE.items()}


def frontend_style_to_backend(style: str) -> str:
    try:
        return FRONTEND_TO_BACKEND_STYLE[style]
    except KeyError as exc:
        raise ValueError(f"Unsupported caption style: {style}") from exc


def backend_style_to_frontend(style: str) -> str:
    try:
        return BACKEND_TO_FRONTEND_STYLE[style]
    except KeyError as exc:
        raise ValueError(f"Unsupported backend caption style: {style}") from exc


def adapt_video_metadata(
    filename: str,
    metadata: VideoMetadata,
    *,
    size_bytes: int,
    has_audio_stream: bool,
) -> dict[str, Any]:
    return {
        "filename": filename,
        "durationSec": metadata.duration_seconds,
        "width": metadata.width or 0,
        "height": metadata.height or 0,
        "fps": metadata.fps or 0.0,
        "codec": metadata.codec or "unknown",
        "sizeBytes": size_bytes,
        "hasAudioStream": has_audio_stream,
    }


def adapt_frames(run_id: str, frames: Sequence[ExtractedFrame]) -> list[dict[str, Any]]:
    adapted: list[dict[str, Any]] = []
    for index, frame in enumerate(frames, start=1):
        frame_id = f"frame_{index:03d}.jpg"
        role = frame.frame_role
        reason = "high-change" if role == "dynamic" else "anchor" if role == "anchor" else "uniform"
        adapted.append(
            {
                "id": frame_id,
                "index": index,
                "timestampSec": frame.timestamp_seconds,
                "reason": reason,
                "changeScore": float(frame.change_score or 0.0),
                "included": True,
                "thumbnailUrl": f"/api/runs/{run_id}/frames/{frame_id}",
            }
        )
    return adapted


def adapt_evidence(evidence: Mapping[str, Any], *, selected_route: str, route_reason: str) -> dict[str, Any]:
    audio = evidence.get("audio") if isinstance(evidence.get("audio"), Mapping) else {}
    style_hooks = evidence.get("style_hooks")
    if isinstance(style_hooks, Mapping):
        adapted_hooks = [f"{key}: {value}" for key, value in style_hooks.items() if str(value).strip()]
    elif isinstance(style_hooks, list):
        adapted_hooks = [str(value) for value in style_hooks if str(value).strip()]
    else:
        adapted_hooks = []
    return {
        "selectedRoute": selected_route,
        "routeReason": route_reason,
        "scene": str(evidence.get("scene", "")),
        "subjects": _strings(evidence.get("main_subjects")),
        "actions": _strings(evidence.get("actions")),
        "setting": str(evidence.get("setting", "")),
        "visibleObjects": _strings(evidence.get("visible_objects")),
        "mood": str(evidence.get("mood", "")),
        "cameraNotes": str(evidence.get("camera_notes", "")),
        "temporalProgression": _strings_or_single(evidence.get("temporal_progression")),
        "verifiedDescription": str(evidence.get("verified_description", "")),
        "possibleMisreads": _strings(evidence.get("possible_misreads_to_avoid")),
        "unsupportedClaims": _strings(evidence.get("unsupported_claim_types")),
        "styleHooks": adapted_hooks,
        "audio": {
            "status": _audio_status(audio.get("status")),
            "speechPresent": bool(audio.get("speech_present", False)),
            "language": _nullable_string(audio.get("language")),
            "transcript": _nullable_string(audio.get("transcript")),
            "visualConsistency": _visual_consistency(audio.get("visual_consistency")),
            "captionSafeFacts": _strings(audio.get("allowed_caption_facts")),
        },
    }


def adapt_captions(captions: Mapping[str, str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for backend_style, text in captions.items():
        frontend_style = backend_style_to_frontend(backend_style)
        results.append(
            {
                "id": f"caption-{frontend_style}",
                "style": frontend_style,
                "text": text,
                "wordCount": len(text.split()),
                "charCount": len(text),
                "status": "valid",
                "evidenceUsed": {"visualScene": True, "visibleAction": True, "allowedAudioFact": False},
            }
        )
    return results


def selected_route_from_evidence(evidence: Mapping[str, Any]) -> tuple[str, str]:
    audio = evidence.get("audio") if isinstance(evidence.get("audio"), Mapping) else {}
    status = str(audio.get("status", "unavailable"))
    if bool(audio.get("available")) and status == "usable":
        return "gemma-4-12b-unified", "A usable audio window was selected and normalized by Gemma alongside the video frames."
    return "gemma-4-26b-a4b", "Visual evidence was used because no caption-safe audio evidence was available."


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _strings_or_single(value: Any) -> list[str]:
    if isinstance(value, list):
        return _strings(value)
    text = str(value or "").strip()
    return [text] if text else []


def _nullable_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _audio_status(value: Any) -> str:
    status = str(value or "unavailable").lower()
    return status if status in {"usable", "uncertain", "silent", "unavailable", "failed"} else "uncertain"


def _visual_consistency(value: Any) -> str:
    status = str(value or "unknown").lower()
    return status if status in {"consistent", "contradictory", "unknown"} else "unknown"
