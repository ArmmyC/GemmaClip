from __future__ import annotations

import base64
import json
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import GemmaClient, GemmaConfig, GemmaModelConfig, extract_json_objects, load_gemma_config
from gemmaclip.io import Task, safe_task_id
from gemmaclip.prompts import (
    EVIDENCE_SCHEMA,
    build_caption_system_prompt,
    build_caption_user_prompt,
    build_evidence_system_prompt,
    build_evidence_user_prompt,
    build_verifier_system_prompt,
    build_verifier_user_prompt,
)

LOGGER = logging.getLogger("gemmaclip.captioner")
MAX_GEMMA_FRAMES = 12
MAX_CAPTION_WORDS = 25
BANNED_SPECULATION_PHRASES = (
    "likely",
    "probably",
    "maybe",
    "appears to be",
    "seems to be",
)
SPECULATION_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(phrase)}\b")
    for phrase in BANNED_SPECULATION_PHRASES
)
TECH_CLAIM_PATTERNS = (
    re.compile(r"\bscript\b"),
    re.compile(r"\bscripts\b"),
    re.compile(r"\bcoding\b"),
    re.compile(r"\bcode\b"),
    re.compile(r"\bdebugging\b"),
    re.compile(r"\bdebug\b"),
    re.compile(r"\bdeveloper\b"),
    re.compile(r"\bprogramming\b"),
    re.compile(r"\bsoftware development\b"),
    re.compile(r"\brunning scripts\b"),
)
TECH_EVIDENCE_PATTERNS = (
    re.compile(r"\bscript\b"),
    re.compile(r"\bscripts\b"),
    re.compile(r"\bcoding\b"),
    re.compile(r"\bcode\b"),
    re.compile(r"\bprogramming\b"),
    re.compile(r"\bdebugging\b"),
    re.compile(r"\bdeveloper\b"),
    re.compile(r"\bsoftware development\b"),
    re.compile(r"\bterminal\b"),
    re.compile(r"\bcommand line\b"),
)


def generate_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    *,
    dry_run: bool = False,
    debug_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    logger: logging.Logger | None = None,
    client_factory: Callable[[GemmaModelConfig], GemmaClient] = GemmaClient,
) -> dict[str, str]:
    active_logger = logger or LOGGER

    if dry_run:
        active_logger.info("Task %s running in dry-run mode; using placeholder captions.", task.task_id)
        return build_placeholder_captions(task.styles)

    values = env if env is not None else os.environ
    config = load_gemma_config(env)
    if config is None:
        active_logger.warning(
            "Task %s missing API credentials; using placeholder captions.",
            task.task_id,
        )
        return build_placeholder_captions(task.styles)

    try:
        selected_frames = select_gemma_frames(frames, max_frames=MAX_GEMMA_FRAMES)
        vision_client = client_factory(config.vision_model_config())
        text_client = client_factory(config.text_model_config())
        evidence = generate_evidence(task.task_id, selected_frames, vision_client)
        if debug_dir is not None:
            write_evidence_debug_file(task.task_id, selected_frames, evidence, debug_dir)
        caption_messages = build_caption_messages(task.task_id, task.styles, evidence)
        caption_text = request_model_text(text_client, caption_messages, temperature=0.7)
        captions = normalize_captions(
            extract_caption_json(caption_text, task.styles),
            task.styles,
            evidence,
        )
        if debug_dir is not None:
            write_captions_debug_file(task.task_id, captions, debug_dir, suffix="raw")
        final_captions = maybe_verify_captions(
            task,
            captions,
            evidence,
            text_client,
            values,
            debug_dir=debug_dir,
        )
        if debug_dir is not None:
            write_captions_debug_file(task.task_id, final_captions, debug_dir, suffix="verified")
        return final_captions
    except Exception as exc:
        active_logger.warning("Task %s failed during Gemma captioning, using fallback captions: %s", task.task_id, exc)
        return build_fallback_captions(task.styles)


def build_placeholder_captions(styles: Sequence[str]) -> dict[str, str]:
    templates = {
        "formal": (
            "A short video clip is available, and a fuller caption will be generated after the visual analysis stage is enabled."
        ),
        "sarcastic": (
            "A short video clip arrives, politely expecting interpretation before the interesting captioning logic has actually shown up."
        ),
        "humorous_tech": (
            "A short video clip is standing by while the pipeline waits for its real captioning upgrade instead of placeholder mode."
        ),
        "humorous_non_tech": (
            "A short video clip shows up, and the caption is still warming up like a comedian before the first joke."
        ),
    }
    return {style: templates[style] for style in styles}


def build_fallback_captions(styles: Sequence[str]) -> dict[str, str]:
    templates = {
        "formal": (
            "The video could not be fully processed, so this placeholder caption notes a short scene with visible activity."
        ),
        "sarcastic": (
            "The video resisted a full analysis, which is a very efficient way to remain slightly mysterious."
        ),
        "humorous_tech": (
            "The clip hit a processing snag, so the captioning stack returned a graceful fallback instead of a dramatic crash."
        ),
        "humorous_non_tech": (
            "The video kept some secrets, so this caption politely fills in while the details stay offstage."
        ),
    }
    return {style: templates[style] for style in styles}


def build_evidence_debug_payload(
    task_id: str,
    selected_frames: Sequence[ExtractedFrame],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "selected_frame_count": len(selected_frames),
        "selected_frames": [
            {
                "path": str(frame.path),
                "timestamp_seconds": frame.timestamp_seconds,
            }
            for frame in selected_frames
        ],
        "evidence": evidence,
    }


def write_evidence_debug_file(
    task_id: str,
    selected_frames: Sequence[ExtractedFrame],
    evidence: dict[str, Any],
    debug_dir: str | Path,
) -> Path:
    output_path = Path(debug_dir) / f"{safe_task_id(task_id)}_evidence.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_evidence_debug_payload(task_id, selected_frames, evidence), indent=2),
        encoding="utf-8",
    )
    return output_path


def write_captions_debug_file(
    task_id: str,
    captions: dict[str, str],
    debug_dir: str | Path,
    *,
    suffix: str,
) -> Path:
    output_path = Path(debug_dir) / f"{safe_task_id(task_id)}_captions_{suffix}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "captions": captions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def select_gemma_frames(
    frames: Sequence[ExtractedFrame],
    max_frames: int = MAX_GEMMA_FRAMES,
) -> list[ExtractedFrame]:
    ordered_frames = list(frames)
    if len(ordered_frames) <= max_frames:
        return ordered_frames

    last_index = len(ordered_frames) - 1
    selected: list[ExtractedFrame] = []
    for slot in range(max_frames):
        index = (slot * last_index) // (max_frames - 1)
        selected.append(ordered_frames[index])
    return selected


def build_evidence_messages(task_id: str, frames: Sequence[ExtractedFrame]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": build_evidence_user_prompt(task_id, frames)}]
    for frame in frames:
        content.append(
            {
                "type": "text",
                "text": f"Frame {frame.path.name} at timestamp_seconds={frame.timestamp_seconds:.3f}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": make_jpeg_data_url(frame.path),
                },
            }
        )

    return [
        {"role": "system", "content": build_evidence_system_prompt()},
        {"role": "user", "content": content},
    ]


def build_caption_messages(
    task_id: str,
    styles: Sequence[str],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_caption_system_prompt()},
        {"role": "user", "content": build_caption_user_prompt(task_id, styles, evidence)},
    ]


def build_verifier_messages(
    task_id: str,
    styles: Sequence[str],
    evidence: dict[str, Any],
    captions: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_verifier_system_prompt()},
        {"role": "user", "content": build_verifier_user_prompt(task_id, styles, evidence, captions)},
    ]


def make_jpeg_data_url(image_path: str | Path) -> str:
    payload = make_resized_jpeg_bytes(image_path)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def make_resized_jpeg_bytes(
    image_path: str | Path,
    max_side: int = 768,
    quality: int = 85,
) -> bytes:
    if max_side <= 0:
        raise ValueError("max_side must be positive.")

    Image = _load_pillow_image()
    output = BytesIO()
    with Image.open(image_path) as image:
        converted = image.convert("RGB")
        converted.thumbnail((max_side, max_side))
        converted.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def generate_evidence(
    task_id: str,
    frames: Sequence[ExtractedFrame],
    client: GemmaClient,
) -> dict[str, Any]:
    evidence_messages = build_evidence_messages(task_id, frames)
    last_text = ""
    last_error: Exception | None = None

    for _ in range(2):
        last_text = request_model_text(
            client,
            evidence_messages,
            temperature=0.1,
            use_response_format=False,
        )
        try:
            return extract_evidence_json(last_text)
        except ValueError as exc:
            last_error = exc

    fallback_evidence = build_plain_text_evidence(last_text)
    if fallback_evidence is not None:
        return fallback_evidence

    if last_error is not None:
        raise last_error
    raise ValueError("Evidence generation did not produce usable output.")


def request_model_text(
    client: GemmaClient,
    messages: Sequence[Mapping[str, Any]],
    *,
    temperature: float,
    use_response_format: bool | None = None,
) -> str:
    if hasattr(client, "chat_completion_text"):
        try:
            return client.chat_completion_text(
                messages,
                temperature,
                use_response_format=use_response_format,
            )
        except TypeError:
            return client.chat_completion_text(messages, temperature)

    if hasattr(client, "chat_completion_json"):
        try:
            payload = client.chat_completion_json(
                messages,
                temperature,
                use_response_format=use_response_format,
            )
        except TypeError:
            payload = client.chat_completion_json(messages, temperature)
        return json.dumps(payload)

    raise TypeError("Client does not support text or JSON chat completion methods.")


def normalize_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, default in EVIDENCE_SCHEMA.items():
        value = payload.get(key, default)
        if isinstance(default, list):
            if not isinstance(value, list):
                value = []
            normalized[key] = [str(item).strip() for item in value if str(item).strip()]
        else:
            normalized[key] = str(value).strip() if value is not None else ""
    return normalized


def extract_evidence_json(text: str) -> dict[str, Any]:
    valid_candidates: list[dict[str, Any]] = []
    for candidate in extract_json_objects(text):
        if not _is_evidence_candidate(candidate):
            continue
        normalized = normalize_evidence(candidate)
        if _has_useful_evidence(normalized):
            valid_candidates.append(normalized)
    if not valid_candidates:
        raise ValueError("Could not extract a useful evidence JSON object from the model response.")
    return valid_candidates[-1]


def build_plain_text_evidence(text: str) -> dict[str, Any] | None:
    cleaned = _clean_plain_text(text)
    if not cleaned or cleaned == "{}":
        return None

    evidence = normalize_evidence({"scene": cleaned})
    return evidence if _has_useful_evidence(evidence) else None


def normalize_captions(
    payload: dict[str, Any],
    styles: Sequence[str],
    evidence: dict[str, Any],
) -> dict[str, str]:
    captions: dict[str, str] = {}
    for style in styles:
        value = payload.get(style)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Model response did not include a valid caption for style {style}.")
        caption = value.strip()
        validate_caption(caption, style, evidence)
        captions[style] = caption
    return captions


def extract_caption_json(text: str, styles: Sequence[str]) -> dict[str, Any]:
    valid_candidates: list[dict[str, Any]] = []
    for candidate in extract_json_objects(text):
        if _is_caption_candidate(candidate, styles):
            valid_candidates.append(candidate)
    if not valid_candidates:
        raise ValueError("Could not extract a valid caption JSON object from the model response.")
    return valid_candidates[-1]


def maybe_verify_captions(
    task: Task,
    captions: dict[str, str],
    evidence: dict[str, Any],
    client: GemmaClient,
    env: Mapping[str, str],
    *,
    debug_dir: str | Path | None = None,
) -> dict[str, str]:
    if _verifier_disabled(env):
        return captions

    try:
        verifier_messages = build_verifier_messages(task.task_id, task.styles, evidence, captions)
        verifier_text = request_model_text(client, verifier_messages, temperature=0.2)
        verified_payload = extract_caption_json(verifier_text, task.styles)
        verified_captions = normalize_captions(verified_payload, task.styles, evidence)
    except Exception:
        return captions

    return verified_captions


def validate_caption(caption: str, style: str, evidence: dict[str, Any]) -> None:
    if not caption.strip():
        raise ValueError("Caption must not be empty.")

    lowered = caption.strip().lower()
    for pattern in SPECULATION_PATTERNS:
        if pattern.search(lowered):
            raise ValueError("Caption contains a banned speculation phrase.")

    if len(caption.split()) > MAX_CAPTION_WORDS:
        raise ValueError("Caption exceeds the maximum allowed word count.")

    if style == "humorous_tech" and _contains_unsupported_tech_claim(lowered, evidence):
        raise ValueError("humorous_tech caption contains an unsupported coding or scripting claim.")


def _is_caption_candidate(payload: dict[str, Any], styles: Sequence[str]) -> bool:
    for style in styles:
        value = payload.get(style)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def _is_evidence_candidate(payload: dict[str, Any]) -> bool:
    return any(key in EVIDENCE_SCHEMA for key in payload)


def _has_useful_evidence(evidence: Mapping[str, Any]) -> bool:
    for value in evidence.values():
        if isinstance(value, list):
            if any(str(item).strip() for item in value):
                return True
            continue
        if str(value).strip():
            return True
    return False


def _clean_plain_text(text: str) -> str:
    cleaned = text.replace("```json", "").replace("```", "")
    return " ".join(cleaned.split()).strip()


def _contains_unsupported_tech_claim(caption: str, evidence: dict[str, Any]) -> bool:
    if not any(pattern.search(caption) for pattern in TECH_CLAIM_PATTERNS):
        return False
    evidence_text = _flatten_evidence_text(evidence)
    return not any(pattern.search(evidence_text) for pattern in TECH_EVIDENCE_PATTERNS)


def _flatten_evidence_text(evidence: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for value in evidence.values():
        if isinstance(value, list):
            parts.extend(str(item).strip().lower() for item in value if str(item).strip())
        else:
            text = str(value).strip().lower()
            if text:
                parts.append(text)
    return " ".join(parts)


def _load_pillow_image():
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for Gemma image payload resizing.") from exc

    return Image


def _verifier_disabled(env: Mapping[str, str]) -> bool:
    return env.get("GEMMACLIP_DISABLE_VERIFIER", "").strip().lower() == "true"
