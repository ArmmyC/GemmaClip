from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
import json
import logging
from pathlib import Path
import re
from typing import Any

from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import GemmaClient, GemmaConfig, load_gemma_config
from gemmaclip.io import Task, safe_task_id
from gemmaclip.prompts import (
    EVIDENCE_SCHEMA,
    build_caption_system_prompt,
    build_caption_user_prompt,
    build_evidence_system_prompt,
    build_evidence_user_prompt,
)

LOGGER = logging.getLogger("gemmaclip.captioner")
MAX_GEMMA_FRAMES = 12
MAX_CAPTION_WORDS = 25
SPECULATION_PATTERNS = (
    re.compile(r"\blikely\b"),
    re.compile(r"\bprobably\b"),
    re.compile(r"\bmaybe\b"),
    re.compile(r"\bappears to be\b"),
    re.compile(r"\bseems to be\b"),
)
TECH_CLAIM_PATTERNS = (
    re.compile(r"\bscript\b"),
    re.compile(r"\bscripts\b"),
    re.compile(r"\bcoding\b"),
    re.compile(r"\bcode\b"),
    re.compile(r"\bprogramming\b"),
    re.compile(r"\brunning scripts\b"),
)
TECH_EVIDENCE_PATTERNS = (
    re.compile(r"\bscript\b"),
    re.compile(r"\bscripts\b"),
    re.compile(r"\bcoding\b"),
    re.compile(r"\bcode\b"),
    re.compile(r"\bprogramming\b"),
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
    client_factory: Callable[[GemmaConfig], GemmaClient] = GemmaClient,
) -> dict[str, str]:
    active_logger = logger or LOGGER

    if dry_run:
        active_logger.info("Task %s running in dry-run mode; using placeholder captions.", task.task_id)
        return build_placeholder_captions(task.styles)

    config = load_gemma_config(env)
    if config is None:
        active_logger.warning(
            "Task %s missing API credentials or GEMMA_MODEL; using placeholder captions.",
            task.task_id,
        )
        return build_placeholder_captions(task.styles)

    try:
        selected_frames = select_gemma_frames(frames, max_frames=MAX_GEMMA_FRAMES)
        evidence_messages = build_evidence_messages(task.task_id, selected_frames)
        client = client_factory(config)
        evidence = normalize_evidence(
            client.chat_completion_json(
                evidence_messages,
                temperature=0.1,
            )
        )
        if debug_dir is not None:
            write_evidence_debug_file(task.task_id, selected_frames, evidence, debug_dir)
        caption_messages = build_caption_messages(task.task_id, task.styles, evidence)
        captions = normalize_captions(
            client.chat_completion_json(
                caption_messages,
                temperature=0.7,
            ),
            task.styles,
            evidence,
        )
        return captions
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


def validate_caption(caption: str, style: str, evidence: dict[str, Any]) -> None:
    lowered = caption.strip().lower()
    for pattern in SPECULATION_PATTERNS:
        if pattern.search(lowered):
            raise ValueError("Caption contains a banned speculation phrase.")

    if len(caption.split()) > MAX_CAPTION_WORDS:
        raise ValueError("Caption exceeds the maximum allowed word count.")

    if style == "humorous_tech" and _contains_unsupported_tech_claim(lowered, evidence):
        raise ValueError("humorous_tech caption contains an unsupported coding or scripting claim.")


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
