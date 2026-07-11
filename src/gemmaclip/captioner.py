from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
import tempfile
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageChops, ImageOps

from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import (
    DEFAULT_PROVIDER_FIREWORKS,
    DEFAULT_PROVIDER_FIREWORKS_JUDGE,
    DEFAULT_PROVIDER_GOOGLE,
    DEFAULT_PROVIDER_OPENROUTER,
    FireworksRuntimeBudgetError,
    FireworksVisionRequestError,
    create_model_client,
    extract_json_objects,
    load_gemma_config,
    load_google_provider_config,
)
from gemmaclip.io import Task, safe_task_id
from gemmaclip.prompts import (
    EVIDENCE_SCHEMA,
    build_caption_repair_user_prompt,
    build_caption_system_prompt,
    build_caption_user_prompt,
    build_direct_caption_repair_user_prompt,
    build_direct_caption_system_prompt,
    build_direct_caption_user_prompt,
    build_evidence_system_prompt,
    build_evidence_user_prompt,
    build_google_visual_evidence_user_prompt,
    build_fireworks_judge_generation_system_prompt,
    build_fireworks_judge_generation_user_prompt,
    build_fireworks_judge_repair_system_prompt,
    build_fireworks_judge_repair_user_prompt,
    build_fireworks_judge_review_system_prompt,
    build_fireworks_judge_review_user_prompt,
    build_verifier_system_prompt,
    build_verifier_user_prompt,
)

LOGGER = logging.getLogger("gemmaclip.captioner")
MAX_GEMMA_FRAMES = 12
MAX_GOOGLE_EVIDENCE_FRAMES = 6
MAX_CAPTION_WORDS = 40
MAX_CAPTION_ATTEMPTS = 3
GOOGLE_CONTACT_SHEET_CELL_MAX_SIDE = 384
GOOGLE_DESCRIPTION_FIRST_MIN_REMAINING_SECONDS = 120.0
FIREWORKS_JUDGE_SKIP_REVIEW_REMAINING_SECONDS = 150.0
FIREWORKS_JUDGE_REQUEST_BUDGET_SECONDS = 45.0
FIREWORKS_JUDGE_FINAL_OUTPUT_BUFFER_SECONDS = 20.0
FIREWORKS_JUDGE_MIN_GENERATION_REMAINING_SECONDS = (
    FIREWORKS_JUDGE_REQUEST_BUDGET_SECONDS + FIREWORKS_JUDGE_FINAL_OUTPUT_BUFFER_SECONDS
)
_STYLE_KEY_ALIASES = {
    "formal": "formal",
    "sarcastic": "sarcastic",
    "humoroustech": "humorous_tech",
    "humorousnontech": "humorous_non_tech",
}


@dataclass(frozen=True, slots=True)
class FireworksCaptionExtraction:
    valid_captions: dict[str, str]
    missing_styles: list[str]
    invalid_styles: list[str]
BANNED_SPECULATION_PHRASES = (
    "likely",
    "probably",
    "maybe",
    "appears to be",
    "seems to be",
    "seem",
    "seems",
    "seeming",
    "seemingly",
    "as if",
    "hoping",
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
SOFT_TECH_REPLACEMENTS = (
    (re.compile(r"\bscripts\b", re.IGNORECASE), "processes"),
    (re.compile(r"\bscript\b", re.IGNORECASE), "process"),
    (re.compile(r"\bcoding\b", re.IGNORECASE), "computer work"),
    (re.compile(r"\bprogramming\b", re.IGNORECASE), "computer work"),
    (re.compile(r"\bsoftware development\b", re.IGNORECASE), "computer work"),
    (re.compile(r"\bcode\b", re.IGNORECASE), "computer work"),
    (re.compile(r"\bdebugging\b", re.IGNORECASE), "troubleshooting"),
    (re.compile(r"\bdebug\b", re.IGNORECASE), "troubleshoot"),
    (re.compile(r"\bdeveloper\b", re.IGNORECASE), "worker"),
)
SOFT_SPECULATION_REPLACEMENTS = (
    (re.compile(r"\bappears to be\b", re.IGNORECASE), "is"),
    (re.compile(r"\bseems to be\b", re.IGNORECASE), "is"),
    (re.compile(r"\bseemingly\b", re.IGNORECASE), ""),
    (re.compile(r"\bseeming\b", re.IGNORECASE), "looking"),
    (re.compile(r"\bseems\b", re.IGNORECASE), "looks"),
    (re.compile(r"\bseem\b", re.IGNORECASE), "look"),
    (re.compile(r"\blikely\b", re.IGNORECASE), ""),
    (re.compile(r"\bprobably\b", re.IGNORECASE), ""),
    (re.compile(r"\bmaybe\b", re.IGNORECASE), ""),
    (re.compile(r"\bas if\b", re.IGNORECASE), "while"),
    (re.compile(r"\bhoping\b", re.IGNORECASE), "waiting"),
)


def generate_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    *,
    dry_run: bool = False,
    debug_dir: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    logger: logging.Logger | None = None,
    client_factory: Callable[[Any], Any] = create_model_client,
    remaining_seconds: float | None = None,
    remaining_time_fn: Callable[[], float] | None = None,
) -> dict[str, str]:
    active_logger = logger or LOGGER

    if dry_run:
        active_logger.info("Task %s running in dry-run mode; using placeholder captions.", task.task_id)
        return build_placeholder_captions(task.styles)

    values = env if env is not None else os.environ
    if _force_placeholder(values):
        active_logger.info("Task %s running with forced placeholder mode.", task.task_id)
        return build_placeholder_captions(task.styles)
    if _force_fallback(values):
        active_logger.info("Task %s running with forced fallback mode.", task.task_id)
        return build_fallback_captions(task.styles)

    config = load_gemma_config(env)
    if config is None:
        active_logger.warning(
            "Task %s missing API credentials; using placeholder captions.",
            task.task_id,
        )
        return build_placeholder_captions(task.styles)

    if config.provider == DEFAULT_PROVIDER_GOOGLE:
        return _generate_google_fast_captions(
            task,
            frames,
            debug_dir=debug_dir,
            env=values,
            logger=active_logger,
            client_factory=client_factory,
            model_config=config.text_model_config(),
            remaining_seconds=remaining_seconds,
        )
    if config.provider == DEFAULT_PROVIDER_OPENROUTER:
        return _generate_openrouter_experiment_captions(
            task,
            frames,
            debug_dir=debug_dir,
            env=values,
            logger=active_logger,
            client_factory=client_factory,
            model_config=config.text_model_config(),
            google_fallback_config=load_google_provider_config(values),
            remaining_seconds=remaining_seconds,
        )
    if config.provider == DEFAULT_PROVIDER_FIREWORKS_JUDGE:
        fireworks_remaining_time_fn = _make_remaining_time_fn(remaining_seconds, remaining_time_fn)
        return _generate_fireworks_judge_captions(
            task,
            frames,
            debug_dir=debug_dir,
            logger=active_logger,
            client_factory=client_factory,
            model_config=config,
            remaining_time_fn=fireworks_remaining_time_fn,
        )

    try:
        selected_frames = select_gemma_frames(frames, max_frames=MAX_GEMMA_FRAMES)
        vision_client = client_factory(config.vision_model_config())
        text_client = client_factory(config.text_model_config())
        evidence = generate_evidence(task.task_id, selected_frames, vision_client)
        if debug_dir is not None:
            write_evidence_debug_file(task.task_id, selected_frames, evidence, debug_dir)
    except Exception as exc:
        active_logger.warning("Task %s failed during evidence generation, using fallback captions: %s", task.task_id, exc)
        return build_fallback_captions(task.styles)

    used_evidence_fallback = False
    try:
        captions = generate_style_captions(
            task.task_id,
            task.styles,
            evidence,
            text_client,
            debug_dir=debug_dir,
        )
    except Exception as exc:
        active_logger.warning(
            "Task %s failed during caption generation after evidence extraction, using evidence-based fallback captions: %s",
            task.task_id,
            exc,
        )
        captions = build_evidence_based_captions(task.styles, evidence)
        used_evidence_fallback = True

    if debug_dir is not None:
        write_captions_debug_file(task.task_id, captions, debug_dir, suffix="raw")

    final_captions = captions
    if not used_evidence_fallback:
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


def write_caption_attempt_debug_file(
    task_id: str,
    response_text: str,
    debug_dir: str | Path,
    *,
    attempt_number: int,
) -> Path:
    output_path = Path(debug_dir) / f"{safe_task_id(task_id)}_caption_attempt_{attempt_number}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response_text, encoding="utf-8")
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


def build_google_evidence_messages(task_id: str, frames: Sequence[ExtractedFrame]) -> list[dict[str, Any]]:
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
                "type": "image_file",
                "path": str(frame.path),
                "mime_type": "image/jpeg",
            }
        )

    return [
        {"role": "system", "content": build_evidence_system_prompt()},
        {"role": "user", "content": content},
    ]


def build_google_contact_sheet_evidence_messages(
    task_id: str,
    frames: Sequence[ExtractedFrame],
    contact_sheet_path: str | Path,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_evidence_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_google_visual_evidence_user_prompt(task_id, frames)},
                {
                    "type": "image_file",
                    "path": str(contact_sheet_path),
                    "mime_type": "image/jpeg",
                },
            ],
        },
    ]


def build_direct_caption_messages(
    task_id: str,
    styles: Sequence[str],
    frames: Sequence[ExtractedFrame],
    contact_sheet_path: str | Path,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_direct_caption_system_prompt()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_direct_caption_user_prompt(task_id, styles, frames)},
                {
                    "type": "image_file",
                    "path": str(contact_sheet_path),
                    "mime_type": "image/jpeg",
                },
            ],
        },
    ]


def build_fireworks_judge_generation_messages(
    task_id: str,
    styles: Sequence[str],
    frames: Sequence[ExtractedFrame],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_fireworks_judge_generation_user_prompt(task_id, styles, frames)}
    ]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": make_jpeg_data_url(frame.path)},
        }
        for frame in frames
    )
    return [
        {"role": "system", "content": build_fireworks_judge_generation_system_prompt()},
        {"role": "user", "content": content},
    ]


def build_fireworks_judge_repair_messages(
    task_id: str,
    styles: Sequence[str],
    valid_captions: dict[str, str],
    frames: Sequence[ExtractedFrame],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_fireworks_judge_repair_user_prompt(task_id, styles, valid_captions, frames),
        }
    ]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": make_jpeg_data_url(frame.path)},
        }
        for frame in frames
    )
    return [
        {"role": "system", "content": build_fireworks_judge_repair_system_prompt()},
        {"role": "user", "content": content},
    ]


def build_fireworks_judge_review_messages(
    task_id: str,
    styles: Sequence[str],
    frames: Sequence[ExtractedFrame],
    captions: dict[str, str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_fireworks_judge_review_user_prompt(task_id, styles, frames, captions)}
    ]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": make_jpeg_data_url(frame.path)},
        }
        for frame in frames
    )
    return [
        {"role": "system", "content": build_fireworks_judge_review_system_prompt()},
        {"role": "user", "content": content},
    ]


def build_direct_caption_repair_messages(
    task_id: str,
    styles: Sequence[str],
    previous_response: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_direct_caption_system_prompt()},
        {
            "role": "user",
            "content": build_direct_caption_repair_user_prompt(task_id, styles, previous_response),
        },
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


def build_caption_repair_messages(
    task_id: str,
    styles: Sequence[str],
    evidence: dict[str, Any],
    previous_response: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": build_caption_system_prompt()},
        {
            "role": "user",
            "content": build_caption_repair_user_prompt(task_id, styles, evidence, previous_response),
        },
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
    client: Any,
) -> dict[str, Any]:
    evidence_messages = (
        build_google_evidence_messages(task_id, frames)
        if _client_provider(client) == DEFAULT_PROVIDER_GOOGLE
        else build_evidence_messages(task_id, frames)
    )
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
    client: Any,
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


def generate_style_captions(
    task_id: str,
    styles: Sequence[str],
    evidence: dict[str, Any],
    client: Any,
    *,
    debug_dir: str | Path | None = None,
    max_attempts: int = MAX_CAPTION_ATTEMPTS,
) -> dict[str, str]:
    last_response = ""
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        messages = (
            build_caption_messages(task_id, styles, evidence)
            if attempt == 1
            else build_caption_repair_messages(task_id, styles, evidence, last_response)
        )
        caption_text = request_model_text(
            client,
            messages,
            temperature=0.7 if attempt == 1 else 0.2,
            use_response_format=False,
        )
        last_response = caption_text
        if debug_dir is not None:
            write_caption_attempt_debug_file(
                task_id,
                caption_text,
                debug_dir,
                attempt_number=attempt,
            )
        try:
            return normalize_captions(
                extract_caption_json(caption_text, styles),
                styles,
                evidence,
            )
        except ValueError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("Caption generation failed without a usable model response.")


def generate_google_direct_captions(
    task_id: str,
    styles: Sequence[str],
    frames: Sequence[ExtractedFrame],
    client: Any,
    *,
    debug_dir: str | Path | None = None,
) -> dict[str, str]:
    contact_sheet_path = create_google_contact_sheet(frames)
    last_error: Exception | None = None
    last_response = ""

    try:
        direct_messages = build_direct_caption_messages(task_id, styles, frames, contact_sheet_path)
        direct_text = request_model_text(
            client,
            direct_messages,
            temperature=0.4,
            use_response_format=False,
        )
        last_response = direct_text
        if debug_dir is not None:
            write_caption_attempt_debug_file(task_id, direct_text, debug_dir, attempt_number=1)
        try:
            return normalize_captions(extract_caption_json(direct_text, styles), styles, {})
        except ValueError as exc:
            last_error = exc

        repair_messages = build_direct_caption_repair_messages(task_id, styles, last_response)
        repair_text = request_model_text(
            client,
            repair_messages,
            temperature=0.1,
            use_response_format=False,
        )
        last_response = repair_text
        if debug_dir is not None:
            write_caption_attempt_debug_file(task_id, repair_text, debug_dir, attempt_number=2)
        return normalize_captions(extract_caption_json(repair_text, styles), styles, {})
    finally:
        Path(contact_sheet_path).unlink(missing_ok=True)


def generate_google_visual_evidence(
    task_id: str,
    frames: Sequence[ExtractedFrame],
    client: Any,
) -> dict[str, Any]:
    contact_sheet_path = create_google_contact_sheet(frames)
    try:
        messages = build_google_contact_sheet_evidence_messages(task_id, frames, contact_sheet_path)
        evidence_text = request_model_text(
            client,
            messages,
            temperature=0.1,
            use_response_format=False,
        )
        return extract_evidence_json(evidence_text)
    finally:
        Path(contact_sheet_path).unlink(missing_ok=True)


def create_google_contact_sheet(frames: Sequence[ExtractedFrame]) -> Path:
    if not frames:
        raise ValueError("Cannot create a Google contact sheet without frames.")

    output_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    output_path = Path(output_handle.name)
    output_handle.close()

    cells = [_load_contact_sheet_cell(frame.path) for frame in frames[:MAX_GOOGLE_EVIDENCE_FRAMES]]
    columns = 3 if len(cells) > 4 else 2
    rows = (len(cells) + columns - 1) // columns
    while len(cells) < rows * columns:
        cells.append(Image.new("RGB", (GOOGLE_CONTACT_SHEET_CELL_MAX_SIDE, GOOGLE_CONTACT_SHEET_CELL_MAX_SIDE), color="white"))

    cell_width = max(cell.width for cell in cells)
    cell_height = max(cell.height for cell in cells)
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), color="white")
    for index, cell in enumerate(cells):
        row = index // columns
        column = index % columns
        offset_x = column * cell_width + (cell_width - cell.width) // 2
        offset_y = row * cell_height + (cell_height - cell.height) // 2
        sheet.paste(cell, (offset_x, offset_y))

    sheet.save(output_path, format="JPEG", quality=85, optimize=True)
    return output_path


def build_visual_heuristic_captions(
    styles: Sequence[str],
    frames: Sequence[ExtractedFrame],
) -> dict[str, str] | None:
    try:
        brightness, tone, action_phrase = _frame_heuristic_summary(frames)
    except Exception:
        return None

    templates = {
        "formal": f"A {brightness} {tone} clip {action_phrase} across the sequence without a major shift in visual focus.",
        "sarcastic": f"A {brightness} {tone} clip {action_phrase}, providing exactly the measured excitement nobody urgently requested today.",
        "humorous_tech": f"A {brightness} {tone} clip {action_phrase}, like a calm buffer handling one more polite little update.",
        "humorous_non_tech": f"A {brightness} {tone} clip {action_phrase}, like the day quietly rehearsing one small neighborhood joke.",
    }
    return {style: templates[style] for style in styles}


def normalize_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, default in EVIDENCE_SCHEMA.items():
        value = payload.get(key, default)
        if isinstance(default, list):
            if not isinstance(value, list):
                value = []
            normalized[key] = [str(item).strip() for item in value if str(item).strip()]
        elif isinstance(default, dict):
            normalized[key] = _normalize_nested_evidence_object(value, default)
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
    captions = validate_caption_structure(payload, styles)
    return {
        style: cleanup_caption(caption, style, evidence)
        for style, caption in captions.items()
    }


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
    client: Any,
    env: Mapping[str, str],
    *,
    debug_dir: str | Path | None = None,
) -> dict[str, str]:
    if _verifier_disabled(env):
        return captions

    try:
        verifier_messages = build_verifier_messages(task.task_id, task.styles, evidence, captions)
        verifier_text = request_model_text(
            client,
            verifier_messages,
            temperature=0.2,
            use_response_format=False,
        )
        verified_payload = extract_caption_json(verifier_text, task.styles)
        verified_captions = normalize_captions(verified_payload, task.styles, evidence)
    except Exception:
        return captions

    return verified_captions


def validate_caption_structure(
    payload: Mapping[str, Any],
    styles: Sequence[str],
) -> dict[str, str]:
    captions: dict[str, str] = {}
    for style in styles:
        value = payload.get(style)
        if not isinstance(value, str):
            raise ValueError(f"Model response did not include a valid caption for style {style}.")
        caption = value.strip()
        if not caption or _is_invalid_caption_text(caption):
            raise ValueError(f"Model response did not include a valid caption for style {style}.")
        captions[style] = caption
    return captions


def build_evidence_based_captions(
    styles: Sequence[str],
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    subject_text = _evidence_subject_phrase(evidence)
    action_text = _evidence_action_phrase(evidence)
    setting_text = _evidence_setting_phrase(evidence)
    object_text = _evidence_object_phrase(evidence)

    templates = {
        "formal": (
            f"The clip captures {subject_text} {action_text} in {setting_text}, with {object_text} visible nearby throughout."
        ),
        "sarcastic": (
            f"The clip captures {subject_text} {action_text} in {setting_text}, delivering exactly the drama this task was clearly craving."
        ),
        "humorous_tech": (
            f"The clip captures {subject_text} {action_text} in {setting_text}, like a calm CPU handling one more background task."
        ),
        "humorous_non_tech": (
            f"The clip captures {subject_text} {action_text} in {setting_text}, like the day quietly rehearsed one small joke."
        ),
    }
    return {style: templates[style] for style in styles}


def cleanup_caption(caption: str, style: str, evidence: dict[str, Any]) -> str:
    cleaned = caption.strip()
    cleaned = _soften_speculation_phrases(cleaned)
    if style == "humorous_tech" and _contains_unsupported_tech_claim(cleaned.lower(), evidence):
        cleaned = _soften_unsupported_tech_claims(cleaned)
    cleaned = _normalize_spacing(cleaned)
    cleaned = _trim_caption_to_max_words(cleaned, max_words=MAX_CAPTION_WORDS)
    return cleaned or caption.strip()


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
        if isinstance(value, Mapping):
            if _has_useful_evidence(value):
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


def _soften_speculation_phrases(caption: str) -> str:
    softened = caption
    for pattern, replacement in SOFT_SPECULATION_REPLACEMENTS:
        softened = pattern.sub(replacement, softened)
    return softened


def _soften_unsupported_tech_claims(caption: str) -> str:
    softened = caption
    for pattern, replacement in SOFT_TECH_REPLACEMENTS:
        softened = pattern.sub(replacement, softened)
    return softened


def _normalize_spacing(caption: str) -> str:
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", caption)
    normalized = re.sub(r"([,.;:!?])\s*([,.;:!?])+", r"\1", normalized)
    normalized = re.sub(r"\(\s+", "(", normalized)
    normalized = re.sub(r"\s+\)", ")", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized.strip()


def _trim_caption_to_max_words(caption: str, *, max_words: int) -> str:
    words = caption.split()
    if len(words) <= max_words:
        return caption
    return " ".join(words[:max_words]).strip()


def _flatten_evidence_text(evidence: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for value in evidence.values():
        if isinstance(value, list):
            parts.extend(str(item).strip().lower() for item in value if str(item).strip())
        elif isinstance(value, Mapping):
            parts.append(_flatten_evidence_text(value))
        else:
            text = str(value).strip().lower()
            if text:
                parts.append(text)
    return " ".join(parts)


def _is_invalid_caption_text(caption: str) -> bool:
    normalized = caption.strip().lower()
    condensed = re.sub(r"[\s_]+", "", normalized)
    if condensed in {"", ".", "-", "--", "...", "n/a", "na", "caption"}:
        return True
    if re.fullmatch(r"[\W_]+", caption):
        return True
    return sum(1 for char in caption if char.isalpha()) < 5


def _evidence_subject_phrase(evidence: Mapping[str, Any]) -> str:
    subjects = evidence.get("main_subjects")
    return _list_phrase(subjects, fallback="the main subject")


def _evidence_action_phrase(evidence: Mapping[str, Any]) -> str:
    actions = evidence.get("actions")
    return _list_phrase(actions, fallback="moving through the scene")


def _evidence_setting_phrase(evidence: Mapping[str, Any]) -> str:
    setting = str(evidence.get("setting", "")).strip()
    if setting:
        return setting
    scene = str(evidence.get("scene", "")).strip()
    return scene or "the visible scene"


def _evidence_object_phrase(evidence: Mapping[str, Any]) -> str:
    objects = evidence.get("visible_objects")
    return _list_phrase(objects, fallback="other scene details")


def _list_phrase(value: Any, *, fallback: str) -> str:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        if not cleaned:
            return fallback
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"
    cleaned_value = str(value).strip()
    return cleaned_value or fallback


def _normalize_nested_evidence_object(value: Any, default: Mapping[str, Any]) -> dict[str, Any]:
    raw_mapping = value if isinstance(value, Mapping) else {}
    normalized: dict[str, Any] = {}
    for nested_key, nested_default in default.items():
        nested_value = raw_mapping.get(nested_key, nested_default)
        normalized[nested_key] = str(nested_value).strip() if nested_value is not None else ""
    return normalized


def select_google_evidence_frames(
    frames: Sequence[ExtractedFrame],
    max_frames: int = MAX_GOOGLE_EVIDENCE_FRAMES,
) -> list[ExtractedFrame]:
    ordered_frames = list(frames)
    if len(ordered_frames) <= max_frames:
        return ordered_frames

    last_index = len(ordered_frames) - 1
    ratios = (0.05, 0.20, 0.35, 0.55, 0.75, 0.95)
    selected_indices = sorted(
        {
            min(last_index, max(0, round(last_index * ratio)))
            for ratio in ratios[:max_frames]
        }
    )
    return [ordered_frames[index] for index in selected_indices]


def select_fireworks_judge_frames(frames: Sequence[ExtractedFrame]) -> list[ExtractedFrame]:
    ordered_frames = list(frames)
    if len(ordered_frames) <= 6:
        return ordered_frames

    last_index = len(ordered_frames) - 1
    ratios = (0.05, 0.23, 0.41, 0.59, 0.77, 0.95)
    return [ordered_frames[min(last_index, max(0, round(last_index * ratio)))] for ratio in ratios]


def _generate_fireworks_judge_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    *,
    debug_dir: str | Path | None,
    logger: logging.Logger,
    client_factory: Callable[[Any], Any],
    model_config: Any,
    remaining_time_fn: Callable[[], float],
) -> dict[str, str]:
    remaining_seconds = remaining_time_fn()
    if remaining_seconds < FIREWORKS_JUDGE_MIN_GENERATION_REMAINING_SECONDS:
        logger.warning(
            "Task %s has only %.1fs remaining; skipping Fireworks judge generation for safe fallback output.",
            task.task_id,
            remaining_seconds,
        )
        return build_fallback_captions(task.styles)

    selected_frames = select_fireworks_judge_frames(frames)
    if len(selected_frames) != 6:
        logger.warning(
            "Task %s did not provide six Fireworks judge frames; using fallback captions.",
            task.task_id,
        )
        return build_fallback_captions(task.styles)

    client = client_factory(model_config)
    try:
        initial = generate_fireworks_judge_direct_captions(
            task,
            selected_frames,
            client,
            remaining_time_fn=remaining_time_fn,
            debug_dir=debug_dir,
            api_key=str(getattr(model_config, "api_key", "")),
            primary_model=str(getattr(model_config, "vision_model", "")),
        )
    except FireworksRuntimeBudgetError as exc:
        logger.warning("Task %s Fireworks judge generation has insufficient runtime; using fallback captions: %s", task.task_id, exc)
        return build_fallback_captions(task.styles)
    except FireworksVisionRequestError as exc:
        if not exc.retryable:
            logger.warning("Task %s Fireworks judge generation failed without retry eligibility; using fallback captions: %s", task.task_id, exc)
            return build_fallback_captions(task.styles)
        logger.warning("Task %s Fireworks judge generation failed retryably; continuing with focused repair: %s", task.task_id, exc)
        initial = FireworksCaptionExtraction({}, list(task.styles), [])
    except Exception as exc:
        logger.warning("Task %s Fireworks judge generation failed; using fallback captions: %s", task.task_id, exc)
        return build_fallback_captions(task.styles)

    if debug_dir is not None:
        write_fireworks_caption_extraction_debug_file(task.task_id, initial, debug_dir, suffix="initial")

    captions = dict(initial.valid_captions)
    outstanding_styles = _outstanding_fireworks_styles(initial)
    if outstanding_styles:
        repaired = _repair_fireworks_judge_captions(
            task,
            selected_frames,
            client,
            valid_captions=captions,
            outstanding_styles=outstanding_styles,
            remaining_time_fn=remaining_time_fn,
            debug_dir=debug_dir,
            api_key=str(getattr(model_config, "api_key", "")),
            primary_model=str(getattr(model_config, "vision_model", "")),
            fallback_model=str(getattr(model_config, "fallback_vision_model", "")),
        )
        captions.update(repaired.valid_captions)
        if debug_dir is not None:
            write_fireworks_caption_extraction_debug_file(task.task_id, repaired, debug_dir, suffix="repair")

    if set(captions) != set(task.styles):
        logger.warning(
            "Task %s Fireworks judge generation remains incomplete after focused repair; using fallback captions.",
            task.task_id,
        )
        return build_fallback_captions(task.styles)

    if debug_dir is not None:
        write_captions_debug_file(task.task_id, captions, debug_dir, suffix="fireworks_merged")

    remaining_seconds = remaining_time_fn()
    if remaining_seconds < FIREWORKS_JUDGE_SKIP_REVIEW_REMAINING_SECONDS:
        logger.info(
            "Task %s skipping Fireworks judge review because remaining time is %.1fs.",
            task.task_id,
            remaining_seconds,
        )
        return captions

    try:
        final_captions = generate_fireworks_judge_review_captions(
            task,
            selected_frames,
            captions,
            client,
            remaining_time_fn=remaining_time_fn,
            debug_dir=debug_dir,
            api_key=str(getattr(model_config, "api_key", "")),
        )
    except Exception as exc:
        logger.warning("Task %s Fireworks judge review failed; preserving first captions: %s", task.task_id, exc)
        return captions

    if debug_dir is not None:
        write_captions_debug_file(task.task_id, final_captions, debug_dir, suffix="fireworks_judged")
    return final_captions


def generate_fireworks_judge_direct_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    client: Any,
    *,
    remaining_time_fn: Callable[[], float],
    debug_dir: str | Path | None,
    api_key: str,
    primary_model: str,
) -> FireworksCaptionExtraction:
    messages = build_fireworks_judge_generation_messages(task.task_id, task.styles, frames)
    return client.complete_json(
        messages,
        temperature=0.4,
        validator=lambda payload: _extract_fireworks_caption_result(payload, task.styles),
        remaining_time_fn=remaining_time_fn,
        minimum_remaining_seconds=FIREWORKS_JUDGE_MIN_GENERATION_REMAINING_SECONDS,
        operation="generation",
        validation_failure_handler=_fireworks_validation_debug_writer(
            task.task_id,
            "generation",
            debug_dir,
            api_key,
        ),
        response_handler=_fireworks_response_debug_writer(task.task_id, "initial", debug_dir, api_key),
        model_attempts=((primary_model, 1),),
    )


def _repair_fireworks_judge_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    client: Any,
    *,
    valid_captions: dict[str, str],
    outstanding_styles: Sequence[str],
    remaining_time_fn: Callable[[], float],
    debug_dir: str | Path | None,
    api_key: str,
    primary_model: str,
    fallback_model: str,
) -> FireworksCaptionExtraction:
    primary_result = _run_fireworks_judge_repair_attempt(
        task,
        frames,
        client,
        valid_captions=valid_captions,
        outstanding_styles=outstanding_styles,
        remaining_time_fn=remaining_time_fn,
        debug_dir=debug_dir,
        api_key=api_key,
        model=primary_model,
        debug_operation="repair_primary",
    )
    merged = dict(valid_captions)
    if primary_result is not None:
        merged.update(primary_result.valid_captions)
    remaining_styles = [style for style in outstanding_styles if style not in merged]
    if not remaining_styles or not fallback_model:
        return FireworksCaptionExtraction(merged, remaining_styles, [])

    fallback_result = _run_fireworks_judge_repair_attempt(
        task,
        frames,
        client,
        valid_captions=merged,
        outstanding_styles=remaining_styles,
        remaining_time_fn=remaining_time_fn,
        debug_dir=debug_dir,
        api_key=api_key,
        model=fallback_model,
        debug_operation="repair_fallback",
    )
    if fallback_result is not None:
        merged.update(fallback_result.valid_captions)
    remaining_styles = [style for style in outstanding_styles if style not in merged]
    return FireworksCaptionExtraction(merged, remaining_styles, [])


def _run_fireworks_judge_repair_attempt(
    task: Task,
    frames: Sequence[ExtractedFrame],
    client: Any,
    *,
    valid_captions: dict[str, str],
    outstanding_styles: Sequence[str],
    remaining_time_fn: Callable[[], float],
    debug_dir: str | Path | None,
    api_key: str,
    model: str,
    debug_operation: str,
) -> FireworksCaptionExtraction | None:
    messages = build_fireworks_judge_repair_messages(task.task_id, outstanding_styles, valid_captions, frames)
    try:
        return client.complete_json(
            messages,
            temperature=0.25,
            validator=lambda payload: _extract_fireworks_caption_result(payload, outstanding_styles),
            remaining_time_fn=remaining_time_fn,
            minimum_remaining_seconds=FIREWORKS_JUDGE_MIN_GENERATION_REMAINING_SECONDS,
            operation="repair",
            validation_failure_handler=_fireworks_validation_debug_writer(
                task.task_id,
                "repair",
                debug_dir,
                api_key,
            ),
            response_handler=_fireworks_response_debug_writer(task.task_id, debug_operation, debug_dir, api_key),
            model_attempts=((model, 1),),
        )
    except Exception as exc:
        LOGGER.warning("Task %s Fireworks focused repair model %s failed: %s", task.task_id, model, exc)
        return None


def generate_fireworks_judge_review_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    captions: dict[str, str],
    client: Any,
    *,
    remaining_time_fn: Callable[[], float],
    debug_dir: str | Path | None,
    api_key: str,
) -> dict[str, str]:
    messages = build_fireworks_judge_review_messages(task.task_id, task.styles, frames, captions)
    return client.complete_json(
        messages,
        temperature=0.0,
        validator=lambda payload: _validate_fireworks_judge_review(payload, task.styles),
        remaining_time_fn=remaining_time_fn,
        minimum_remaining_seconds=FIREWORKS_JUDGE_MIN_GENERATION_REMAINING_SECONDS,
        operation="review",
        validation_failure_handler=_fireworks_validation_debug_writer(
            task.task_id,
            "review",
            debug_dir,
            api_key,
        ),
    )


def _normalize_exact_caption_keys(payload: Mapping[str, Any], styles: Sequence[str]) -> dict[str, str]:
    extraction = _extract_fireworks_caption_result(payload, styles)
    if extraction.missing_styles:
        raise ValueError(f"missing requested style: {extraction.missing_styles[0]}")
    if extraction.invalid_styles:
        raise ValueError(f"invalid requested style: {extraction.invalid_styles[0]}")
    return extraction.valid_captions


def _validate_fireworks_judge_review(payload: Mapping[str, Any], styles: Sequence[str]) -> dict[str, str]:
    scores = payload.get("scores")
    captions = payload.get("captions")
    if not isinstance(captions, Mapping):
        raise ValueError(f"invalid review caption: {styles[0]}")
    normalized_captions = _extract_requested_fireworks_captions(captions, styles, review=True)

    if not isinstance(scores, Mapping) or not _has_usable_requested_review_scores(scores, styles):
        LOGGER.warning("Fireworks judge review scores unavailable.")
    return normalized_captions


def _extract_requested_fireworks_captions(
    payload: Mapping[str, Any],
    styles: Sequence[str],
    *,
    review: bool = False,
) -> dict[str, str]:
    extraction = _extract_fireworks_caption_result(payload, styles)
    if extraction.missing_styles:
        style = extraction.missing_styles[0]
        if review:
            raise ValueError(f"invalid review caption: {style}")
        raise ValueError(f"missing requested style: {style}")
    if extraction.invalid_styles:
        style = extraction.invalid_styles[0]
        if review:
            raise ValueError(f"invalid review caption: {style}")
        raise ValueError(f"invalid requested style: {style}")
    return extraction.valid_captions


def _extract_fireworks_caption_result(
    payload: Mapping[str, Any],
    styles: Sequence[str],
) -> FireworksCaptionExtraction:
    source = payload.get("captions")
    caption_payload = source if isinstance(source, Mapping) else payload
    found: dict[str, Any] = {}
    for raw_key, value in caption_payload.items():
        if not isinstance(raw_key, str):
            continue
        style = _normalize_fireworks_style_key(raw_key)
        if style in styles and style not in found:
            found[style] = value

    captions: dict[str, str] = {}
    missing_styles: list[str] = []
    invalid_styles: list[str] = []
    for style in styles:
        if style not in found:
            missing_styles.append(style)
            continue
        value = found[style]
        if not isinstance(value, str):
            invalid_styles.append(style)
            continue
        if not value.strip() or _is_invalid_caption_text(value):
            invalid_styles.append(style)
            continue
        captions[style] = cleanup_caption(value, style, {})
    return FireworksCaptionExtraction(captions, missing_styles, invalid_styles)


def _outstanding_fireworks_styles(extraction: FireworksCaptionExtraction) -> list[str]:
    return [*extraction.missing_styles, *extraction.invalid_styles]


def _normalize_fireworks_style_key(value: str) -> str | None:
    compact = re.sub(r"[\s_-]+", "", value.strip().casefold())
    return _STYLE_KEY_ALIASES.get(compact)


def _has_usable_requested_review_scores(scores: Mapping[str, Any], styles: Sequence[str]) -> bool:
    for style in styles:
        score = scores.get(style)
        if not isinstance(score, Mapping):
            return False
        for score_name in ("accuracy", "style_match"):
            value = score.get(score_name)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                return False
    return True


def _fireworks_validation_debug_writer(
    task_id: str,
    operation: str,
    debug_dir: str | Path | None,
    api_key: str,
) -> Callable[[str, int, str], None] | None:
    if debug_dir is None:
        return None

    def write_failed_response(model: str, attempt: int, response_text: str) -> None:
        del model
        output_path = Path(debug_dir) / f"{safe_task_id(task_id)}_fireworks_{operation}_attempt_{attempt}.txt"
        sanitized = _sanitize_fireworks_debug_response(response_text, api_key)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(sanitized, encoding="utf-8")

    return write_failed_response


def _fireworks_response_debug_writer(
    task_id: str,
    operation: str,
    debug_dir: str | Path | None,
    api_key: str,
) -> Callable[[str, int, str], None] | None:
    if debug_dir is None:
        return None

    def write_response(model: str, attempt: int, response_text: str) -> None:
        del model, attempt
        output_path = Path(debug_dir) / f"{safe_task_id(task_id)}_fireworks_{operation}_raw.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_sanitize_fireworks_debug_response(response_text, api_key), encoding="utf-8")

    return write_response


def write_fireworks_caption_extraction_debug_file(
    task_id: str,
    extraction: FireworksCaptionExtraction,
    debug_dir: str | Path,
    *,
    suffix: str,
) -> Path:
    output_path = Path(debug_dir) / f"{safe_task_id(task_id)}_fireworks_{suffix}_extraction.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "valid_captions": extraction.valid_captions,
                "missing_styles": extraction.missing_styles,
                "invalid_styles": extraction.invalid_styles,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def _sanitize_fireworks_debug_response(response_text: str, api_key: str) -> str:
    sanitized = response_text.replace(api_key, "[redacted]") if api_key else response_text
    sanitized = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(
        r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+",
        "[redacted image data]",
        sanitized,
        flags=re.IGNORECASE,
    )
    return sanitized


def _make_remaining_time_fn(
    remaining_seconds: float | None,
    remaining_time_fn: Callable[[], float] | None,
) -> Callable[[], float]:
    if remaining_time_fn is not None:
        return remaining_time_fn
    if remaining_seconds is None:
        return lambda: float("inf")

    started_at = time.monotonic()
    return lambda: max(0.0, remaining_seconds - (time.monotonic() - started_at))


def _generate_google_fast_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    *,
    debug_dir: str | Path | None,
    env: Mapping[str, str],
    logger: logging.Logger,
    client_factory: Callable[[Any], Any],
    model_config: Any,
    remaining_seconds: float | None,
) -> dict[str, str]:
    selected_frames = select_google_evidence_frames(frames)
    client = client_factory(model_config)

    if remaining_seconds is not None and remaining_seconds < GOOGLE_DESCRIPTION_FIRST_MIN_REMAINING_SECONDS:
        logger.info(
            "Task %s using Google direct mode because remaining time is %.1fs.",
            task.task_id,
            remaining_seconds,
        )
        return _generate_google_direct_mode_captions(
            task,
            selected_frames,
            client,
            env,
            debug_dir=debug_dir,
            logger=logger,
        )

    try:
        evidence = generate_google_visual_evidence(task.task_id, selected_frames, client)
        if debug_dir is not None:
            write_evidence_debug_file(task.task_id, selected_frames, evidence, debug_dir)
    except Exception as exc:
        logger.warning(
            "Task %s failed during Google visual evidence generation, falling back to direct mode: %s",
            task.task_id,
            exc,
        )
        return _generate_google_direct_mode_captions(
            task,
            selected_frames,
            client,
            env,
            debug_dir=debug_dir,
            logger=logger,
        )

    used_evidence_fallback = False
    try:
        captions = generate_style_captions(
            task.task_id,
            task.styles,
            evidence,
            client,
            debug_dir=debug_dir,
            max_attempts=2,
        )
    except Exception as exc:
        logger.warning(
            "Task %s failed during Google caption generation after evidence extraction, using evidence-based fallback captions: %s",
            task.task_id,
            exc,
        )
        captions = build_evidence_based_captions(task.styles, evidence)
        used_evidence_fallback = True

    if debug_dir is not None:
        write_captions_debug_file(task.task_id, captions, debug_dir, suffix="raw")

    final_captions = captions
    if not used_evidence_fallback and not _verifier_disabled(env):
        final_captions = maybe_verify_captions(
            task,
            captions,
            evidence,
            client,
            env,
            debug_dir=debug_dir,
        )

    if debug_dir is not None:
        write_captions_debug_file(task.task_id, final_captions, debug_dir, suffix="verified")
    return final_captions


def _generate_google_direct_mode_captions(
    task: Task,
    selected_frames: Sequence[ExtractedFrame],
    client: Any,
    env: Mapping[str, str],
    *,
    debug_dir: str | Path | None,
    logger: logging.Logger,
) -> dict[str, str]:
    used_heuristic_fallback = False
    try:
        captions = generate_google_direct_captions(
            task.task_id,
            task.styles,
            selected_frames,
            client,
            debug_dir=debug_dir,
        )
    except Exception as exc:
        logger.warning(
            "Task %s failed during Google fast caption generation, using heuristic fallback captions: %s",
            task.task_id,
            exc,
        )
        captions = build_visual_heuristic_captions(task.styles, selected_frames)
        if captions is None:
            return build_fallback_captions(task.styles)
        used_heuristic_fallback = True

    if debug_dir is not None:
        write_captions_debug_file(task.task_id, captions, debug_dir, suffix="raw")

    final_captions = captions
    if not used_heuristic_fallback and not _verifier_disabled(env):
        final_captions = maybe_verify_captions(
            task,
            captions,
            {},
            client,
            env,
            debug_dir=debug_dir,
        )

    if debug_dir is not None:
        write_captions_debug_file(task.task_id, final_captions, debug_dir, suffix="verified")
    return final_captions


def _generate_openrouter_experiment_captions(
    task: Task,
    frames: Sequence[ExtractedFrame],
    *,
    debug_dir: str | Path | None,
    env: Mapping[str, str],
    logger: logging.Logger,
    client_factory: Callable[[Any], Any],
    model_config: Any,
    google_fallback_config: Any,
    remaining_seconds: float | None,
) -> dict[str, str]:
    selected_frames = select_google_evidence_frames(frames)
    client = client_factory(model_config)
    caption_client = client

    if remaining_seconds is not None and remaining_seconds < GOOGLE_DESCRIPTION_FIRST_MIN_REMAINING_SECONDS:
        logger.info(
            "Task %s using OpenRouter direct mode because remaining time is %.1fs.",
            task.task_id,
            remaining_seconds,
        )
        return _generate_google_direct_mode_captions(
            task,
            selected_frames,
            client,
            env,
            debug_dir=debug_dir,
            logger=logger,
        )

    try:
        evidence = generate_google_visual_evidence(task.task_id, selected_frames, client)
        if debug_dir is not None:
            write_evidence_debug_file(task.task_id, selected_frames, evidence, debug_dir)
    except Exception as exc:
        if google_fallback_config is not None:
            logger.warning(
                "Task %s failed during OpenRouter evidence generation, falling back to Google v7: %s",
                task.task_id,
                exc,
            )
            return _generate_google_fast_captions(
                task,
                frames,
                debug_dir=debug_dir,
                env=env,
                logger=logger,
                client_factory=client_factory,
                model_config=google_fallback_config.text_model_config(),
                remaining_seconds=remaining_seconds,
            )

        logger.warning(
            "Task %s failed during OpenRouter evidence generation, falling back to direct mode: %s",
            task.task_id,
            exc,
        )
        return _generate_google_direct_mode_captions(
            task,
            selected_frames,
            client,
            env,
            debug_dir=debug_dir,
            logger=logger,
        )

    used_evidence_fallback = False
    try:
        captions = generate_style_captions(
            task.task_id,
            task.styles,
            evidence,
            caption_client,
            debug_dir=debug_dir,
            max_attempts=2,
        )
    except Exception as exc:
        if google_fallback_config is not None:
            logger.warning(
                "Task %s OpenRouter caption failed, falling back to Google caption generation from OpenRouter evidence: %s",
                task.task_id,
                exc,
            )
            try:
                caption_client = client_factory(google_fallback_config.text_model_config())
                captions = generate_style_captions(
                    task.task_id,
                    task.styles,
                    evidence,
                    caption_client,
                    debug_dir=debug_dir,
                    max_attempts=2,
                )
            except Exception as google_exc:
                logger.warning(
                    "Task %s failed during Google caption generation from OpenRouter evidence, using evidence-based fallback captions: %s",
                    task.task_id,
                    google_exc,
                )
                captions = build_evidence_based_captions(task.styles, evidence)
                used_evidence_fallback = True
        else:
            logger.warning(
                "Task %s failed during OpenRouter caption generation after evidence extraction, using evidence-based fallback captions: %s",
                task.task_id,
                exc,
            )
            captions = build_evidence_based_captions(task.styles, evidence)
            used_evidence_fallback = True

    if debug_dir is not None:
        write_captions_debug_file(task.task_id, captions, debug_dir, suffix="raw")

    final_captions = captions
    if not used_evidence_fallback and not _verifier_disabled(env):
        final_captions = maybe_verify_captions(
            task,
            captions,
            evidence,
            caption_client,
            env,
            debug_dir=debug_dir,
        )

    if debug_dir is not None:
        write_captions_debug_file(task.task_id, final_captions, debug_dir, suffix="verified")
    return final_captions


def _load_contact_sheet_cell(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return ImageOps.contain(image.convert("RGB"), (GOOGLE_CONTACT_SHEET_CELL_MAX_SIDE, GOOGLE_CONTACT_SHEET_CELL_MAX_SIDE))


def _frame_heuristic_summary(frames: Sequence[ExtractedFrame]) -> tuple[str, str, str]:
    if not frames:
        raise ValueError("Frames are required for heuristic captions.")

    luminance_values: list[float] = []
    red_values: list[float] = []
    green_values: list[float] = []
    blue_values: list[float] = []
    previous_preview: Image.Image | None = None
    motion_scores: list[float] = []

    for frame in frames:
        with Image.open(frame.path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((96, 96))
            pixels = list(preview.getdata())
            if not pixels:
                continue
            red_values.append(sum(pixel[0] for pixel in pixels) / len(pixels))
            green_values.append(sum(pixel[1] for pixel in pixels) / len(pixels))
            blue_values.append(sum(pixel[2] for pixel in pixels) / len(pixels))
            luminance_values.append(sum(sum(pixel) / 3.0 for pixel in pixels) / len(pixels))
            if previous_preview is not None:
                diff = ImageChops.difference(previous_preview, preview)
                diff_pixels = list(diff.getdata())
                motion_scores.append(sum(sum(pixel) / 3.0 for pixel in diff_pixels) / len(diff_pixels))
            previous_preview = preview.copy()

    if not luminance_values:
        raise ValueError("Could not derive heuristic frame summary.")

    brightness_value = sum(luminance_values) / len(luminance_values)
    brightness = "bright" if brightness_value >= 170 else "darker" if brightness_value <= 100 else "mid-lit"

    mean_red = sum(red_values) / len(red_values)
    mean_green = sum(green_values) / len(green_values)
    mean_blue = sum(blue_values) / len(blue_values)
    if mean_green >= mean_red + 12 and mean_green >= mean_blue + 12:
        tone = "green-toned"
    elif mean_red >= mean_blue + 12 and mean_red >= mean_green + 6:
        tone = "warm-toned"
    elif mean_blue >= mean_red + 12 and mean_blue >= mean_green + 6:
        tone = "cool-toned"
    else:
        tone = "neutral-toned"

    mean_motion = (sum(motion_scores) / len(motion_scores)) if motion_scores else 0.0
    action_phrase = "shows visible movement" if mean_motion >= 18.0 else "stays mostly steady"
    return brightness, tone, action_phrase


def _verifier_disabled(env: Mapping[str, str]) -> bool:
    return env.get("GEMMACLIP_DISABLE_VERIFIER", "").strip().lower() == "true"


def _client_provider(client: Any) -> str:
    config = getattr(client, "_config", None)
    provider = getattr(config, "provider", None)
    return provider if isinstance(provider, str) else DEFAULT_PROVIDER_FIREWORKS


def _load_pillow_image():
    try:
        from PIL import Image as PillowImage
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for Gemma image payload resizing.") from exc

    return PillowImage


def _force_placeholder(env: Mapping[str, str]) -> bool:
    return env.get("GEMMACLIP_FORCE_PLACEHOLDER", "").strip().lower() == "true"


def _force_fallback(env: Mapping[str, str]) -> bool:
    return env.get("GEMMACLIP_FORCE_FALLBACK", "").strip().lower() == "true"
