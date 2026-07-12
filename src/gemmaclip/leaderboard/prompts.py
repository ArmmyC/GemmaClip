from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

from gemmaclip.frames import ExtractedFrame


def build_generation_messages(
    task_id: str,
    styles: Sequence[str],
    frames: Sequence[ExtractedFrame],
) -> list[dict[str, Any]]:
    ordered = _ordered_frames(frames)
    schema = {style: "<18-35 word caption>" for style in styles}
    frame_lines = _frame_lines(ordered)
    user_text = (
        f"Task ID: {task_id}\n"
        "These are six separate JPEG images from one video, listed from earliest to latest. "
        "Use the timestamps to understand temporal progression.\n"
        f"{frame_lines}\n"
        f"Requested styles: {', '.join(styles)}\n"
        "Return exactly this JSON object, with no additional keys:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Every requested key is mandatory and must appear exactly once. Return no markdown, explanation, or prose."
    )
    return [
        {"role": "system", "content": build_generation_system_prompt()},
        {"role": "user", "content": _multimodal_content(user_text, ordered)},
    ]


def build_repair_messages(
    task_id: str,
    missing_styles: Sequence[str],
    valid_captions: Mapping[str, str],
    frames: Sequence[ExtractedFrame],
) -> list[dict[str, Any]]:
    ordered = _ordered_frames(frames)
    schema = {style: "<18-35 word caption>" for style in missing_styles}
    user_text = (
        f"Task ID: {task_id}\n"
        f"Missing or invalid styles to repair: {', '.join(missing_styles)}\n"
        f"Retained valid captions JSON:\n{json.dumps(dict(valid_captions), indent=2)}\n"
        "Use the six separate JPEG frames below in chronological timestamp order. "
        "The repaired captions must describe the same visible subject and action.\n"
        f"{_frame_lines(ordered)}\n"
        "Return only this JSON object with exactly the missing style keys. Do not return any retained valid style:\n"
        f"{json.dumps(schema, indent=2)}"
    )
    return [
        {"role": "system", "content": build_repair_system_prompt()},
        {"role": "user", "content": _multimodal_content(user_text, ordered)},
    ]


def build_review_messages(
    task_id: str,
    styles: Sequence[str],
    frames: Sequence[ExtractedFrame],
    captions: Mapping[str, str],
) -> list[dict[str, Any]]:
    ordered = _ordered_frames(frames)
    schema = {
        "scores": {
            style: {"accuracy": 0.0, "style_match": 0.0}
            for style in styles
        },
        "captions": {style: "caption" for style in styles},
    }
    user_text = (
        f"Task ID: {task_id}\n"
        "Review these captions against the same six separate chronological JPEG frames.\n"
        f"Requested styles: {', '.join(styles)}\n"
        f"Current captions JSON:\n{json.dumps(dict(captions), indent=2)}\n"
        f"{_frame_lines(ordered)}\n"
        "Return exactly this JSON shape, with every requested score and caption key:\n"
        f"{json.dumps(schema, indent=2)}"
    )
    return [
        {"role": "system", "content": build_review_system_prompt()},
        {"role": "user", "content": _multimodal_content(user_text, ordered)},
    ]


def build_generation_system_prompt() -> str:
    return (
        "You write grounded captions for six separate chronological frames from one video. Identify the visible "
        "main subject, main visible action, setting, and meaningful progression across frames. Distinguish observed "
        "facts from assumptions. Return only one JSON object with exactly the requested keys; no markdown, code fences, "
        "analysis, or explanations. Every caption must be 18 to 35 words, a complete sentence, and describe the same "
        "central visible event.\n\n"
        "Grounding: describe only visible information. Do not invent names, brands, relationships, occupations, motives, "
        "thoughts, emotions, offscreen events, dialogue, speech, music, sound, noise, or camera equipment. Do not quote "
        "text unless clearly legible and necessary. Never use uncertain phrasing as a substitute for grounding.\n\n"
        "formal is factual, clear, neutral, and complete. sarcastic is dry, lightly mocking, and grounded without cruelty. "
        "humorous_tech uses one familiar technology analogy figuratively while keeping the visible event central; do not "
        "claim code, software, servers, CPUs, debugging, or programming are literally present unless visibly supported. "
        "humorous_non_tech uses one everyday comparison or mild punchline without technical jargon, memes, or invented "
        "backstory. Make every style clearly different in voice but faithful to the same subject and action. Silently "
        "verify word bounds, grounding, exact keys, and JSON before responding."
    )


def build_repair_system_prompt() -> str:
    return (
        "You are a focused caption repair writer. Use only the six separate chronological video frames and the retained "
        "valid captions. Return only the missing or invalid requested style keys as JSON. Never return or rewrite an "
        "already valid style. Every repair must be a complete 18 to 35 word caption describing the same visible subject "
        "and action, with the requested style grounded in the frames. Do not add names, brands, relationships, motives, "
        "thoughts, dialogue, speech, music, sound, noise, or unseen events. No markdown, analysis, or extra prose."
    )


def build_review_system_prompt() -> str:
    return (
        "You are an independent visual caption reviewer. Check every caption against all six chronological frames for "
        "visual accuracy, subject and action coverage, temporal consistency, hallucination, requested style match, "
        "18-35 word bounds, distinction from other captions, generic filler, awkward technology humor, and unsupported "
        "audio claims. Return scores from 0.0 to 1.0 and the complete requested captions. Preserve a caption when it is "
        "accurate and stylistically strong. Rewrite only when needed, making the smallest useful correction. Never add "
        "facts absent from the frames. Return only the required JSON object, with no markdown or explanations."
    )


def _ordered_frames(frames: Sequence[ExtractedFrame]) -> list[ExtractedFrame]:
    return sorted(frames, key=lambda frame: (frame.timestamp_seconds, str(frame.path)))


def _frame_lines(frames: Sequence[ExtractedFrame]) -> str:
    return "\n".join(
        f"Frame {index}: timestamp_seconds={frame.timestamp_seconds:.3f}; the next image is this frame."
        for index, frame in enumerate(frames, start=1)
    )


def _multimodal_content(text: str, frames: Sequence[ExtractedFrame]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for frame in frames:
        encoded = base64.b64encode(frame.path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
        )
    return content


# Compatibility-oriented names make the request builder easy to exercise in
# isolated tests without exposing any provider client internals.
build_fireworks_generation_messages = build_generation_messages
build_fireworks_repair_messages = build_repair_messages
build_fireworks_review_messages = build_review_messages
