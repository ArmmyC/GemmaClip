from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from gemmaclip.frames import ExtractedFrame

EVIDENCE_SCHEMA = {
    "scene": "",
    "main_subjects": [],
    "actions": [],
    "setting": "",
    "visible_objects": [],
    "mood": "",
    "camera_notes": "",
    "uncertain_details": [],
}


def build_evidence_system_prompt() -> str:
    return (
        "You are GemmaClip's factual video analyst. Use only the provided frames and timestamps. "
        "Return only the final JSON object with these keys: "
        "scene, main_subjects, actions, setting, visible_objects, mood, camera_notes, uncertain_details. "
        "Use strings for scene, setting, mood, camera_notes. Use arrays of strings for main_subjects, "
        "actions, visible_objects, uncertain_details. Do not include analysis, reasoning, markdown, code fences, "
        "or extra commentary."
    )


def build_evidence_user_prompt(task_id: str, frames: Sequence[ExtractedFrame]) -> str:
    frame_lines = [
        f"- {frame.path.name}: timestamp_seconds={frame.timestamp_seconds:.3f}"
        for frame in frames
    ]
    return (
        f"Task ID: {task_id}\n"
        "Analyze the video frames in timestamp order and produce factual evidence JSON.\n"
        "Do not invent speech, brands, locations, identities, or events.\n"
        "Frames:\n"
        f"{chr(10).join(frame_lines)}"
    )


def build_caption_system_prompt() -> str:
    return (
        "Describe visible content faithfully. Do not invent details that are not visible. Keep captions natural, "
        "concise, and specific to the clip. Return only the final JSON object with the requested style keys and no "
        "other text. Each caption must be 12 to 22 words and must mention the main visible subject and main visible "
        "action. Prioritize evidence fields in this order: main_subjects, actions, setting, visible_objects, mood, "
        "camera_notes. Use camera_notes only when they help describe the video, not as the main joke. Do not mention "
        "exact sign text, brand text, or other readable text unless it is central to the scene. Avoid likely, "
        "probably, maybe, appears to be, seems to be, seem, seems, seeming, seemingly, as if, and hoping. Prefer "
        "neutral person words unless the evidence clearly supports something more specific. sarcastic must stay dry "
        "and light. humorous_tech should use light tech metaphors while keeping the real subject and action clear; "
        "avoid over-technical wording like protocol, substrate, visual sensors, module, or collision domains; prefer "
        "common metaphors like loading, buffering, data packets, CPU, update, network, or algorithm. humorous_non_tech "
        "should use everyday humor and must not invent offscreen thoughts, future actions, or unrelated details."
    )


def build_caption_user_prompt(
    task_id: str,
    styles: Sequence[str],
    evidence: dict[str, Any],
) -> str:
    return (
        f"Task ID: {task_id}\n"
        f"Requested styles: {', '.join(styles)}\n"
        "Generate captions only from this evidence JSON:\n"
        f"{json.dumps(evidence, indent=2)}\n"
        "Return only the final JSON object with only the requested style keys.\n"
        "Do not include analysis, reasoning, markdown, or code fences.\n"
        "In every style, mention the main visible subject and the main visible action.\n"
        "Prioritize evidence fields in this order: main_subjects, actions, setting, visible_objects, mood, camera_notes.\n"
    )


def build_verifier_system_prompt() -> str:
    return (
        "You are GemmaClip's caption verifier and minimal refiner. Use only the provided evidence JSON. "
        "Return only the final JSON object with the same requested style keys and final caption strings. "
        "Keep good captions unchanged. Only minimally rewrite captions that invent unsupported facts, use banned "
        "speculation phrases, exceed 25 words, weakly match the requested style, make unsupported coding, script, "
        "developer, debugging, programming, or software-development claims, or lose the main visible subject or "
        "action. Keep captions natural, concise, and specific. Do not add new visual facts beyond the evidence. Do "
        "not include analysis, reasoning, markdown, or code fences."
    )


def build_verifier_user_prompt(
    task_id: str,
    styles: Sequence[str],
    evidence: dict[str, Any],
    captions: dict[str, str],
) -> str:
    return (
        f"Task ID: {task_id}\n"
        f"Requested styles: {', '.join(styles)}\n"
        "Evidence JSON:\n"
        f"{json.dumps(evidence, indent=2)}\n"
        "Current captions JSON:\n"
        f"{json.dumps(captions, indent=2)}\n"
        "Return only the final JSON object with the same style keys and final caption strings."
    )
