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
    "temporal_progression": "",
    "caption_focus": "",
}


def build_evidence_system_prompt() -> str:
    return (
        "You are GemmaClip's factual video analyst. Use only the provided frames and timestamps. "
        "Return only the final JSON object with these keys: "
        "scene, main_subjects, actions, setting, visible_objects, mood, camera_notes, temporal_progression, caption_focus. "
        "Use strings for scene, setting, mood, camera_notes. Use arrays of strings for main_subjects, "
        "actions, visible_objects. Use strings for temporal_progression and caption_focus. Produce dense factual "
        "understanding, not generic summaries. Identify the main subject, the main action, the setting, relevant "
        "objects, and how the clip changes over time. Do not invent identities, brands, exact locations, dialogue, "
        "or offscreen motivations. Do not include analysis, reasoning, markdown, code fences, or extra commentary."
    )


def build_evidence_user_prompt(task_id: str, frames: Sequence[ExtractedFrame]) -> str:
    frame_lines = [
        f"- {frame.path.name}: timestamp_seconds={frame.timestamp_seconds:.3f}"
        for frame in frames
    ]
    return (
        f"Task ID: {task_id}\n"
        "Analyze the video frames in timestamp order and produce factual evidence JSON.\n"
        "Do not invent speech, brands, exact locations, identities, or events.\n"
        "Frames:\n"
        f"{chr(10).join(frame_lines)}"
    )


def build_google_visual_evidence_user_prompt(task_id: str, frames: Sequence[ExtractedFrame]) -> str:
    frame_lines = [
        f"- slot {index + 1}: {frame.path.name} at timestamp_seconds={frame.timestamp_seconds:.3f}"
        for index, frame in enumerate(frames)
    ]
    return (
        f"Task ID: {task_id}\n"
        "The uploaded image is a chronological contact sheet from one video.\n"
        "Read it as row-major order from earliest to latest frame.\n"
        "Produce dense factual evidence JSON about the clip, including the main subject, the main action, the "
        "setting, visible objects, motion or progression over time, and the best factual focus for captions.\n"
        "Avoid generic descriptions. Do not invent identities, brands, exact locations, dialogue, or offscreen "
        "motivations.\n"
        "Frame order:\n"
        f"{chr(10).join(frame_lines)}\n"
        "Return only the JSON object."
    )


def build_direct_caption_system_prompt() -> str:
    return (
        "Describe visible video content faithfully from the provided contact sheet. Return only a JSON object with "
        "exactly the requested style keys. No markdown. No explanation. No code fences. Each caption must be 12 to "
        "22 words, must mention the main visible subject and the main visible action, and must stay specific to the "
        "clip. formal must be factual. sarcastic must be dry and lightly ironic. humorous_tech must use a light tech "
        "metaphor. humorous_non_tech must use everyday humor. Do not invent offscreen thoughts, future events, "
        "dialogue, names, brands, or exact locations."
    )


def build_direct_caption_user_prompt(
    task_id: str,
    styles: Sequence[str],
    frames: Sequence[ExtractedFrame],
) -> str:
    frame_lines = [
        f"- {frame.path.name}: timestamp_seconds={frame.timestamp_seconds:.3f}"
        for frame in frames
    ]
    return (
        f"Task ID: {task_id}\n"
        f"Requested styles: {', '.join(styles)}\n"
        "The uploaded image is a chronological contact sheet built from representative video frames.\n"
        "Read it in row-major order from earliest to latest frame.\n"
        "Frame timestamps:\n"
        f"{chr(10).join(frame_lines)}\n"
        "Return only the final JSON object with exactly the requested style keys."
    )


def build_direct_caption_repair_user_prompt(
    task_id: str,
    styles: Sequence[str],
    previous_response: str,
) -> str:
    return (
        f"Task ID: {task_id}\n"
        f"Requested styles: {', '.join(styles)}\n"
        "The previous response was invalid. Return only a JSON object with exactly these style keys. "
        "No markdown. No explanation. Each value must be a complete caption, not punctuation.\n"
        "Previous raw model response:\n"
        f"{previous_response}\n"
        "Return only the corrected JSON object."
    )


def build_caption_system_prompt() -> str:
    return (
        "You are GemmaClip's caption writer. Describe visible content faithfully. Do not invent details that are not "
        "visible. Keep captions natural, concise, and specific to the clip. Return only the final JSON object with "
        "the requested style keys and no other text. Each caption must be 12 to 22 words and must mention the main "
        "visible subject and main visible action. Prioritize evidence fields in this order: main_subjects, actions, "
        "setting, visible_objects, mood, camera_notes, temporal_progression, caption_focus. Use camera_notes only "
        "when they genuinely help describe the clip, not as default joke material. Do not mention exact sign text, "
        "brand text, or other readable text unless it is central to the scene. Do not invent offscreen thoughts, "
        "future events, dialogue, or unseen motives. Avoid likely, probably, maybe, appears to be, seems to be, "
        "seem, seems, seeming, seemingly, as if, and hoping. Avoid generic phrases like 'the scene shows', 'short "
        "video', 'visible activity', 'ordinary moment', or other filler summaries. Captions for different styles "
        "must describe the same factual event but should not look like one template with tone words swapped. Prefer "
        "neutral person words unless the evidence clearly supports something more specific. formal must stay factual "
        "and objective. sarcastic must be dry and lightly mocking while staying grounded in the specific action. "
        "humorous_tech should use light, common tech metaphors while keeping the real subject and action clear; avoid "
        "awkward jargon like organic hardware, human interface, processing data, protocol, substrate, visual sensors, "
        "module, or collision domains; prefer common metaphors like loading, buffering, data packets, CPU, update, "
        "network, or algorithm. humorous_non_tech should use everyday humor without tech words or invented details."
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
        "Prioritize evidence fields in this order: main_subjects, actions, setting, visible_objects, mood, "
        "camera_notes, temporal_progression, caption_focus.\n"
        "Keep humor natural and grounded. Avoid offscreen thoughts, future events, or using camera notes as the joke "
        "unless central.\n"
    )


def build_caption_repair_user_prompt(
    task_id: str,
    styles: Sequence[str],
    evidence: dict[str, Any],
    previous_response: str,
) -> str:
    return (
        f"Task ID: {task_id}\n"
        f"Requested styles: {', '.join(styles)}\n"
        "The previous response was invalid. Return only a JSON object with exactly these style keys. "
        "No markdown. No explanation. Each value must be a complete caption, not punctuation.\n"
        "Evidence JSON:\n"
        f"{json.dumps(evidence, indent=2)}\n"
        "Previous raw model response:\n"
        f"{previous_response}\n"
        "Return only the corrected JSON object."
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
