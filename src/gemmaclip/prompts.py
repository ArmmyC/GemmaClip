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
        "Return exactly one JSON object with these keys: "
        "scene, main_subjects, actions, setting, visible_objects, mood, camera_notes, uncertain_details. "
        "Use strings for scene, setting, mood, camera_notes. Use arrays of strings for main_subjects, "
        "actions, visible_objects, uncertain_details. Do not include markdown or extra commentary."
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
        "You are GemmaClip's caption writer. Use only the provided evidence JSON. "
        "Return exactly one JSON object whose keys are the requested styles and whose values are English captions. "
        "Each caption must be 12 to 25 words, accurate before funny, and must not invent objects, speech, brands, "
        "locations, identities, or events. Sarcastic must be dry and light, humorous_tech must include a clear "
        "technology or programming flavor, and humorous_non_tech must avoid technical jargon."
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
        "Return a JSON object with only the requested style keys."
    )
