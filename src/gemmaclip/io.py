from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SUPPORTED_STYLES = {
    "formal",
    "sarcastic",
    "humorous_tech",
    "humorous_non_tech",
}

_SAFE_TASK_ID_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    video_url: str
    styles: tuple[str, ...]


def read_tasks(input_path: str | Path) -> list[Task]:
    path = Path(input_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Task file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Task file is not valid JSON: {path}") from exc

    if not isinstance(payload, list):
        raise ValueError("Task file must contain a JSON array of tasks.")

    tasks = [parse_task(item, index) for index, item in enumerate(payload)]
    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task file contains duplicate task_id values.")
    return tasks


def parse_task(payload: Any, index: int) -> Task:
    if not isinstance(payload, dict):
        raise ValueError(f"Task at index {index} must be an object.")

    task_id = _require_non_empty_string(payload.get("task_id"), f"Task {index} task_id")
    video_url = _require_non_empty_string(payload.get("video_url"), f"Task {task_id} video_url")
    _validate_video_url(video_url, task_id)

    styles_value = payload.get("styles")
    if not isinstance(styles_value, list) or not styles_value:
        raise ValueError(f"Task {task_id} styles must be a non-empty list.")

    styles: list[str] = []
    for style in styles_value:
        style_name = _require_non_empty_string(style, f"Task {task_id} style")
        if style_name not in SUPPORTED_STYLES:
            raise ValueError(f"Task {task_id} requested unsupported style: {style_name}")
        styles.append(style_name)

    if len(styles) != len(set(styles)):
        raise ValueError(f"Task {task_id} contains duplicate styles.")

    return Task(task_id=task_id, video_url=video_url, styles=tuple(styles))


def write_results(results: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")


def safe_task_id(task_id: str) -> str:
    cleaned = _SAFE_TASK_ID_PATTERN.sub("_", task_id).strip("._-")
    if cleaned == task_id and cleaned:
        return cleaned
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned or 'task'}_{digest}"


def _require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value.strip()


def _validate_video_url(video_url: str, task_id: str) -> None:
    parsed = urlparse(video_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Task {task_id} video_url must be an absolute http or https URL.")
