from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from gemmaclip.io import Task


def validate_results(tasks: Sequence[Task], results: Sequence[Mapping[str, Any]]) -> None:
    if len(results) != len(tasks):
        raise ValueError("Result count must match task count.")

    task_by_id = {task.task_id: task for task in tasks}
    result_task_ids: set[str] = set()

    for result in results:
        task_id = result.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("Each result item must include a non-empty task_id.")
        if task_id in result_task_ids:
            raise ValueError(f"Duplicate result task_id found: {task_id}")
        if task_id not in task_by_id:
            raise ValueError(f"Unexpected result task_id found: {task_id}")

        captions = result.get("captions")
        if not isinstance(captions, Mapping):
            raise ValueError(f"Result for task {task_id} must include a captions object.")

        requested_styles = task_by_id[task_id].styles
        for style in requested_styles:
            caption = captions.get(style)
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"Result for task {task_id} is missing caption text for style {style}.")

        result_task_ids.add(task_id)

    missing_task_ids = set(task_by_id) - result_task_ids
    if missing_task_ids:
        missing = ", ".join(sorted(missing_task_ids))
        raise ValueError(f"Results are missing task_ids: {missing}")
