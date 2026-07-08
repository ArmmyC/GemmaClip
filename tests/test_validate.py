from __future__ import annotations

import pytest

from gemmaclip.io import Task
from gemmaclip.validate import validate_results


def test_validate_results_rejects_missing_requested_style():
    tasks = [
        Task(
            task_id="clip-1",
            video_url="https://example.com/video.mp4",
            styles=("formal", "sarcastic"),
        )
    ]
    results = [
        {
            "task_id": "clip-1",
            "captions": {
                "formal": "A valid placeholder caption for the formal style.",
            },
        }
    ]

    with pytest.raises(ValueError, match="missing caption text for style sarcastic"):
        validate_results(tasks, results)


def test_validate_results_accepts_correct_result():
    tasks = [
        Task(
            task_id="clip-1",
            video_url="https://example.com/video.mp4",
            styles=("formal", "sarcastic"),
        )
    ]
    results = [
        {
            "task_id": "clip-1",
            "captions": {
                "formal": "A valid placeholder caption for the formal style.",
                "sarcastic": "A valid placeholder caption for the sarcastic style.",
            },
        }
    ]

    validate_results(tasks, results)
