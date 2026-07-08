from __future__ import annotations

import pytest

from gemmaclip.io import Task
from gemmaclip.validate import validate_results


def make_tasks():
    return [
        Task(
            task_id="clip-1",
            video_url="https://example.com/video-1.mp4",
            styles=("formal", "sarcastic"),
        ),
        Task(
            task_id="clip-2",
            video_url="https://example.com/video-2.mp4",
            styles=("formal",),
        ),
    ]


def test_validate_results_accepts_valid_results():
    validate_results(
        make_tasks(),
        [
            {
                "task_id": "clip-1",
                "captions": {
                    "formal": "A valid placeholder caption for the formal style.",
                    "sarcastic": "A valid placeholder caption for the sarcastic style.",
                },
            },
            {
                "task_id": "clip-2",
                "captions": {
                    "formal": "Another valid placeholder caption for the formal style.",
                },
            },
        ],
    )


def test_validate_results_rejects_missing_requested_style():
    with pytest.raises(ValueError, match="missing caption text for style sarcastic"):
        validate_results(
            make_tasks(),
            [
                {
                    "task_id": "clip-1",
                    "captions": {
                        "formal": "A valid placeholder caption for the formal style.",
                    },
                },
                {
                    "task_id": "clip-2",
                    "captions": {
                        "formal": "Another valid placeholder caption for the formal style.",
                    },
                },
            ],
        )


def test_validate_results_rejects_unexpected_task_id():
    with pytest.raises(ValueError, match="Unexpected result task_id found: clip-3"):
        validate_results(
            make_tasks(),
            [
                {
                    "task_id": "clip-3",
                    "captions": {
                        "formal": "A valid placeholder caption for the formal style.",
                    },
                },
                {
                    "task_id": "clip-2",
                    "captions": {
                        "formal": "Another valid placeholder caption for the formal style.",
                    },
                },
            ],
        )


def test_validate_results_rejects_duplicate_result_task_id():
    with pytest.raises(ValueError, match="Duplicate result task_id found: clip-1"):
        validate_results(
            make_tasks(),
            [
                {
                    "task_id": "clip-1",
                    "captions": {
                        "formal": "A valid placeholder caption for the formal style.",
                        "sarcastic": "A valid placeholder caption for the sarcastic style.",
                    },
                },
                {
                    "task_id": "clip-1",
                    "captions": {
                        "formal": "A second formal placeholder caption.",
                        "sarcastic": "A second sarcastic placeholder caption.",
                    },
                },
            ],
        )
