from __future__ import annotations

import json

import pytest

from gemmaclip.io import read_tasks


def test_read_tasks_accepts_valid_payload(tmp_path):
    input_path = tmp_path / "tasks.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "clip-1",
                    "video_url": "https://example.com/video.mp4",
                    "styles": ["formal", "sarcastic"],
                }
            ]
        ),
        encoding="utf-8",
    )

    tasks = read_tasks(input_path)

    assert len(tasks) == 1
    assert tasks[0].task_id == "clip-1"
    assert tasks[0].video_url == "https://example.com/video.mp4"
    assert tasks[0].styles == ("formal", "sarcastic")


def test_read_tasks_rejects_malformed_payload(tmp_path):
    input_path = tmp_path / "tasks.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "task_id": "clip-1",
                    "video_url": "https://example.com/video.mp4",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="styles must be a non-empty list"):
        read_tasks(input_path)
