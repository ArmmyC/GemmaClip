from __future__ import annotations

import json

import pytest

from gemmaclip.io import read_tasks


def write_tasks_file(tmp_path, payload) -> None:
    (tmp_path / "tasks.json").write_text(json.dumps(payload), encoding="utf-8")


def test_read_tasks_accepts_valid_payload(tmp_path):
    write_tasks_file(
        tmp_path,
        [
            {
                "task_id": "clip-1",
                "video_url": "https://example.com/video.mp4",
                "styles": ["formal", "sarcastic"],
            }
        ],
    )

    tasks = read_tasks(tmp_path / "tasks.json")

    assert len(tasks) == 1
    assert tasks[0].task_id == "clip-1"
    assert tasks[0].video_url == "https://example.com/video.mp4"
    assert tasks[0].styles == ("formal", "sarcastic")


def test_read_tasks_rejects_invalid_json_shape(tmp_path):
    write_tasks_file(
        tmp_path,
        {
            "task_id": "clip-1",
            "video_url": "https://example.com/video.mp4",
            "styles": ["formal"],
        },
    )

    with pytest.raises(ValueError, match="JSON array of tasks"):
        read_tasks(tmp_path / "tasks.json")


def test_read_tasks_rejects_unsupported_style(tmp_path):
    write_tasks_file(
        tmp_path,
        [
            {
                "task_id": "clip-1",
                "video_url": "https://example.com/video.mp4",
                "styles": ["formal", "dramatic"],
            }
        ],
    )

    with pytest.raises(ValueError, match="unsupported style"):
        read_tasks(tmp_path / "tasks.json")


def test_read_tasks_rejects_duplicate_styles(tmp_path):
    write_tasks_file(
        tmp_path,
        [
            {
                "task_id": "clip-1",
                "video_url": "https://example.com/video.mp4",
                "styles": ["formal", "formal"],
            }
        ],
    )

    with pytest.raises(ValueError, match="duplicate styles"):
        read_tasks(tmp_path / "tasks.json")


def test_read_tasks_rejects_invalid_video_url(tmp_path):
    write_tasks_file(
        tmp_path,
        [
            {
                "task_id": "clip-1",
                "video_url": "not-a-url",
                "styles": ["formal"],
            }
        ],
    )

    with pytest.raises(ValueError, match="absolute http or https URL"):
        read_tasks(tmp_path / "tasks.json")
