from __future__ import annotations

import json
import builtins

import pytest

from gemmaclip.frames import ExtractedFrame, generate_contact_sheet
from gemmaclip.io import make_frame_manifest_entry, read_tasks, write_frame_manifest
from gemmaclip.video import VideoMetadata


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


def test_write_frame_manifest_writes_expected_payload(tmp_path):
    manifest_path = tmp_path / "frame_manifest.json"
    entry = make_frame_manifest_entry(
        task_id="clip-1",
        video_path=tmp_path / "videos" / "clip-1.mp4",
        frames=[
            ExtractedFrame(
                path=tmp_path / "frames" / "clip-1" / "frame_001.jpg",
                timestamp_seconds=0.25,
            ),
            ExtractedFrame(
                path=tmp_path / "frames" / "clip-1" / "frame_002.jpg",
                timestamp_seconds=3.75,
            ),
        ],
        metadata=VideoMetadata(
            duration_seconds=42.5,
            fps=24.0,
            width=1920,
            height=1080,
            frame_count=1020,
        ),
    )

    write_frame_manifest([entry], manifest_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload == [
        {
            "task_id": "clip-1",
            "video_path": str(tmp_path / "videos" / "clip-1.mp4"),
            "frame_paths": [
                str(tmp_path / "frames" / "clip-1" / "frame_001.jpg"),
                str(tmp_path / "frames" / "clip-1" / "frame_002.jpg"),
            ],
            "frames": [
                {
                    "path": str(tmp_path / "frames" / "clip-1" / "frame_001.jpg"),
                    "timestamp_seconds": 0.25,
                },
                {
                    "path": str(tmp_path / "frames" / "clip-1" / "frame_002.jpg"),
                    "timestamp_seconds": 3.75,
                },
            ],
            "metadata": {
                "duration_seconds": 42.5,
                "fps": 24.0,
                "width": 1920,
                "height": 1080,
                "frame_count": 1020,
            },
        }
    ]


def test_generate_contact_sheet_reports_missing_pillow(tmp_path, monkeypatch):
    frame_path = tmp_path / "frame_001.jpg"
    frame_path.write_bytes(b"placeholder")

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "PIL":
            raise ModuleNotFoundError("No module named 'PIL'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="Pillow is required for --debug-dir contact sheets"):
        generate_contact_sheet([frame_path], tmp_path / "contact_sheet.jpg")
