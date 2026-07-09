from __future__ import annotations

from pathlib import Path

from gemmaclip.frames import ExtractedFrame
from gemmaclip.io import Task
from gemmaclip.main import build_progress_results, process_task, process_tasks, should_fill_remaining_with_fallbacks
from gemmaclip.video import VideoMetadata


def make_tasks() -> list[Task]:
    return [
        Task(
            task_id="clip-1",
            video_url="https://example.com/video-1.mp4",
            styles=("formal", "sarcastic"),
        ),
        Task(
            task_id="clip-2",
            video_url="https://example.com/video-2.mp4",
            styles=("formal", "sarcastic"),
        ),
        Task(
            task_id="clip-3",
            video_url="https://example.com/video-3.mp4",
            styles=("formal", "sarcastic"),
        ),
    ]


def test_process_tasks_writes_results_after_each_task(tmp_path, monkeypatch):
    tasks = make_tasks()[:2]
    write_calls: list[list[dict[str, object]]] = []

    def fake_process_task(task, **kwargs):
        return (
            {
                "task_id": task.task_id,
                "captions": {
                    "formal": f"{task.task_id} formal caption.",
                    "sarcastic": f"{task.task_id} sarcastic caption.",
                },
            },
            None,
        )

    def fake_write_results(results, output_path):
        write_calls.append(results)

    monkeypatch.setattr("gemmaclip.main.process_task", fake_process_task)
    monkeypatch.setattr("gemmaclip.main.write_results", fake_write_results)
    monkeypatch.setattr("gemmaclip.main.write_frame_manifest", lambda entries, path: None)

    process_tasks(
        tasks,
        workdir=tmp_path,
        output_path=tmp_path / "results.json",
        max_runtime_seconds=570.0,
        start_time=0.0,
        now_fn=lambda: 0.0,
    )

    assert len(write_calls) == 3
    assert write_calls[0][0]["task_id"] == "clip-1"
    assert write_calls[0][1]["task_id"] == "clip-2"
    assert write_calls[0][1]["captions"]["formal"]


def test_process_tasks_deadline_fills_remaining_tasks_with_fallbacks(tmp_path, monkeypatch):
    tasks = make_tasks()
    write_calls: list[list[dict[str, object]]] = []
    current_time = {"value": 0.0}

    def fake_now():
        return current_time["value"]

    def fake_process_task(task, **kwargs):
        current_time["value"] = 530.0
        return (
            {
                "task_id": task.task_id,
                "captions": {
                    "formal": f"{task.task_id} formal caption.",
                    "sarcastic": f"{task.task_id} sarcastic caption.",
                },
            },
            None,
        )

    def fake_write_results(results, output_path):
        write_calls.append(results)

    monkeypatch.setattr("gemmaclip.main.process_task", fake_process_task)
    monkeypatch.setattr("gemmaclip.main.write_results", fake_write_results)
    monkeypatch.setattr("gemmaclip.main.write_frame_manifest", lambda entries, path: None)

    results = process_tasks(
        tasks,
        workdir=tmp_path,
        output_path=tmp_path / "results.json",
        max_runtime_seconds=570.0,
        start_time=0.0,
        now_fn=fake_now,
    )

    assert results[0]["task_id"] == "clip-1"
    assert results[0]["captions"]["formal"] == "clip-1 formal caption."
    assert results[1]["captions"]["formal"]
    assert results[2]["captions"]["formal"]
    assert results[1]["captions"]["formal"] != "clip-2 formal caption."
    assert len(write_calls) == 2


def test_runtime_guard_continues_when_enough_time_remains_for_one_more_task():
    assert not should_fill_remaining_with_fallbacks(
        0.0,
        completed_count=3,
        remaining_task_count=9,
        max_runtime_seconds=570.0,
        now_fn=lambda: 300.0,
    )


def test_build_progress_results_returns_valid_fallbacks_for_pending_tasks():
    tasks = make_tasks()[:2]
    results = build_progress_results(
        tasks,
        {
            "clip-1": {
                "task_id": "clip-1",
                "captions": {
                    "formal": "clip-1 formal caption.",
                    "sarcastic": "clip-1 sarcastic caption.",
                },
            }
        },
    )

    assert results[0]["task_id"] == "clip-1"
    assert results[1]["task_id"] == "clip-2"
    assert results[1]["captions"]["formal"]


def test_runtime_guard_stops_only_near_hard_budget(caplog):
    caplog.set_level("INFO")

    should_stop = should_fill_remaining_with_fallbacks(
        0.0,
        completed_count=3,
        remaining_task_count=9,
        max_runtime_seconds=570.0,
        now_fn=lambda: 461.0,
    )

    assert should_stop is True
    assert "Runtime guard: elapsed=461.0s remaining=109.0s completed=3 remaining_tasks=9 next_task_budget=90.0s fill_remaining=True" in caplog.text


def test_process_task_uses_google_fast_frame_extraction(tmp_path, monkeypatch):
    task = make_tasks()[0]
    extraction_modes: list[bool] = []
    frame_path = tmp_path / "frame_001.jpg"
    frame_path.write_bytes(b"jpeg")

    monkeypatch.setattr("gemmaclip.main.download_video", lambda task: tmp_path / "video.mp4")
    monkeypatch.setattr(
        "gemmaclip.main.probe_video",
        lambda video_path: VideoMetadata(duration_seconds=100.0, fps=24.0, width=1920, height=1080, frame_count=2400),
    )

    def fake_extract_frames(task_id, video_path, metadata, **kwargs):
        extraction_modes.append(kwargs["google_fast"])
        return [ExtractedFrame(path=frame_path, timestamp_seconds=5.0)]

    monkeypatch.setattr("gemmaclip.main.extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        "gemmaclip.main.generate_captions",
        lambda task, frames, **kwargs: {
            "formal": "clip-1 formal caption.",
            "sarcastic": "clip-1 sarcastic caption.",
        },
    )

    result, _ = process_task(
        task,
        workdir=tmp_path,
        env={"GEMINI_API_KEY": "gemini-key", "GEMMACLIP_DISABLE_VERIFIER": "true"},
    )

    assert result["captions"]["formal"] == "clip-1 formal caption."
    assert extraction_modes == [True]


def test_process_task_fireworks_path_uses_standard_frame_extraction(tmp_path, monkeypatch):
    task = make_tasks()[0]
    extraction_modes: list[bool] = []
    frame_path = tmp_path / "frame_001.jpg"
    frame_path.write_bytes(b"jpeg")

    monkeypatch.setattr("gemmaclip.main.download_video", lambda task: tmp_path / "video.mp4")
    monkeypatch.setattr(
        "gemmaclip.main.probe_video",
        lambda video_path: VideoMetadata(duration_seconds=20.0, fps=24.0, width=1920, height=1080, frame_count=480),
    )

    def fake_extract_frames(task_id, video_path, metadata, **kwargs):
        extraction_modes.append(kwargs["google_fast"])
        return [ExtractedFrame(path=frame_path, timestamp_seconds=0.5)] * 12

    monkeypatch.setattr("gemmaclip.main.extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        "gemmaclip.main.generate_captions",
        lambda task, frames, **kwargs: {
            "formal": "clip-1 formal caption.",
            "sarcastic": "clip-1 sarcastic caption.",
        },
    )

    result, _ = process_task(
        task,
        workdir=tmp_path,
        env={"FIREWORKS_API_KEY": "fireworks-key", "GEMMACLIP_DISABLE_VERIFIER": "true"},
    )

    assert result["captions"]["formal"] == "clip-1 formal caption."
    assert extraction_modes == [False]


def test_process_task_openrouter_path_uses_fast_frame_extraction(tmp_path, monkeypatch):
    task = make_tasks()[0]
    extraction_modes: list[bool] = []
    frame_path = tmp_path / "frame_001.jpg"
    frame_path.write_bytes(b"jpeg")

    monkeypatch.setattr("gemmaclip.main.download_video", lambda task: tmp_path / "video.mp4")
    monkeypatch.setattr(
        "gemmaclip.main.probe_video",
        lambda video_path: VideoMetadata(duration_seconds=100.0, fps=24.0, width=1920, height=1080, frame_count=2400),
    )

    def fake_extract_frames(task_id, video_path, metadata, **kwargs):
        extraction_modes.append(kwargs["google_fast"])
        return [ExtractedFrame(path=frame_path, timestamp_seconds=5.0)]

    monkeypatch.setattr("gemmaclip.main.extract_frames", fake_extract_frames)
    monkeypatch.setattr(
        "gemmaclip.main.generate_captions",
        lambda task, frames, **kwargs: {
            "formal": "clip-1 formal caption.",
            "sarcastic": "clip-1 sarcastic caption.",
        },
    )

    result, _ = process_task(
        task,
        workdir=tmp_path,
        env={
            "GEMMACLIP_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "openrouter-key",
            "OPENROUTER_MODEL": "openrouter/model",
        },
    )

    assert result["captions"]["formal"] == "clip-1 formal caption."
    assert extraction_modes == [True]
