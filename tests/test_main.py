from __future__ import annotations

from pathlib import Path

from gemmaclip.io import Task
from gemmaclip.main import build_progress_results, process_tasks, should_fill_remaining_with_fallbacks


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
        max_runtime_seconds=540.0,
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
        max_runtime_seconds=540.0,
        start_time=0.0,
        now_fn=fake_now,
    )

    assert results[0]["task_id"] == "clip-1"
    assert results[0]["captions"]["formal"] == "clip-1 formal caption."
    assert results[1]["captions"]["formal"]
    assert results[2]["captions"]["formal"]
    assert results[1]["captions"]["formal"] != "clip-2 formal caption."
    assert len(write_calls) == 2


def test_should_fill_remaining_with_fallbacks_uses_runtime_estimate():
    assert should_fill_remaining_with_fallbacks(
        0.0,
        completed_count=4,
        remaining_task_count=4,
        max_runtime_seconds=540.0,
        now_fn=lambda: 520.0,
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
