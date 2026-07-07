from __future__ import annotations

from pathlib import Path

import httpx

from gemmaclip.io import Task, safe_task_id

DEFAULT_VIDEO_DIR = Path("/tmp/gemmaclip/videos")
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=60.0)


def download_video(
    task: Task,
    destination_dir: Path = DEFAULT_VIDEO_DIR,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"{safe_task_id(task.task_id)}.mp4"
    temp_path = output_path.with_suffix(".part")

    try:
        with httpx.stream("GET", task.video_url, follow_redirects=True, timeout=timeout) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
    except httpx.HTTPError as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download video for task {task.task_id}: {exc}") from exc

    temp_path.replace(output_path)
    return output_path
