from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import httpx

from gemmaclip.io import Task, safe_task_id

DEFAULT_VIDEO_DIR = Path(tempfile.gettempdir()) / "gemmaclip" / "videos"
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
DEFAULT_DOWNLOAD_CHUNK_SIZE = 1024 * 1024

LOGGER = logging.getLogger("gemmaclip.download")


def download_video(
    task: Task,
    destination_dir: Path = DEFAULT_VIDEO_DIR,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    chunk_size: int = DEFAULT_DOWNLOAD_CHUNK_SIZE,
) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"{safe_task_id(task.task_id)}.mp4"
    temp_path = output_path.with_suffix(".part")
    started_at = time.monotonic()
    total_bytes = 0

    try:
        with httpx.stream("GET", task.video_url, follow_redirects=True, timeout=timeout) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)
                        total_bytes += len(chunk)
    except httpx.HTTPError as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download video for task {task.task_id}: {exc}") from exc

    temp_path.replace(output_path)
    LOGGER.info(
        "Downloaded task %s to %s in %.2fs (%s bytes).",
        task.task_id,
        output_path,
        time.monotonic() - started_at,
        total_bytes,
    )
    return output_path
