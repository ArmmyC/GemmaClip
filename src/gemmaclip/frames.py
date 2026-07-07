from __future__ import annotations

import subprocess
from pathlib import Path

from gemmaclip.io import safe_task_id
from gemmaclip.video import VideoMetadata

DEFAULT_FRAMES_DIR = Path("/tmp/gemmaclip/frames")


def select_frame_count(duration_seconds: float) -> int:
    if duration_seconds <= 45:
        return 12
    if duration_seconds <= 90:
        return 16
    return 20


def extract_uniform_frames(
    task_id: str,
    video_path: str | Path,
    metadata: VideoMetadata,
    destination_root: Path = DEFAULT_FRAMES_DIR,
    ffmpeg_binary: str = "ffmpeg",
) -> list[Path]:
    video_file = Path(video_path)
    destination_dir = destination_root / safe_task_id(task_id)
    destination_dir.mkdir(parents=True, exist_ok=True)

    for stale_frame in destination_dir.glob("*.jpg"):
        stale_frame.unlink()

    frame_count = select_frame_count(metadata.duration_seconds)
    timestamps = _uniform_timestamps(metadata.duration_seconds, frame_count)

    output_paths: list[Path] = []
    for index, timestamp in enumerate(timestamps, start=1):
        output_path = destination_dir / f"frame_{index:03d}.jpg"
        _extract_frame(video_file, output_path, timestamp, ffmpeg_binary=ffmpeg_binary)
        output_paths.append(output_path)
    return output_paths


def _uniform_timestamps(duration_seconds: float, frame_count: int) -> list[float]:
    if duration_seconds <= 0:
        raise ValueError("Video duration must be positive.")
    if frame_count <= 0:
        raise ValueError("Frame count must be positive.")

    timestamps: list[float] = []
    upper_bound = max(duration_seconds - 0.001, 0.0)
    for index in range(frame_count):
        timestamp = ((index + 0.5) / frame_count) * duration_seconds
        timestamps.append(min(timestamp, upper_bound))
    return timestamps


def _extract_frame(
    video_path: Path,
    output_path: Path,
    timestamp: float,
    ffmpeg_binary: str = "ffmpeg",
) -> None:
    command = [
        ffmpeg_binary,
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not available on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed to extract frame at {timestamp:.3f}s from {video_path}: {exc.stderr.strip()}"
        ) from exc
