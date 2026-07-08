from __future__ import annotations

import subprocess
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path
import shutil

from gemmaclip.io import safe_task_id
from gemmaclip.video import VideoMetadata

DEFAULT_FRAMES_DIR = Path("/tmp/gemmaclip/frames")


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    path: Path
    timestamp_seconds: float


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
) -> list[ExtractedFrame]:
    video_file = Path(video_path)
    destination_dir = destination_root / safe_task_id(task_id)
    destination_dir.mkdir(parents=True, exist_ok=True)

    for stale_frame in destination_dir.glob("*.jpg"):
        stale_frame.unlink()

    frame_count = select_frame_count(metadata.duration_seconds)
    timestamps = _uniform_timestamps(metadata.duration_seconds, frame_count)

    output_frames: list[ExtractedFrame] = []
    for index, timestamp in enumerate(timestamps, start=1):
        output_path = destination_dir / f"frame_{index:03d}.jpg"
        _extract_frame(video_file, output_path, timestamp, ffmpeg_binary=ffmpeg_binary)
        output_frames.append(ExtractedFrame(path=output_path, timestamp_seconds=timestamp))
    return output_frames


def export_debug_artifacts(
    task_id: str,
    frames: list[ExtractedFrame],
    debug_dir: str | Path,
) -> None:
    debug_root = Path(debug_dir)
    debug_root.mkdir(parents=True, exist_ok=True)

    task_name = safe_task_id(task_id)
    task_debug_dir = debug_root / task_name
    task_debug_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in task_debug_dir.glob("*.jpg"):
        stale_frame.unlink()

    copied_frames: list[Path] = []
    for frame in frames:
        destination = task_debug_dir / frame.path.name
        shutil.copy2(frame.path, destination)
        copied_frames.append(destination)

    contact_sheet_path = debug_root / f"{task_name}_contact_sheet.jpg"
    generate_contact_sheet(copied_frames, contact_sheet_path)


def generate_contact_sheet(frame_paths: list[Path], output_path: str | Path) -> None:
    if not frame_paths:
        raise ValueError("Cannot generate a contact sheet without frames.")

    Image, ImageDraw, ImageFont, ImageOps = _load_pillow()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    thumb_width = 320
    thumb_height = 180
    label_height = 24
    padding = 16
    columns = max(1, min(4, ceil(sqrt(len(frame_paths)))))
    rows = ceil(len(frame_paths) / columns)
    cell_width = thumb_width
    cell_height = thumb_height + label_height

    sheet_width = padding + (columns * cell_width) + ((columns - 1) * padding) + padding
    sheet_height = padding + (rows * cell_height) + ((rows - 1) * padding) + padding

    contact_sheet = Image.new("RGB", (sheet_width, sheet_height), color="white")
    draw = ImageDraw.Draw(contact_sheet)
    font = ImageFont.load_default()

    for index, frame_path in enumerate(frame_paths):
        row = index // columns
        column = index % columns
        origin_x = padding + column * (cell_width + padding)
        origin_y = padding + row * (cell_height + padding)

        with Image.open(frame_path) as image:
            thumbnail = ImageOps.contain(image.convert("RGB"), (thumb_width, thumb_height))

        thumb_x = origin_x + (thumb_width - thumbnail.width) // 2
        thumb_y = origin_y + (thumb_height - thumbnail.height) // 2
        contact_sheet.paste(thumbnail, (thumb_x, thumb_y))

        label = frame_path.name
        draw.text((origin_x, origin_y + thumb_height + 4), label, fill="black", font=font)

    contact_sheet.save(output, format="JPEG", quality=90)


def _load_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required for --debug-dir contact sheets. Install dependencies with `python -m pip install -e .` "
            "or install Pillow directly."
        ) from exc

    return Image, ImageDraw, ImageFont, ImageOps


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
