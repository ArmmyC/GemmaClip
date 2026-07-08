from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path

from gemmaclip.io import safe_task_id
from gemmaclip.video import VideoMetadata

DEFAULT_FRAMES_DIR = Path("/tmp/gemmaclip/frames")
DEFAULT_FRAME_STRATEGY = "aks-lite"
MAX_GEMMA_FRAMES = 12
VALID_FRAME_STRATEGIES = {"uniform", "aks-lite"}


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    path: Path
    timestamp_seconds: float


@dataclass(frozen=True, slots=True)
class _CandidateMetrics:
    frame: ExtractedFrame
    gray_image: object
    sharpness: float
    scene_change: float
    motion: float


def select_frame_count(duration_seconds: float) -> int:
    if duration_seconds <= 45:
        return 12
    if duration_seconds <= 90:
        return 16
    return 20


def select_candidate_frame_count(duration_seconds: float) -> int:
    if duration_seconds <= 30:
        return 18
    if duration_seconds <= 60:
        return 24
    if duration_seconds <= 120:
        return 36
    return 40


def extract_frames(
    task_id: str,
    video_path: str | Path,
    metadata: VideoMetadata,
    *,
    strategy: str = DEFAULT_FRAME_STRATEGY,
    destination_root: Path = DEFAULT_FRAMES_DIR,
    ffmpeg_binary: str = "ffmpeg",
) -> list[ExtractedFrame]:
    if strategy not in VALID_FRAME_STRATEGIES:
        raise ValueError(f"Unsupported frame strategy: {strategy}")

    if strategy == "uniform":
        return extract_uniform_frames(
            task_id,
            video_path,
            metadata,
            destination_root=destination_root,
            ffmpeg_binary=ffmpeg_binary,
        )

    return extract_aks_lite_frames(
        task_id,
        video_path,
        metadata,
        destination_root=destination_root,
        ffmpeg_binary=ffmpeg_binary,
    )


def extract_uniform_frames(
    task_id: str,
    video_path: str | Path,
    metadata: VideoMetadata,
    destination_root: Path = DEFAULT_FRAMES_DIR,
    ffmpeg_binary: str = "ffmpeg",
) -> list[ExtractedFrame]:
    video_file = Path(video_path)
    destination_dir = destination_root / safe_task_id(task_id)
    _prepare_destination_dir(destination_dir)

    frame_count = select_frame_count(metadata.duration_seconds)
    timestamps = _uniform_timestamps(metadata.duration_seconds, frame_count)
    return _extract_frames_at_timestamps(
        video_file,
        destination_dir,
        timestamps,
        prefix="frame",
        ffmpeg_binary=ffmpeg_binary,
    )


def extract_aks_lite_frames(
    task_id: str,
    video_path: str | Path,
    metadata: VideoMetadata,
    destination_root: Path = DEFAULT_FRAMES_DIR,
    ffmpeg_binary: str = "ffmpeg",
) -> list[ExtractedFrame]:
    video_file = Path(video_path)
    destination_dir = destination_root / safe_task_id(task_id)
    _prepare_destination_dir(destination_dir)

    candidate_dir = destination_dir / "_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_count = select_candidate_frame_count(metadata.duration_seconds)
    candidate_timestamps = _uniform_timestamps(metadata.duration_seconds, candidate_count)
    candidates = _extract_frames_at_timestamps(
        video_file,
        candidate_dir,
        candidate_timestamps,
        prefix="candidate",
        ffmpeg_binary=ffmpeg_binary,
    )
    selected_candidates = select_aks_lite_frames(candidates, max_frames=MAX_GEMMA_FRAMES)

    selected_frames: list[ExtractedFrame] = []
    for index, frame in enumerate(selected_candidates, start=1):
        destination = destination_dir / f"frame_{index:03d}.jpg"
        shutil.copy2(frame.path, destination)
        selected_frames.append(ExtractedFrame(path=destination, timestamp_seconds=frame.timestamp_seconds))

    shutil.rmtree(candidate_dir, ignore_errors=True)
    return selected_frames


def select_aks_lite_frames(
    candidates: list[ExtractedFrame],
    max_frames: int = MAX_GEMMA_FRAMES,
) -> list[ExtractedFrame]:
    if max_frames <= 0:
        return []
    if not candidates:
        return []

    metrics = _load_candidate_metrics(candidates)
    if not metrics:
        return []
    if len(metrics) <= max_frames:
        return [metric.frame for metric in metrics]

    selected_indices: list[int] = []
    for bin_index in range(max_frames):
        start = (bin_index * len(metrics)) // max_frames
        end = ((bin_index + 1) * len(metrics)) // max_frames
        bin_metrics = metrics[start:end]
        if not bin_metrics:
            continue
        if bin_index == 0:
            best_index = start
        elif bin_index == max_frames - 1:
            best_index = end - 1
        else:
            best_index = max(
                range(start, end),
                key=lambda index: _quality_score(metrics[index], metrics),
            )
        selected_indices.append(best_index)

    unique_indices = sorted(set(selected_indices))
    return [metrics[index].frame for index in unique_indices]


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

    Image, ImageDraw, ImageFont, ImageOps, _, _ = _load_pillow()

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


def _prepare_destination_dir(destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in destination_dir.glob("*.jpg"):
        stale_frame.unlink()
    shutil.rmtree(destination_dir / "_candidates", ignore_errors=True)


def _extract_frames_at_timestamps(
    video_path: Path,
    destination_dir: Path,
    timestamps: list[float],
    *,
    prefix: str,
    ffmpeg_binary: str,
) -> list[ExtractedFrame]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in destination_dir.glob("*.jpg"):
        stale_frame.unlink()

    frames: list[ExtractedFrame] = []
    for index, timestamp in enumerate(timestamps, start=1):
        output_path = destination_dir / f"{prefix}_{index:03d}.jpg"
        _extract_frame(video_path, output_path, timestamp, ffmpeg_binary=ffmpeg_binary)
        frames.append(ExtractedFrame(path=output_path, timestamp_seconds=timestamp))
    return frames


def _load_candidate_metrics(candidates: list[ExtractedFrame]) -> list[_CandidateMetrics]:
    _, _, _, _, ImageChops, ImageStat = _load_pillow()

    metrics: list[_CandidateMetrics] = []
    previous_gray = None
    for candidate in candidates:
        gray = _open_candidate_preview(candidate.path)
        if gray is None:
            continue
        diff_mean = 1.0 if previous_gray is None else _mean_difference(gray, previous_gray, ImageChops, ImageStat)
        diff_rms = 1.0 if previous_gray is None else _rms_difference(gray, previous_gray, ImageChops, ImageStat)
        metrics.append(
            _CandidateMetrics(
                frame=candidate,
                gray_image=gray,
                sharpness=_laplacian_variance(gray),
                scene_change=diff_mean,
                motion=diff_rms,
            )
        )
        previous_gray = gray
    return metrics


def _quality_score(
    metric: _CandidateMetrics,
    metrics: list[_CandidateMetrics],
) -> float:
    sharpness_score = _normalize_metric(metric.sharpness, [item.sharpness for item in metrics])
    scene_score = _normalize_metric(metric.scene_change, [item.scene_change for item in metrics])
    motion_score = _normalize_metric(metric.motion, [item.motion for item in metrics])
    return 0.40 * sharpness_score + 0.30 * scene_score + 0.30 * motion_score


def _open_candidate_preview(frame_path: Path):
    Image, _, _, _, _, _ = _load_pillow()
    try:
        with Image.open(frame_path) as image:
            preview = image.convert("L")
            preview.thumbnail((96, 96))
            return preview.copy()
    except (OSError, ValueError):
        return None


def _mean_difference(image_a, image_b, ImageChops, ImageStat) -> float:
    difference = ImageChops.difference(image_a, image_b)
    stat = ImageStat.Stat(difference)
    return float(stat.mean[0]) / 255.0


def _rms_difference(image_a, image_b, ImageChops, ImageStat) -> float:
    difference = ImageChops.difference(image_a, image_b)
    stat = ImageStat.Stat(difference)
    return float(stat.rms[0]) / 255.0


def _laplacian_variance(gray_image) -> float:
    width, height = gray_image.size
    if width < 3 or height < 3:
        return 0.0

    pixels = gray_image.load()
    responses: list[float] = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            response = (
                float(pixels[x - 1, y])
                + float(pixels[x + 1, y])
                + float(pixels[x, y - 1])
                + float(pixels[x, y + 1])
                - 4.0 * float(pixels[x, y])
            )
            responses.append(response)

    if not responses:
        return 0.0
    mean = sum(responses) / len(responses)
    variance = sum((response - mean) ** 2 for response in responses) / len(responses)
    return variance


def _normalize_metric(value: float, values: list[float]) -> float:
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def _load_pillow():
    try:
        from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps, ImageStat
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required for frame debug artifacts and AKS-lite frame selection. "
            "Install dependencies with `python -m pip install -e .` or install Pillow directly."
        ) from exc

    return Image, ImageDraw, ImageFont, ImageOps, ImageChops, ImageStat


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
