from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil, sqrt
from pathlib import Path

from gemmaclip.io import safe_task_id
from gemmaclip.video import VideoMetadata

LOGGER = logging.getLogger("gemmaclip")
DEFAULT_FRAMES_DIR = Path("/tmp/gemmaclip/frames")
DEFAULT_FRAME_STRATEGY = "aks-lite"
MAX_GEMMA_FRAMES = 12
GOOGLE_FAST_FRAME_COUNT = 6
GOOGLE_FAST_FRAME_WIDTH = 512
GOOGLE_FAST_FRAME_RATIOS = (0.05, 0.20, 0.35, 0.55, 0.75, 0.95)
FIREWORKS_JUDGE_FRAME_RATIOS = (0.05, 0.23, 0.41, 0.59, 0.77, 0.95)
FIREWORKS_HYBRID_ANCHOR_RATIOS = (0.05, 0.35, 0.65, 0.95)
FIREWORKS_HYBRID_BACKUP_RATIOS = (0.20, 0.50, 0.80)
FIREWORKS_SCAN_FRAME_COUNT = 16
FIREWORKS_SCAN_FRAME_WIDTH = 96
FIREWORKS_FRAME_MODE_ENV = "GEMMACLIP_FIREWORKS_FRAME_MODE"
VALID_FRAME_STRATEGIES = {"uniform", "aks-lite"}


@dataclass(frozen=True, slots=True)
class ExtractedFrame:
    path: Path
    timestamp_seconds: float
    frame_role: str = ""
    change_score: float | None = None


@dataclass(frozen=True, slots=True)
class FrameCandidate:
    timestamp_seconds: float
    change_score: float


@dataclass(frozen=True, slots=True)
class FireworksFrameSelection:
    mode: str
    duration_seconds: float
    anchors: tuple[FrameCandidate, ...]
    dynamic: tuple[FrameCandidate, ...]
    final_timestamps: tuple[float, ...]
    uniform_fallback_used: bool = False


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
    google_fast: bool = False,
    fireworks_judge: bool = False,
    command_timeout_seconds: float = 15.0,
    env: Mapping[str, str] | None = None,
) -> list[ExtractedFrame]:
    if fireworks_judge:
        return extract_fireworks_judge_frames(
            task_id,
            video_path,
            metadata,
            destination_root=destination_root,
            ffmpeg_binary=ffmpeg_binary,
            command_timeout_seconds=command_timeout_seconds,
            env=env,
        )

    if google_fast:
        return extract_google_fast_frames(
            task_id,
            video_path,
            metadata,
            destination_root=destination_root,
            ffmpeg_binary=ffmpeg_binary,
            command_timeout_seconds=command_timeout_seconds,
        )

    if strategy not in VALID_FRAME_STRATEGIES:
        raise ValueError(f"Unsupported frame strategy: {strategy}")

    if strategy == "uniform":
        return extract_uniform_frames(
            task_id,
            video_path,
            metadata,
            destination_root=destination_root,
            ffmpeg_binary=ffmpeg_binary,
            command_timeout_seconds=command_timeout_seconds,
        )

    return extract_aks_lite_frames(
        task_id,
        video_path,
        metadata,
        destination_root=destination_root,
        ffmpeg_binary=ffmpeg_binary,
        command_timeout_seconds=command_timeout_seconds,
    )


def extract_google_fast_frames(
    task_id: str,
    video_path: str | Path,
    metadata: VideoMetadata,
    destination_root: Path = DEFAULT_FRAMES_DIR,
    ffmpeg_binary: str = "ffmpeg",
    max_width: int = GOOGLE_FAST_FRAME_WIDTH,
    command_timeout_seconds: float = 15.0,
) -> list[ExtractedFrame]:
    video_file = Path(video_path)
    destination_dir = destination_root / safe_task_id(task_id)
    _prepare_destination_dir(destination_dir)

    timestamps = _google_fast_timestamps(metadata.duration_seconds)
    return _extract_frames_at_timestamps(
        video_file,
        destination_dir,
        timestamps,
        prefix="frame",
        ffmpeg_binary=ffmpeg_binary,
        output_width=max_width,
        command_timeout_seconds=command_timeout_seconds,
    )


def extract_fireworks_judge_frames(
    task_id: str,
    video_path: str | Path,
    metadata: VideoMetadata,
    destination_root: Path = DEFAULT_FRAMES_DIR,
    ffmpeg_binary: str = "ffmpeg",
    max_width: int = GOOGLE_FAST_FRAME_WIDTH,
    command_timeout_seconds: float = 15.0,
    env: Mapping[str, str] | None = None,
) -> list[ExtractedFrame]:
    video_file = Path(video_path)
    destination_dir = destination_root / safe_task_id(task_id)
    _prepare_destination_dir(destination_dir)

    mode = resolve_fireworks_frame_mode(env)
    selection = select_fireworks_uniform_frame_selection(metadata.duration_seconds)
    if mode == "hybrid":
        try:
            scan_frames = extract_fireworks_scan_frames(
                task_id,
                video_file,
                metadata,
                destination_root=destination_root,
                ffmpeg_binary=ffmpeg_binary,
                command_timeout_seconds=command_timeout_seconds,
            )
            scan_candidates = score_fireworks_scan_frames(scan_frames)
            selection = select_fireworks_hybrid_timestamps(
                metadata.duration_seconds,
                [candidate.timestamp_seconds for candidate in scan_candidates],
                [candidate.change_score for candidate in scan_candidates],
            )
        except Exception as exc:
            LOGGER.warning(
                "Fireworks hybrid frame scan failed for task=%s; falling back to uniform frames: %s",
                task_id,
                exc,
            )
            selection = select_fireworks_uniform_frame_selection(metadata.duration_seconds, fallback_used=True)
        finally:
            shutil.rmtree(destination_dir / "_fireworks_scan", ignore_errors=True)

    try:
        frames = _extract_frames_at_timestamps(
            video_file,
            destination_dir,
            list(selection.final_timestamps),
            prefix="frame",
            ffmpeg_binary=ffmpeg_binary,
            output_width=max_width,
            command_timeout_seconds=command_timeout_seconds,
        )
    except Exception as exc:
        if mode != "hybrid" or selection.uniform_fallback_used:
            raise
        LOGGER.warning(
            "Fireworks hybrid final frame extraction failed for task=%s; falling back to uniform frames: %s",
            task_id,
            exc,
        )
        selection = select_fireworks_uniform_frame_selection(metadata.duration_seconds, fallback_used=True)
        frames = _extract_frames_at_timestamps(
            video_file,
            destination_dir,
            list(selection.final_timestamps),
            prefix="frame",
            ffmpeg_binary=ffmpeg_binary,
            output_width=max_width,
            command_timeout_seconds=command_timeout_seconds,
        )

    roles = _frame_roles_by_timestamp(selection)
    scores = {round(item.timestamp_seconds, 3): item.change_score for item in selection.dynamic}
    default_role = "uniform_fallback" if selection.uniform_fallback_used else ("uniform" if selection.mode == "uniform" else "")
    annotated_frames = [
        ExtractedFrame(
            path=frame.path,
            timestamp_seconds=frame.timestamp_seconds,
            frame_role=roles.get(round(frame.timestamp_seconds, 3), default_role),
            change_score=scores.get(round(frame.timestamp_seconds, 3)),
        )
        for frame in frames
    ]
    _log_fireworks_frame_selection(task_id, selection)
    return annotated_frames


def resolve_fireworks_frame_mode(env: Mapping[str, str] | None = None) -> str:
    values = env if env is not None else os.environ
    raw_mode = values.get(FIREWORKS_FRAME_MODE_ENV, "hybrid").strip().lower()
    if raw_mode in {"hybrid", "uniform"}:
        return raw_mode
    LOGGER.warning("Invalid %s=%r; using hybrid Fireworks frame mode.", FIREWORKS_FRAME_MODE_ENV, raw_mode)
    return "hybrid"


def select_fireworks_uniform_frame_selection(
    duration_seconds: float,
    *,
    fallback_used: bool = False,
) -> FireworksFrameSelection:
    timestamps = _safe_fixed_ratio_timestamps(duration_seconds, FIREWORKS_JUDGE_FRAME_RATIOS)
    return FireworksFrameSelection(
        mode="uniform",
        duration_seconds=duration_seconds,
        anchors=tuple(),
        dynamic=tuple(),
        final_timestamps=tuple(timestamps),
        uniform_fallback_used=fallback_used,
    )


def select_fireworks_hybrid_timestamps(
    duration_seconds: float,
    candidate_timestamps: Sequence[float],
    change_scores: Sequence[float],
) -> FireworksFrameSelection:
    """Select Fireworks frames using anchors plus high-change candidate timestamps.

    Change score i is associated with candidate[i], the later frame in the
    consecutive pair that produced the difference.
    """
    if duration_seconds <= 0 or len(candidate_timestamps) != len(change_scores):
        return select_fireworks_uniform_frame_selection(duration_seconds, fallback_used=True)

    anchors = tuple(
        FrameCandidate(timestamp, 0.0)
        for timestamp in _fixed_ratio_timestamps(duration_seconds, FIREWORKS_HYBRID_ANCHOR_RATIOS)
    )
    selected_dynamic: list[FrameCandidate] = []
    selected_timestamps = [anchor.timestamp_seconds for anchor in anchors]
    minimum_separation = max(0.5, 0.08 * duration_seconds)

    ranked_candidates = sorted(
        (
            FrameCandidate(round(_clamp_timestamp(timestamp, duration_seconds), 3), float(score))
            for timestamp, score in zip(candidate_timestamps, change_scores)
            if timestamp >= 0
        ),
        key=lambda candidate: (-candidate.change_score, candidate.timestamp_seconds),
    )
    for candidate in ranked_candidates:
        if len(selected_dynamic) >= 2:
            break
        if _is_separated(candidate.timestamp_seconds, selected_timestamps, minimum_separation):
            selected_dynamic.append(candidate)
            selected_timestamps.append(candidate.timestamp_seconds)

    for backup_timestamp in _fixed_ratio_timestamps(duration_seconds, FIREWORKS_HYBRID_BACKUP_RATIOS):
        if len(selected_dynamic) >= 2:
            break
        if _is_separated(backup_timestamp, selected_timestamps, minimum_separation):
            selected_dynamic.append(FrameCandidate(backup_timestamp, 0.0))
            selected_timestamps.append(backup_timestamp)

    if len(selected_dynamic) < 2:
        for backup_timestamp in _uniform_timestamps(duration_seconds, 6):
            if len(selected_dynamic) >= 2:
                break
            rounded = round(_clamp_timestamp(backup_timestamp, duration_seconds), 3)
            if not _timestamp_already_selected(rounded, selected_timestamps):
                selected_dynamic.append(FrameCandidate(rounded, 0.0))
                selected_timestamps.append(rounded)

    final_timestamps = sorted(round(timestamp, 3) for timestamp in selected_timestamps)
    if len(set(final_timestamps)) != 6:
        final_timestamps = _dedupe_and_fill_timestamps(final_timestamps, duration_seconds)
    if len(final_timestamps) != 6:
        return select_fireworks_uniform_frame_selection(duration_seconds, fallback_used=True)

    return FireworksFrameSelection(
        mode="hybrid",
        duration_seconds=duration_seconds,
        anchors=anchors,
        dynamic=tuple(selected_dynamic[:2]),
        final_timestamps=tuple(final_timestamps),
        uniform_fallback_used=False,
    )


def extract_fireworks_scan_frames(
    task_id: str,
    video_path: str | Path,
    metadata: VideoMetadata,
    *,
    destination_root: Path = DEFAULT_FRAMES_DIR,
    ffmpeg_binary: str = "ffmpeg",
    command_timeout_seconds: float = 15.0,
) -> list[ExtractedFrame]:
    candidate_dir = destination_root / safe_task_id(task_id) / "_fireworks_scan"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    timestamps = _fixed_range_timestamps(metadata.duration_seconds, 0.05, 0.95, FIREWORKS_SCAN_FRAME_COUNT)
    return _extract_frames_at_timestamps(
        Path(video_path),
        candidate_dir,
        timestamps,
        prefix="scan",
        ffmpeg_binary=ffmpeg_binary,
        output_width=FIREWORKS_SCAN_FRAME_WIDTH,
        command_timeout_seconds=command_timeout_seconds,
    )


def score_fireworks_scan_frames(frames: Sequence[ExtractedFrame]) -> list[FrameCandidate]:
    Image, _, _, _, _, _ = _load_pillow()
    candidates: list[FrameCandidate] = []
    previous = None
    for frame in frames:
        with Image.open(frame.path) as image:
            current = image.convert("L").resize((96, 54))
        if previous is not None:
            candidates.append(
                FrameCandidate(
                    timestamp_seconds=round(frame.timestamp_seconds, 3),
                    change_score=compute_frame_change_score(previous, current),
                )
            )
        previous = current.copy()
    return candidates


def compute_frame_change_score(previous_image, current_image) -> float:
    _, _, _, _, ImageChops, ImageStat = _load_pillow()
    previous = previous_image.convert("L").resize((96, 54))
    current = current_image.convert("L").resize((96, 54))
    difference = ImageChops.difference(previous, current)
    stat = ImageStat.Stat(difference)
    return float(stat.mean[0])


def _frame_roles_by_timestamp(selection: FireworksFrameSelection) -> dict[float, str]:
    roles: dict[float, str] = {}
    for anchor in selection.anchors:
        roles[round(anchor.timestamp_seconds, 3)] = "anchor"
    for dynamic in selection.dynamic:
        roles[round(dynamic.timestamp_seconds, 3)] = "dynamic"
    return roles


def _log_fireworks_frame_selection(task_id: str, selection: FireworksFrameSelection) -> None:
    LOGGER.info(
        "Fireworks frames task=%s mode=%s anchors=%s dynamic=%s scores=%s fallback=%s",
        task_id,
        selection.mode,
        _format_timestamps([item.timestamp_seconds for item in selection.anchors]),
        _format_timestamps([item.timestamp_seconds for item in selection.dynamic]),
        [round(item.change_score, 1) for item in selection.dynamic],
        str(selection.uniform_fallback_used).lower(),
    )


def _format_timestamps(timestamps: Sequence[float]) -> list[float]:
    return [round(timestamp, 2) for timestamp in timestamps]


def _is_separated(timestamp: float, selected_timestamps: Sequence[float], minimum_separation: float) -> bool:
    return all(abs(timestamp - selected) >= minimum_separation for selected in selected_timestamps)


def _timestamp_already_selected(timestamp: float, selected_timestamps: Sequence[float]) -> bool:
    return any(round(selected, 3) == round(timestamp, 3) for selected in selected_timestamps)


def _dedupe_and_fill_timestamps(timestamps: Sequence[float], duration_seconds: float) -> list[float]:
    deduped = sorted(set(round(timestamp, 3) for timestamp in timestamps))
    for timestamp in _safe_fixed_ratio_timestamps(duration_seconds, FIREWORKS_JUDGE_FRAME_RATIOS):
        if len(deduped) >= 6:
            break
        if timestamp not in deduped:
            deduped.append(timestamp)
    return sorted(deduped)[:6]


def extract_uniform_frames(
    task_id: str,
    video_path: str | Path,
    metadata: VideoMetadata,
    destination_root: Path = DEFAULT_FRAMES_DIR,
    ffmpeg_binary: str = "ffmpeg",
    command_timeout_seconds: float = 15.0,
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
        command_timeout_seconds=command_timeout_seconds,
    )


def extract_aks_lite_frames(
    task_id: str,
    video_path: str | Path,
    metadata: VideoMetadata,
    destination_root: Path = DEFAULT_FRAMES_DIR,
    ffmpeg_binary: str = "ffmpeg",
    command_timeout_seconds: float = 15.0,
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
        command_timeout_seconds=command_timeout_seconds,
    )
    selected_candidates = select_aks_lite_frames(candidates, max_frames=MAX_GEMMA_FRAMES)

    selected_frames: list[ExtractedFrame] = []
    for index, frame in enumerate(selected_candidates, start=1):
        destination = destination_dir / f"frame_{index:03d}.jpg"
        shutil.copy2(frame.path, destination)
        selected_frames.append(ExtractedFrame(path=destination, timestamp_seconds=frame.timestamp_seconds))

    shutil.rmtree(candidate_dir, ignore_errors=True)
    return selected_frames


def extract_configured_frames(
    task_id: str,
    video_path: str | Path,
    metadata: VideoMetadata,
    *,
    strategy: str = "hybrid",
    total_frames: int = 6,
    anchor_count: int = 4,
    high_change_count: int = 2,
    min_spacing_seconds: float = 1.0,
    change_sensitivity: float = 0.5,
    destination_root: Path = DEFAULT_FRAMES_DIR,
    ffmpeg_binary: str = "ffmpeg",
    command_timeout_seconds: float = 15.0,
) -> list[ExtractedFrame]:
    """Extract a user-configured frame set using the shared media primitives.

    The web Lab uses this entry point; the competition presets continue to use
    ``extract_frames`` unchanged.  Candidate scoring and timestamp extraction
    are deliberately shared with AKS-Lite and the existing Fireworks hybrid
    selector rather than being reimplemented in the web layer.
    """
    if strategy not in {"uniform", "aks-lite", "hybrid"}:
        raise ValueError("Unsupported frame strategy.")
    if total_frames < 2 or total_frames > 16:
        raise ValueError("Total frames must be between 2 and 16.")
    if anchor_count < 0 or high_change_count < 0 or anchor_count + high_change_count > total_frames:
        raise ValueError("Anchor and high-change counts must fit within total frames.")
    if min_spacing_seconds <= 0 or min_spacing_seconds > 5:
        raise ValueError("Minimum frame spacing must be between 0 and 5 seconds.")
    if not 0 <= change_sensitivity <= 1:
        raise ValueError("Change sensitivity must be between 0 and 1.")

    video_file = Path(video_path)
    destination_dir = destination_root / safe_task_id(task_id)
    _prepare_destination_dir(destination_dir)

    if strategy == "uniform":
        raw = _extract_frames_at_timestamps(
            video_file,
            destination_dir,
            _uniform_timestamps(metadata.duration_seconds, total_frames),
            prefix="frame",
            ffmpeg_binary=ffmpeg_binary,
            command_timeout_seconds=command_timeout_seconds,
        )
        return [ExtractedFrame(frame.path, frame.timestamp_seconds, "uniform", 0.0) for frame in raw]

    candidate_dir = destination_dir / "_lab_candidates"
    candidate_count = max(total_frames * 3, 18)
    candidates = _extract_frames_at_timestamps(
        video_file,
        candidate_dir,
        _uniform_timestamps(metadata.duration_seconds, candidate_count),
        prefix="candidate",
        ffmpeg_binary=ffmpeg_binary,
        command_timeout_seconds=command_timeout_seconds,
    )
    metrics = _load_candidate_metrics(candidates)
    if not metrics:
        raise RuntimeError("Frame extraction produced no readable candidates.")

    if strategy == "aks-lite":
        selected = select_aks_lite_frames([metric.frame for metric in metrics], max_frames=total_frames)
        roles = {id(frame): "dynamic" for frame in selected}
    else:
        selected, roles = _select_configured_hybrid_frames(
            metrics,
            duration_seconds=metadata.duration_seconds,
            total_frames=total_frames,
            anchor_count=anchor_count,
            high_change_count=high_change_count,
            min_spacing_seconds=min_spacing_seconds,
            change_sensitivity=change_sensitivity,
        )

    if len(selected) < 2:
        raise RuntimeError("Frame extraction did not produce enough usable frames.")
    result: list[ExtractedFrame] = []
    for index, frame in enumerate(sorted(selected, key=lambda item: item.timestamp_seconds), start=1):
        destination = destination_dir / f"frame_{index:03d}.jpg"
        shutil.copy2(frame.path, destination)
        result.append(ExtractedFrame(destination, frame.timestamp_seconds, roles.get(id(frame), "uniform"), _metric_score(frame, metrics)))
    shutil.rmtree(candidate_dir, ignore_errors=True)
    return result


def _select_configured_hybrid_frames(
    metrics: Sequence[_CandidateMetrics],
    *,
    duration_seconds: float,
    total_frames: int,
    anchor_count: int,
    high_change_count: int,
    min_spacing_seconds: float,
    change_sensitivity: float,
) -> tuple[list[ExtractedFrame], dict[int, str]]:
    anchors = _uniform_timestamps(duration_seconds, anchor_count) if anchor_count else []
    selected: list[ExtractedFrame] = []
    roles: dict[int, str] = {}
    for timestamp in anchors:
        nearest = min(metrics, key=lambda item: abs(item.frame.timestamp_seconds - timestamp))
        if all(abs(nearest.frame.timestamp_seconds - item.timestamp_seconds) >= min_spacing_seconds for item in selected):
            selected.append(nearest.frame)
            roles[id(nearest.frame)] = "anchor"

    ranked = sorted(
        metrics,
        key=lambda item: _configured_quality_score(item, list(metrics), change_sensitivity),
        reverse=True,
    )
    wanted_dynamic = min(high_change_count, max(0, total_frames - len(selected)))
    for metric in ranked:
        if len([item for item in selected if roles.get(id(item)) == "dynamic"]) >= wanted_dynamic:
            break
        if all(abs(metric.frame.timestamp_seconds - item.timestamp_seconds) >= min_spacing_seconds for item in selected):
            selected.append(metric.frame)
            roles[id(metric.frame)] = "dynamic"

    if len(selected) < total_frames:
        for metric in sorted(metrics, key=lambda item: item.frame.timestamp_seconds):
            if len(selected) >= total_frames:
                break
            if not any(abs(metric.frame.timestamp_seconds - item.timestamp_seconds) < min_spacing_seconds for item in selected):
                selected.append(metric.frame)
                roles[id(metric.frame)] = "uniform"
    return selected[:total_frames], roles


def _metric_score(frame: ExtractedFrame, metrics: Sequence[_CandidateMetrics]) -> float:
    for metric in metrics:
        if metric.frame is frame:
            return _quality_score(metric, list(metrics))
    return float(frame.change_score or 0.0)


def _configured_quality_score(
    metric: _CandidateMetrics,
    metrics: list[_CandidateMetrics],
    change_sensitivity: float,
) -> float:
    """Weight visual quality versus temporal change without a global multiplier."""
    sharpness_score = _normalize_metric(metric.sharpness, [item.sharpness for item in metrics])
    scene_score = _normalize_metric(metric.scene_change, [item.scene_change for item in metrics])
    motion_score = _normalize_metric(metric.motion, [item.motion for item in metrics])
    change_weight = 0.4 + (0.4 * change_sensitivity)
    quality_weight = 1.0 - change_weight
    return quality_weight * sharpness_score + change_weight * ((scene_score + motion_score) / 2.0)


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
    output_width: int | None = None,
    command_timeout_seconds: float = 15.0,
) -> list[ExtractedFrame]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in destination_dir.glob("*.jpg"):
        stale_frame.unlink()

    frames: list[ExtractedFrame] = []
    for index, timestamp in enumerate(timestamps, start=1):
        output_path = destination_dir / f"{prefix}_{index:03d}.jpg"
        _extract_frame(
            video_path,
            output_path,
            timestamp,
            ffmpeg_binary=ffmpeg_binary,
            output_width=output_width,
            timeout_seconds=command_timeout_seconds,
        )
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


def _google_fast_timestamps(duration_seconds: float) -> list[float]:
    return _fixed_ratio_timestamps(duration_seconds, GOOGLE_FAST_FRAME_RATIOS[:GOOGLE_FAST_FRAME_COUNT])


def _fixed_ratio_timestamps(duration_seconds: float, ratios: tuple[float, ...]) -> list[float]:
    if duration_seconds <= 0:
        raise ValueError("Video duration must be positive.")

    return [round(_clamp_timestamp(duration_seconds * ratio, duration_seconds), 3) for ratio in ratios]


def _safe_fixed_ratio_timestamps(duration_seconds: float, ratios: tuple[float, ...]) -> list[float]:
    safe_duration = duration_seconds if duration_seconds > 0 else 1.0
    return _fixed_ratio_timestamps(safe_duration, ratios)


def _fixed_range_timestamps(duration_seconds: float, start_ratio: float, end_ratio: float, frame_count: int) -> list[float]:
    if duration_seconds <= 0:
        raise ValueError("Video duration must be positive.")
    if frame_count <= 0:
        raise ValueError("Frame count must be positive.")
    if frame_count == 1:
        return [round(_clamp_timestamp(duration_seconds * start_ratio, duration_seconds), 3)]

    span = end_ratio - start_ratio
    return [
        round(_clamp_timestamp(duration_seconds * (start_ratio + span * index / (frame_count - 1)), duration_seconds), 3)
        for index in range(frame_count)
    ]


def _clamp_timestamp(timestamp: float, duration_seconds: float) -> float:
    upper_bound = max(duration_seconds - 0.001, 0.0)
    return min(max(float(timestamp), 0.0), upper_bound)


def _extract_frame(
    video_path: Path,
    output_path: Path,
    timestamp: float,
    ffmpeg_binary: str = "ffmpeg",
    output_width: int | None = None,
    timeout_seconds: float = 15.0,
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
    ]
    if output_width is not None and output_width > 0:
        command.extend(
            [
                "-vf",
                f"scale={output_width}:{output_width}:force_original_aspect_ratio=decrease",
            ]
        )
    command.append(str(output_path))
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is not installed or not available on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed to extract frame at {timestamp:.3f}s from {video_path}: {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out after {timeout_seconds:.0f} seconds while extracting a frame from {video_path}."
        ) from exc
