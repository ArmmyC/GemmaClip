from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    duration_seconds: float
    fps: float | None
    width: int | None
    height: int | None
    frame_count: int | None


def probe_video(video_path: str | Path, ffprobe_binary: str = "ffprobe") -> VideoMetadata:
    path = Path(video_path)
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffprobe is not installed or not available on PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {path}: {exc.stderr.strip()}") from exc

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path}.") from exc

    return _parse_ffprobe_payload(payload, path)


def _parse_ffprobe_payload(payload: dict[str, Any], path: Path) -> VideoMetadata:
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError(f"No video stream metadata found for {path}.")

    stream = streams[0]
    duration = _first_float(stream.get("duration"), payload.get("format", {}).get("duration"))
    if duration is None or duration <= 0:
        raise RuntimeError(f"Could not determine a positive video duration for {path}.")

    return VideoMetadata(
        duration_seconds=duration,
        fps=_parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate")),
        width=_parse_int(stream.get("width")),
        height=_parse_int(stream.get("height")),
        frame_count=_parse_int(stream.get("nb_frames")),
    )


def _first_float(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_fps(value: Any) -> float | None:
    if value in (None, "", "0/0"):
        return None
    try:
        return float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError):
        return None
