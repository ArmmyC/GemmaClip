from __future__ import annotations

import json
import math
import struct
import subprocess
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


DEFAULT_AUDIO_MAX_SECONDS = 30.0
DEFAULT_AUDIO_SAMPLE_RATE = 16_000
DEFAULT_AUDIO_MIN_RMS = 0.01
DEFAULT_AUDIO_MIN_REMAINING_SECONDS = 170.0
DEFAULT_AUDIO_COMMAND_TIMEOUT_SECONDS = 15.0
AUDIO_ENERGY_WINDOW_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class AudioSettings:
    mode: str = "auto"
    max_seconds: float = DEFAULT_AUDIO_MAX_SECONDS
    sample_rate: int = DEFAULT_AUDIO_SAMPLE_RATE
    min_rms: float = DEFAULT_AUDIO_MIN_RMS
    min_remaining_seconds: float = DEFAULT_AUDIO_MIN_REMAINING_SECONDS
    command_timeout_seconds: float = DEFAULT_AUDIO_COMMAND_TIMEOUT_SECONDS
    # ``first-non-silent`` is kept as a backwards-compatible API spelling for
    # the MVP's first-N-seconds selector.  It deliberately does not claim that
    # the window contains speech; RMS is only an energy heuristic.
    strategy: str = "highest-energy"


@dataclass(frozen=True, slots=True)
class AudioEvidenceCandidate:
    path: Path | None
    available: bool
    energy_candidate: bool
    silent: bool
    start_seconds: float
    duration_seconds: float
    sample_rate: int
    rms: float
    reason: str

    @property
    def speech_candidate(self) -> bool:
        """Compatibility alias; non-silent energy does not prove speech."""
        return self.energy_candidate


def load_audio_settings(env: Mapping[str, str]) -> AudioSettings:
    mode = env.get("GEMMACLIP_AUDIO_MODE", "auto").strip().lower()
    if mode not in {"off", "auto", "always"}:
        mode = "auto"
    return AudioSettings(
        mode=mode,
        max_seconds=_positive_float(env.get("GEMMACLIP_AUDIO_MAX_SECONDS"), DEFAULT_AUDIO_MAX_SECONDS),
        sample_rate=_positive_int(env.get("GEMMACLIP_AUDIO_SAMPLE_RATE"), DEFAULT_AUDIO_SAMPLE_RATE),
        min_rms=_nonnegative_float(env.get("GEMMACLIP_AUDIO_MIN_RMS"), DEFAULT_AUDIO_MIN_RMS),
        min_remaining_seconds=_positive_float(
            env.get("GEMMACLIP_AUDIO_MIN_REMAINING_SECONDS"),
            DEFAULT_AUDIO_MIN_REMAINING_SECONDS,
        ),
    )


def has_audio_stream(
    video_path: str | Path,
    *,
    ffprobe_binary: str = "ffprobe",
    timeout_seconds: float = DEFAULT_AUDIO_COMMAND_TIMEOUT_SECONDS,
    runner=subprocess.run,
) -> bool:
    command = [
        ffprobe_binary, "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=index", "-of", "json", str(Path(video_path)),
    ]
    completed = _run(command, timeout_seconds=timeout_seconds, runner=runner, operation="audio probe")
    try:
        payload = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError("ffprobe returned invalid audio stream metadata.") from exc
    return bool(payload.get("streams"))


def prepare_audio_candidate(
    video_path: str | Path,
    destination_dir: str | Path,
    *,
    settings: AudioSettings,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
    runner=subprocess.run,
) -> AudioEvidenceCandidate:
    if settings.mode == "off":
        return unavailable_audio(settings.sample_rate, "audio mode is off")
    try:
        if not has_audio_stream(
            video_path, ffprobe_binary=ffprobe_binary,
            timeout_seconds=settings.command_timeout_seconds, runner=runner,
        ):
            return unavailable_audio(settings.sample_rate, "no audio stream")
    except RuntimeError as exc:
        return unavailable_audio(settings.sample_rate, f"audio probe failed: {exc}")

    output_dir = Path(destination_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_path = output_dir / "audio_full.wav"
    selected_path = output_dir / "audio_selected.wav"
    try:
        _extract_wav(
            video_path, full_path, sample_rate=settings.sample_rate,
            timeout_seconds=settings.command_timeout_seconds,
            ffmpeg_binary=ffmpeg_binary, runner=runner,
        )
        samples, sample_rate = read_pcm16_mono(full_path)
        if settings.strategy == "first-non-silent":
            maximum_samples = max(1, int(settings.max_seconds * sample_rate))
            selected = samples[:maximum_samples]
            start_seconds = 0.0
            duration_seconds = len(selected) / sample_rate
            rms = pcm_rms(selected)
        else:
            start_seconds, duration_seconds, rms = select_highest_energy_window(
                samples, sample_rate, settings.max_seconds,
            )
        silent = rms < settings.min_rms
        if silent:
            return AudioEvidenceCandidate(None, True, False, True, start_seconds, duration_seconds, sample_rate, rms, "selected audio is effectively silent")
        _extract_wav(
            video_path, selected_path, sample_rate=sample_rate,
            timeout_seconds=settings.command_timeout_seconds,
            ffmpeg_binary=ffmpeg_binary, runner=runner,
            start_seconds=start_seconds, duration_seconds=duration_seconds,
        )
        reason = "first window selected" if settings.strategy == "first-non-silent" else "highest-energy window selected"
        return AudioEvidenceCandidate(selected_path, True, True, False, start_seconds, duration_seconds, sample_rate, rms, reason)
    except (RuntimeError, OSError, ValueError, wave.Error) as exc:
        selected_path.unlink(missing_ok=True)
        return unavailable_audio(settings.sample_rate, f"audio extraction failed: {exc}")
    finally:
        try:
            full_path.unlink(missing_ok=True)
        except OSError:
            pass


def read_pcm16_mono(path: str | Path) -> tuple[list[int], int]:
    with wave.open(str(path), "rb") as audio:
        if audio.getnchannels() != 1 or audio.getsampwidth() != 2:
            raise ValueError("Expected mono 16-bit PCM WAV audio.")
        sample_rate = audio.getframerate()
        payload = audio.readframes(audio.getnframes())
    samples = list(struct.unpack(f"<{len(payload) // 2}h", payload)) if payload else []
    return samples, sample_rate


def pcm_rms(samples: Sequence[int]) -> float:
    if not samples:
        return 0.0
    mean_square = sum((sample / 32768.0) ** 2 for sample in samples) / len(samples)
    return math.sqrt(mean_square)


def select_highest_energy_window(
    samples: Sequence[int], sample_rate: int, max_seconds: float,
    *, energy_window_seconds: float = AUDIO_ENERGY_WINDOW_SECONDS,
) -> tuple[float, float, float]:
    if sample_rate <= 0 or max_seconds <= 0:
        raise ValueError("sample_rate and max_seconds must be positive.")
    if not samples:
        return 0.0, 0.0, 0.0
    maximum_samples = max(1, int(max_seconds * sample_rate))
    if len(samples) <= maximum_samples:
        return 0.0, len(samples) / sample_rate, pcm_rms(samples)
    hop = max(1, int(energy_window_seconds * sample_rate))
    best_start = 0
    current_start = 0
    current_energy = sum(sample * sample for sample in samples[:maximum_samples])
    best_energy = current_energy
    last_start = len(samples) - maximum_samples
    for start in range(hop, last_start + 1, hop):
        current_energy -= sum(sample * sample for sample in samples[current_start:start])
        current_energy += sum(sample * sample for sample in samples[current_start + maximum_samples:start + maximum_samples])
        current_start = start
        if current_energy > best_energy:
            best_energy = current_energy
            best_start = start
    last_energy = sum(sample * sample for sample in samples[last_start:])
    if last_start % hop and last_energy > best_energy:
        best_start = last_start
    selected = samples[best_start:best_start + maximum_samples]
    return best_start / sample_rate, len(selected) / sample_rate, pcm_rms(selected)


def cleanup_audio_candidate(candidate: AudioEvidenceCandidate) -> None:
    if candidate.path is not None:
        try:
            candidate.path.unlink(missing_ok=True)
            parent = candidate.path.parent
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass


def unavailable_audio(sample_rate: int, reason: str) -> AudioEvidenceCandidate:
    return AudioEvidenceCandidate(None, False, False, False, 0.0, 0.0, sample_rate, 0.0, reason)


def _extract_wav(video_path, output_path, *, sample_rate, timeout_seconds, ffmpeg_binary, runner, start_seconds=None, duration_seconds=None):
    command = [ffmpeg_binary, "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds is not None:
        command.extend(["-ss", f"{start_seconds:.3f}"])
    command.extend(["-i", str(Path(video_path))])
    if duration_seconds is not None:
        command.extend(["-t", f"{duration_seconds:.3f}"])
    command.extend(["-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(output_path)])
    _run(command, timeout_seconds=timeout_seconds, runner=runner, operation="audio extraction")


def _run(command, *, timeout_seconds, runner, operation):
    try:
        return runner(command, check=True, capture_output=True, text=True, timeout=timeout_seconds)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required media tool is unavailable during {operation}.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{operation} timed out after {timeout_seconds:.0f} seconds.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{operation} failed.") from exc


def _positive_float(value, default):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _nonnegative_float(value, default):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _positive_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
