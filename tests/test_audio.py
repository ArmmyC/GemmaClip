from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

from gemmaclip.audio import (
    AudioEvidenceCandidate,
    cleanup_audio_candidate,
    has_audio_stream,
    load_audio_settings,
    pcm_rms,
    prepare_audio_candidate,
    select_highest_energy_window,
)


def test_audio_settings_load_configured_values_and_invalid_mode_is_safe():
    settings = load_audio_settings({
        "GEMMACLIP_AUDIO_MODE": "invalid",
        "GEMMACLIP_AUDIO_MAX_SECONDS": "12",
        "GEMMACLIP_AUDIO_SAMPLE_RATE": "8000",
        "GEMMACLIP_AUDIO_MIN_RMS": "0.02",
        "GEMMACLIP_AUDIO_MIN_REMAINING_SECONDS": "140",
    })
    assert settings.mode == "auto"
    assert settings.max_seconds == 12
    assert settings.sample_rate == 8000
    assert settings.min_rms == 0.02
    assert settings.min_remaining_seconds == 140


def test_has_audio_stream_handles_present_and_missing_streams(tmp_path):
    responses = [json.dumps({"streams": [{"index": 1}]}), json.dumps({"streams": []})]
    def runner(*args, **kwargs):
        return SimpleNamespace(stdout=responses.pop(0), stderr="")
    assert has_audio_stream(tmp_path / "video.mp4", runner=runner)
    assert not has_audio_stream(tmp_path / "video.mp4", runner=runner)


def test_highest_energy_window_is_deterministic_and_limited():
    rate = 10
    samples = [0] * 300 + [20_000] * 300 + [1_000] * 300
    first = select_highest_energy_window(samples, rate, 30)
    second = select_highest_energy_window(samples, rate, 30)
    assert first == second
    assert first[0] == 30.0
    assert first[1] == 30.0
    assert first[2] > 0.5


def test_short_silent_and_non_silent_rms():
    assert pcm_rms([0] * 100) == 0.0
    start, duration, rms = select_highest_energy_window([16_384] * 80, 10, 30)
    assert (start, duration) == (0.0, 8.0)
    assert 0.49 < rms < 0.51


def test_prepare_audio_candidate_no_stream(tmp_path):
    settings = load_audio_settings({})
    def runner(*args, **kwargs):
        return SimpleNamespace(stdout='{"streams": []}', stderr="")
    candidate = prepare_audio_candidate(tmp_path / "video.mp4", tmp_path / "audio", settings=settings, runner=runner)
    assert not candidate.available
    assert candidate.path is None
    assert candidate.reason == "no audio stream"


def test_prepare_audio_candidate_handles_timeout(tmp_path):
    settings = load_audio_settings({})
    def runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 15)
    candidate = prepare_audio_candidate(tmp_path / "video.mp4", tmp_path / "audio", settings=settings, runner=runner)
    assert not candidate.available
    assert "probe failed" in candidate.reason


def test_prepare_audio_candidate_handles_extraction_failure(tmp_path):
    settings = load_audio_settings({})
    def runner(command, **kwargs):
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout='{"streams": [{"index": 1}]}', stderr="")
        raise subprocess.CalledProcessError(1, command, stderr="private provider detail")
    candidate = prepare_audio_candidate(tmp_path / "video.mp4", tmp_path / "audio", settings=settings, runner=runner)
    assert not candidate.available
    assert candidate.path is None
    assert candidate.reason == "audio extraction failed: audio extraction failed."


def test_cleanup_removes_selected_audio_artifact(tmp_path):
    path = tmp_path / "audio" / "selected.wav"
    path.parent.mkdir()
    path.write_bytes(b"wav")
    candidate = AudioEvidenceCandidate(path, True, True, False, 0, 1, 16000, 0.1, "test")
    cleanup_audio_candidate(candidate)
    assert not path.exists()
    assert not path.parent.exists()


def test_prepare_audio_candidate_extracts_selected_window_and_cleans_full_wav(tmp_path):
    settings = load_audio_settings({"GEMMACLIP_AUDIO_SAMPLE_RATE": "100", "GEMMACLIP_AUDIO_MAX_SECONDS": "2"})
    commands = []
    def runner(command, **kwargs):
        commands.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout='{"streams": [{"index": 1}]}', stderr="")
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        samples = [0] * 200 + [20_000] * 200 if output.name == "audio_full.wav" else [20_000] * 200
        with wave.open(str(output), "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(100)
            wav.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))
        return SimpleNamespace(stdout="", stderr="")
    candidate = prepare_audio_candidate(tmp_path / "video.mp4", tmp_path / "audio", settings=settings, runner=runner)
    assert candidate.available and candidate.speech_candidate and not candidate.silent
    assert candidate.start_seconds == 2.0
    assert candidate.duration_seconds == 2.0
    assert candidate.path and candidate.path.exists()
    assert not (tmp_path / "audio" / "audio_full.wav").exists()
    assert "-ar" in commands[-1] and "100" in commands[-1]
