from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from gemmaclip.frames import (
    ExtractedFrame,
    compute_frame_change_score,
    extract_frames,
    resolve_fireworks_frame_mode,
    select_aks_lite_frames,
    select_fireworks_hybrid_timestamps,
    select_fireworks_uniform_frame_selection,
)
from gemmaclip.video import VideoMetadata


def make_candidate_frames(tmp_path: Path, count: int) -> list[ExtractedFrame]:
    frames: list[ExtractedFrame] = []
    for index in range(count):
        image_path = tmp_path / f"candidate_{index + 1:03d}.jpg"
        image = Image.new("RGB", (160, 90), color=(index * 7 % 255, index * 13 % 255, index * 17 % 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((10 + index % 20, 10, 70, 70), outline="white", width=3)
        draw.line((0, index % 90, 159, 89 - (index % 90)), fill="yellow", width=2)
        image.save(image_path, format="JPEG", quality=90)
        frames.append(ExtractedFrame(path=image_path, timestamp_seconds=float(index)))
    return frames


def test_aks_lite_never_selects_more_than_twelve_frames(tmp_path):
    frames = make_candidate_frames(tmp_path, 24)

    selected = select_aks_lite_frames(frames, max_frames=12)

    assert len(selected) <= 12


def test_aks_lite_selected_frames_are_chronological(tmp_path):
    frames = make_candidate_frames(tmp_path, 24)

    selected = select_aks_lite_frames(frames, max_frames=12)

    assert [frame.timestamp_seconds for frame in selected] == sorted(frame.timestamp_seconds for frame in selected)


def test_aks_lite_returns_empty_when_max_frames_is_zero(tmp_path):
    frames = make_candidate_frames(tmp_path, 24)

    selected = select_aks_lite_frames(frames, max_frames=0)

    assert selected == []


def test_aks_lite_includes_first_and_last_frames_when_available(tmp_path):
    frames = make_candidate_frames(tmp_path, 24)

    selected = select_aks_lite_frames(frames, max_frames=12)

    assert selected[0].path == frames[0].path
    assert selected[-1].path == frames[-1].path


def test_aks_lite_covers_early_middle_and_late_portions(tmp_path):
    frames = make_candidate_frames(tmp_path, 36)

    selected = select_aks_lite_frames(frames, max_frames=12)
    indices = [int(frame.timestamp_seconds) for frame in selected]

    assert any(index <= 5 for index in indices)
    assert any(14 <= index <= 21 for index in indices)
    assert any(index >= 30 for index in indices)


def test_aks_lite_gaps_are_not_excessively_large_for_even_candidates(tmp_path):
    frames = make_candidate_frames(tmp_path, 36)

    selected = select_aks_lite_frames(frames, max_frames=12)
    indices = [int(frame.timestamp_seconds) for frame in selected]
    gaps = [current - previous for previous, current in zip(indices, indices[1:])]

    assert gaps
    assert max(gaps) <= 5


def test_aks_lite_first_and_last_bins_are_represented(tmp_path):
    frames = make_candidate_frames(tmp_path, 36)

    selected = select_aks_lite_frames(frames, max_frames=12)
    indices = [int(frame.timestamp_seconds) for frame in selected]

    assert any(0 <= index <= 2 for index in indices)
    assert any(33 <= index <= 35 for index in indices)


def test_uniform_strategy_still_works(tmp_path, monkeypatch):
    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None, **kwargs):
        image = Image.new("RGB", (80, 45), color="blue")
        image.save(output_path, format="JPEG", quality=85)

    monkeypatch.setattr("gemmaclip.frames._extract_frame", fake_extract_frame)

    frames = extract_frames(
        "clip-1",
        tmp_path / "video.mp4",
        VideoMetadata(duration_seconds=20.0, fps=24.0, width=1920, height=1080, frame_count=480),
        strategy="uniform",
        destination_root=tmp_path / "frames",
    )

    assert len(frames) == 12
    assert [frame.timestamp_seconds for frame in frames] == sorted(frame.timestamp_seconds for frame in frames)
    assert all(frame.path.exists() for frame in frames)


def test_aks_lite_extract_frames_integration(tmp_path, monkeypatch):
    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None, **kwargs):
        red = int((timestamp * 37) % 255)
        green = int((timestamp * 61) % 255)
        blue = int((timestamp * 89) % 255)
        image = Image.new("RGB", (160, 90), color=(red, green, blue))
        draw = ImageDraw.Draw(image)
        offset = int((timestamp * 10) % 40)
        draw.rectangle((10 + offset, 12, 70 + offset, 74), outline="white", width=3)
        draw.line((0, offset, 159, 89 - offset), fill="yellow", width=2)
        image.save(output_path, format="JPEG", quality=90)

    monkeypatch.setattr("gemmaclip.frames._extract_frame", fake_extract_frame)

    destination_root = tmp_path / "frames"
    frames = extract_frames(
        "clip-1",
        tmp_path / "video.mp4",
        VideoMetadata(duration_seconds=75.0, fps=24.0, width=1920, height=1080, frame_count=1800),
        strategy="aks-lite",
        destination_root=destination_root,
    )

    task_dir = destination_root / "clip-1"
    assert len(frames) <= 12
    assert [frame.timestamp_seconds for frame in frames] == sorted(frame.timestamp_seconds for frame in frames)
    assert all(frame.path.exists() for frame in frames)
    assert [frame.path.name for frame in frames] == [f"frame_{index:03d}.jpg" for index in range(1, len(frames) + 1)]
    assert not (task_dir / "_candidates").exists()


def test_google_v7_fast_mode_extracts_only_six_frames(tmp_path, monkeypatch):
    extracted_widths: list[int | None] = []

    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None, **kwargs):
        extracted_widths.append(output_width)
        image = Image.new("RGB", (160, 90), color="green")
        image.save(output_path, format="JPEG", quality=85)

    monkeypatch.setattr("gemmaclip.frames._extract_frame", fake_extract_frame)

    frames = extract_frames(
        "clip-1",
        tmp_path / "video.mp4",
        VideoMetadata(duration_seconds=100.0, fps=24.0, width=1920, height=1080, frame_count=2400),
        strategy="aks-lite",
        destination_root=tmp_path / "frames",
        google_fast=True,
    )

    assert len(frames) == 6
    assert [frame.path.name for frame in frames] == [f"frame_{index:03d}.jpg" for index in range(1, 7)]
    assert extracted_widths == [512, 512, 512, 512, 512, 512]


def test_google_v7_fast_mode_uses_six_timestamp_seeks_based_on_duration(tmp_path, monkeypatch):
    extracted_timestamps: list[float] = []

    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None, **kwargs):
        extracted_timestamps.append(timestamp)
        image = Image.new("RGB", (160, 90), color="purple")
        image.save(output_path, format="JPEG", quality=85)

    monkeypatch.setattr("gemmaclip.frames._extract_frame", fake_extract_frame)

    frames = extract_frames(
        "clip-1",
        tmp_path / "video.mp4",
        VideoMetadata(duration_seconds=100.0, fps=24.0, width=1920, height=1080, frame_count=2400),
        strategy="aks-lite",
        destination_root=tmp_path / "frames",
        google_fast=True,
    )

    assert [frame.timestamp_seconds for frame in frames] == extracted_timestamps
    assert extracted_timestamps == [5.0, 20.0, 35.0, 55.0, 75.0, 95.0]


def test_fireworks_uniform_mode_uses_exactly_six_separate_frames_with_requested_timestamps(tmp_path, monkeypatch):
    extracted_timestamps: list[float] = []
    extracted_widths: list[int | None] = []

    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None, **kwargs):
        extracted_timestamps.append(timestamp)
        extracted_widths.append(output_width)
        Image.new("RGB", (512, 288), color="orange").save(output_path, format="JPEG", quality=95)

    monkeypatch.setattr("gemmaclip.frames._extract_frame", fake_extract_frame)
    frames = extract_frames(
        "clip-1",
        tmp_path / "video.mp4",
        VideoMetadata(duration_seconds=100.0, fps=24.0, width=1920, height=1080, frame_count=2400),
        destination_root=tmp_path / "frames",
        fireworks_judge=True,
        env={"GEMMACLIP_FIREWORKS_FRAME_MODE": "uniform"},
    )

    assert len(frames) == 6
    assert extracted_timestamps == [5.0, 23.0, 41.0, 59.0, 77.0, 95.0]
    assert extracted_widths == [512, 512, 512, 512, 512, 512]



def test_fireworks_hybrid_includes_four_anchors_and_two_highest_valid_candidates():
    selection = select_fireworks_hybrid_timestamps(
        100.0,
        [10.0, 20.0, 50.0, 80.0],
        [5.0, 30.0, 20.0, 25.0],
    )

    assert [anchor.timestamp_seconds for anchor in selection.anchors] == [5.0, 35.0, 65.0, 95.0]
    assert [dynamic.timestamp_seconds for dynamic in selection.dynamic] == [20.0, 80.0]
    assert len(selection.final_timestamps) == 6
    assert list(selection.final_timestamps) == sorted(selection.final_timestamps)


def test_fireworks_hybrid_skips_candidates_too_close_to_anchors():
    selection = select_fireworks_hybrid_timestamps(
        100.0,
        [35.5, 20.0, 80.0],
        [100.0, 50.0, 40.0],
    )

    assert [dynamic.timestamp_seconds for dynamic in selection.dynamic] == [20.0, 80.0]
    assert 35.5 not in selection.final_timestamps


def test_fireworks_hybrid_skips_second_dynamic_too_close_to_first():
    selection = select_fireworks_hybrid_timestamps(
        100.0,
        [20.0, 21.0, 80.0],
        [100.0, 99.0, 20.0],
    )

    assert [dynamic.timestamp_seconds for dynamic in selection.dynamic] == [20.0, 80.0]


def test_fireworks_hybrid_removes_duplicate_timestamps_and_fills_backups():
    selection = select_fireworks_hybrid_timestamps(100.0, [20.0, 20.0], [10.0, 9.0])

    assert len(selection.final_timestamps) == 6
    assert len(set(selection.final_timestamps)) == 6
    assert 20.0 in selection.final_timestamps
    assert 50.0 in selection.final_timestamps or 80.0 in selection.final_timestamps


def test_fireworks_hybrid_backup_timestamps_fill_missing_dynamic_slots():
    selection = select_fireworks_hybrid_timestamps(100.0, [35.1, 65.1], [10.0, 9.0])

    assert len(selection.dynamic) == 2
    assert [dynamic.timestamp_seconds for dynamic in selection.dynamic] == [20.0, 50.0]
    assert len(selection.final_timestamps) == 6


def test_fireworks_hybrid_very_short_video_still_produces_six_when_possible():
    selection = select_fireworks_hybrid_timestamps(5.0, [1.0, 2.5, 4.0], [3.0, 2.0, 1.0])

    assert len(selection.final_timestamps) == 6
    assert list(selection.final_timestamps) == sorted(selection.final_timestamps)


def test_fireworks_hybrid_invalid_duration_uses_safe_uniform_fallback():
    selection = select_fireworks_hybrid_timestamps(0.0, [1.0], [1.0])

    assert selection.uniform_fallback_used is True
    assert selection.final_timestamps == (0.05, 0.23, 0.41, 0.59, 0.77, 0.95)


def test_fireworks_hybrid_equal_change_scores_are_deterministic():
    selection = select_fireworks_hybrid_timestamps(100.0, [80.0, 20.0, 50.0], [10.0, 10.0, 10.0])

    assert [dynamic.timestamp_seconds for dynamic in selection.dynamic] == [20.0, 50.0]


def test_fireworks_scan_scoring_detects_large_visual_change():
    black = Image.new("RGB", (96, 54), color="black")
    white = Image.new("RGB", (96, 54), color="white")

    assert compute_frame_change_score(black, white) > 200.0


def test_fireworks_scan_scoring_identical_images_are_low_change():
    image = Image.new("RGB", (96, 54), color="gray")

    assert compute_frame_change_score(image, image) == 0.0


def test_fireworks_uniform_selection_reproduces_exact_ratios():
    selection = select_fireworks_uniform_frame_selection(100.0)

    assert selection.final_timestamps == (5.0, 23.0, 41.0, 59.0, 77.0, 95.0)


def test_fireworks_invalid_environment_mode_falls_back_to_hybrid(caplog):
    caplog.set_level("WARNING")

    assert resolve_fireworks_frame_mode({"GEMMACLIP_FIREWORKS_FRAME_MODE": "bad"}) == "hybrid"
    assert "Invalid GEMMACLIP_FIREWORKS_FRAME_MODE" in caplog.text


def test_fireworks_candidate_extraction_failure_falls_back_to_uniform(tmp_path, monkeypatch):
    extracted_timestamps: list[float] = []

    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None, **kwargs):
        if output_path.parent.name == "_fireworks_scan":
            raise RuntimeError("scan failed")
        extracted_timestamps.append(timestamp)
        Image.new("RGB", (512, 288), color="orange").save(output_path, format="JPEG", quality=95)

    monkeypatch.setattr("gemmaclip.frames._extract_frame", fake_extract_frame)

    frames = extract_frames(
        "clip-1",
        tmp_path / "video.mp4",
        VideoMetadata(duration_seconds=100.0, fps=24.0, width=1920, height=1080, frame_count=2400),
        destination_root=tmp_path / "frames",
        fireworks_judge=True,
        env={"GEMMACLIP_FIREWORKS_FRAME_MODE": "hybrid"},
    )

    assert len(frames) == 6
    assert extracted_timestamps == [5.0, 23.0, 41.0, 59.0, 77.0, 95.0]


def test_google_frame_selection_ignores_fireworks_frame_mode(tmp_path, monkeypatch):
    extracted_timestamps: list[float] = []

    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None, **kwargs):
        extracted_timestamps.append(timestamp)
        Image.new("RGB", (160, 90), color="green").save(output_path, format="JPEG", quality=85)

    monkeypatch.setattr("gemmaclip.frames._extract_frame", fake_extract_frame)

    extract_frames(
        "clip-1",
        tmp_path / "video.mp4",
        VideoMetadata(duration_seconds=100.0, fps=24.0, width=1920, height=1080, frame_count=2400),
        destination_root=tmp_path / "frames",
        google_fast=True,
        env={"GEMMACLIP_FIREWORKS_FRAME_MODE": "uniform"},
    )

    assert extracted_timestamps == [5.0, 20.0, 35.0, 55.0, 75.0, 95.0]


def test_openrouter_frame_selection_uses_google_fast_path_unchanged(tmp_path, monkeypatch):
    extracted_timestamps: list[float] = []

    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None, **kwargs):
        extracted_timestamps.append(timestamp)
        Image.new("RGB", (160, 90), color="green").save(output_path, format="JPEG", quality=85)

    monkeypatch.setattr("gemmaclip.frames._extract_frame", fake_extract_frame)

    extract_frames(
        "clip-1",
        tmp_path / "video.mp4",
        VideoMetadata(duration_seconds=100.0, fps=24.0, width=1920, height=1080, frame_count=2400),
        destination_root=tmp_path / "frames",
        google_fast=True,
        env={"GEMMACLIP_FIREWORKS_FRAME_MODE": "hybrid"},
    )

    assert extracted_timestamps == [5.0, 20.0, 35.0, 55.0, 75.0, 95.0]


def test_fireworks_hybrid_still_extracts_exactly_six_final_images(tmp_path, monkeypatch):
    final_timestamps: list[float] = []

    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None, **kwargs):
        if output_path.parent.name == "_fireworks_scan":
            color = "black" if len(list(output_path.parent.glob("*.jpg"))) < 8 else "white"
            Image.new("RGB", (96, 54), color=color).save(output_path, format="JPEG", quality=85)
            return
        final_timestamps.append(timestamp)
        Image.new("RGB", (512, 288), color="orange").save(output_path, format="JPEG", quality=95)

    monkeypatch.setattr("gemmaclip.frames._extract_frame", fake_extract_frame)

    frames = extract_frames(
        "clip-1",
        tmp_path / "video.mp4",
        VideoMetadata(duration_seconds=100.0, fps=24.0, width=1920, height=1080, frame_count=2400),
        destination_root=tmp_path / "frames",
        fireworks_judge=True,
        env={"GEMMACLIP_FIREWORKS_FRAME_MODE": "hybrid"},
    )

    assert len(frames) == 6
    assert len(final_timestamps) == 6
    assert final_timestamps == sorted(final_timestamps)
    assert [frame.path.name for frame in frames] == [f"frame_{index:03d}.jpg" for index in range(1, 7)]
