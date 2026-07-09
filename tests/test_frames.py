from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from gemmaclip.frames import ExtractedFrame, extract_frames, select_aks_lite_frames
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
    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None):
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
    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None):
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


def test_google_fast_mode_extracts_only_four_frames(tmp_path, monkeypatch):
    extracted_widths: list[int | None] = []

    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None):
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

    assert len(frames) == 4
    assert [frame.path.name for frame in frames] == ["frame_001.jpg", "frame_002.jpg", "frame_003.jpg", "frame_004.jpg"]
    assert extracted_widths == [512, 512, 512, 512]


def test_google_fast_mode_uses_four_timestamp_seeks_based_on_duration(tmp_path, monkeypatch):
    extracted_timestamps: list[float] = []

    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg", output_width=None):
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
    assert extracted_timestamps == [5.0, 35.0, 65.0, 95.0]
