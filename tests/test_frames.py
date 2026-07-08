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


def test_aks_lite_includes_first_and_last_frames_when_available(tmp_path):
    frames = make_candidate_frames(tmp_path, 24)

    selected = select_aks_lite_frames(frames, max_frames=12)

    assert selected[0].path == frames[0].path
    assert selected[-1].path == frames[-1].path


def test_uniform_strategy_still_works(tmp_path, monkeypatch):
    def fake_extract_frame(video_path, output_path, timestamp, ffmpeg_binary="ffmpeg"):
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
