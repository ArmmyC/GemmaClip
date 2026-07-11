import json
from pathlib import Path

from gemmaclip.frames import ExtractedFrame
from gemmaclip.video import VideoMetadata
from gemmaclip.web.services import PipelineDependencies, WebServices
from gemmaclip.web.storage import RunStorage


def test_complete_quick_caption_job_persists_real_adapted_run(tmp_path):
    storage = RunStorage(tmp_path / "runs")
    run = storage.create_run("clip.mp4", ".mp4", "video/mp4")
    storage.upload_path(run["id"], ".mp4").write_bytes(b"video")

    def extract(task_id, video_path, metadata, **kwargs):
        root = Path(kwargs["destination_root"]); root.mkdir(parents=True)
        frames = []
        for index in range(6):
            path = root / f"source-{index}.jpg"; path.write_bytes(b"jpeg")
            frames.append(ExtractedFrame(path, float(index), "anchor" if index < 4 else "dynamic", index / 10))
        return frames

    captions = {"formal": "A person walks across a room in a steady indoor scene.", "sarcastic": "A person bravely completes the historic journey across one ordinary room.", "humorous_tech": "A person traverses the room with excellent uptime and zero navigation exceptions.", "humorous_non_tech": "A person crosses the room, giving the furniture front-row seats."}
    def generate(task, frames, **kwargs):
        evidence = {"scene": "indoor room", "main_subjects": ["person"], "actions": ["walking"], "audio": {"status": "unavailable", "visual_consistency": "unknown"}}
        Path(kwargs["debug_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(kwargs["debug_dir"]) / f"{task.task_id}_routed_evidence.json").write_text(json.dumps(evidence))
        kwargs["stage_callback"]("building_evidence"); kwargs["stage_callback"]("writing_captions")
        return captions

    deps = PipelineDependencies(probe_fn=lambda path: VideoMetadata(12.0, 30.0, 1280, 720, 360, "h264"), audio_probe_fn=lambda path: False, frame_extract_fn=extract, caption_generate_fn=generate)
    result = WebServices(storage, env={"GOOGLE_API_KEY": "test-key"}, dependencies=deps).run_quick_caption(run["id"])
    assert result["status"] == "ready"
    assert [item["style"] for item in result["captions"]["results"]] == ["formal", "sarcastic", "humorous-tech", "humorous-non-tech"]
    assert len(result["frames"]["frames"]) == 6
    assert result["video"]["codec"] == "h264"
    assert result["evidence"]["result"]["subjects"] == ["person"]
    assert storage.read_artifact_json(run["id"], "results/captions.json") == captions
