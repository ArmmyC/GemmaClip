import json
from pathlib import Path

import pytest

from gemmaclip.frames import ExtractedFrame
from gemmaclip.routed import EvidenceExecution
from gemmaclip.video import VideoMetadata
from gemmaclip.web.services import PipelineDependencies, WebPipelineError, WebServices
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
        kwargs["outcome_callback"]("model_generated")
        kwargs["evidence_execution_callback"](EvidenceExecution("google", "gemma-4-31b-it", "visual", True, False, True, "Audio fallback used frames only."))
        return captions

    deps = PipelineDependencies(probe_fn=lambda path: VideoMetadata(12.0, 30.0, 1280, 720, 360, "h264"), audio_probe_fn=lambda path: False, frame_extract_fn=extract, caption_generate_fn=generate)
    result = WebServices(storage, env={"GOOGLE_API_KEY": "test-key"}, dependencies=deps).run_quick_caption(run["id"])
    assert result["status"] == "ready"
    assert [item["style"] for item in result["captions"]["results"]] == ["formal", "sarcastic", "humorous-tech", "humorous-non-tech"]
    assert len(result["frames"]["frames"]) == 6
    assert result["video"]["codec"] == "h264"
    assert result["evidence"]["result"]["subjects"] == ["person"]
    assert result["generationOutcome"] == "model_generated"
    assert result["degraded"] is False
    assert result["evidence"]["result"]["selectedRoute"] == "gemma-4-31b"
    assert result["evidence"]["result"]["routeProvider"] == "google"
    assert result["evidence"]["result"]["routeModality"] == "visual"
    assert result["evidence"]["result"]["audioFallbackOccurred"] is True
    assert storage.read_artifact_json(run["id"], "results/captions.json") == captions


@pytest.mark.parametrize("outcome,expected_degraded", [("model_generated", False), ("evidence_fallback", True)])
def test_quick_caption_persists_honest_success_outcome(tmp_path, outcome, expected_degraded):
    storage, run, deps, captions = _quick_fixture(tmp_path, outcome)
    result = WebServices(storage, env={"GOOGLE_API_KEY": "test-key"}, dependencies=deps).run_quick_caption(run["id"])
    assert result["status"] == "ready"
    assert result["generationOutcome"] == outcome
    assert result["degraded"] is expected_degraded
    assert len(result["captions"]["results"]) == 4


def test_deterministic_fallback_is_not_published_as_success(tmp_path):
    storage, run, deps, _ = _quick_fixture(tmp_path, "deterministic_fallback")
    with pytest.raises(WebPipelineError, match="could not produce grounded evidence"):
        WebServices(storage, env={"GOOGLE_API_KEY": "test-key"}, dependencies=deps).run_quick_caption(run["id"])
    assert storage.read_run(run["id"])["generationOutcome"] == "deterministic_fallback"


def test_evidence_fallback_requires_a_nonempty_evidence_artifact(tmp_path):
    storage, run, deps, _ = _quick_fixture(tmp_path, "evidence_fallback")
    original = deps.caption_generate_fn
    def generate_without_evidence(*args, **kwargs):
        captions = original(*args, **kwargs)
        for path in Path(kwargs["debug_dir"]).glob("*_routed_evidence.json"):
            path.unlink()
        return captions
    deps.caption_generate_fn = generate_without_evidence
    with pytest.raises(WebPipelineError, match="could not preserve grounded evidence"):
        WebServices(storage, env={"GOOGLE_API_KEY": "test-key"}, dependencies=deps).run_quick_caption(run["id"])


def test_ready_run_requires_all_four_nonempty_styles(tmp_path):
    storage, run, deps, captions = _quick_fixture(tmp_path, "model_generated")
    del captions["sarcastic"]
    with pytest.raises(WebPipelineError, match="all required caption styles"):
        WebServices(storage, env={"GOOGLE_API_KEY": "test-key"}, dependencies=deps).run_quick_caption(run["id"])


def _quick_fixture(tmp_path, outcome):
    storage = RunStorage(tmp_path / outcome)
    run = storage.create_run("clip.mp4", ".mp4", "video/mp4")
    storage.upload_path(run["id"], ".mp4").write_bytes(b"video")
    captions = {style: f"A sufficiently grounded caption for the requested {style} style with visible activity in the room." for style in ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")}
    def extract(*args, **kwargs):
        root = Path(kwargs["destination_root"]); root.mkdir(parents=True)
        result = []
        for index in range(6):
            path = root / f"f{index}.jpg"; path.write_bytes(b"jpeg")
            result.append(ExtractedFrame(path, float(index), "anchor"))
        return result
    def generate(task, frames, **kwargs):
        debug = Path(kwargs["debug_dir"]); debug.mkdir(parents=True, exist_ok=True)
        if outcome != "deterministic_fallback":
            (debug / f"{task.task_id}_routed_evidence.json").write_text(json.dumps({"scene": "room", "audio": {"available": False, "status": "unavailable"}}))
        kwargs["outcome_callback"](outcome)
        return captions
    deps = PipelineDependencies(probe_fn=lambda path: VideoMetadata(12, 30, 640, 480, 360, "h264"), audio_probe_fn=lambda path: False, frame_extract_fn=extract, caption_generate_fn=generate)
    return storage, run, deps, captions
