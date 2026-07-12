from concurrent.futures import Executor, Future, ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import wave

from fastapi.testclient import TestClient
import pytest

from gemmaclip.audio import AudioEvidenceCandidate, unavailable_audio
from gemmaclip.captioner import build_bounded_evidence_captions
from gemmaclip.frames import ExtractedFrame
from gemmaclip.gemma_client import load_routed_gemma_config
from gemmaclip.io import Task
from gemmaclip.routed import RouteDecision, generate_routed_captions_from_evidence, generate_routed_evidence
from gemmaclip.video import VideoMetadata
from gemmaclip.web.app import create_app
from gemmaclip.web.jobs import JobManager
from gemmaclip.web.services import PipelineDependencies, WebServices
from gemmaclip.web.services import WebPipelineError
from gemmaclip.web.storage import RunStorage


class ImmediateExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):
        fn(*args, **kwargs)
        future = Future()
        future.set_result(None)
        return future


class BlockingExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        future.set_result(None)
        return future


class FakeGemma:
    def chat_completion_text(self, messages, temperature, **kwargs):
        del temperature, kwargs
        prompt = messages[-1]["content"][-1]["text"]
        if "Return exactly this dynamic JSON object" in prompt:
            return json.dumps({
                style: "A grounded person walks across a bright room while the camera observes the visible movement in six chronological frames throughout."
                for style in ("formal", "sarcastic", "humorous_tech", "humorous_non_tech")
            })
        return json.dumps({
            "scene": "bright room",
            "main_subjects": ["person"],
            "actions": ["walking"],
            "setting": "room",
            "visible_objects": ["chair"],
            "verified_description": "A person walks across a bright room.",
            "audio": {"available": False, "status": "unavailable"},
        })


def _extract(task_id, video_path, metadata, **kwargs):
    del video_path, metadata
    root = Path(kwargs["destination_root"]) / task_id
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for index in range(1, 7):
        path = root / f"frame_{index:03d}.jpg"
        path.write_bytes(b"jpeg")
        result.append(ExtractedFrame(path, float(index), "anchor" if index <= 4 else "dynamic", 0.2))
    return result


def _services(tmp_path, *, executor=None, extract=_extract, audio_prepare=None):
    storage = RunStorage(tmp_path / "runs")
    dependencies = PipelineDependencies(
        probe_fn=lambda path: VideoMetadata(12.0, 24.0, 1280, 720, 288, "h264"),
        audio_probe_fn=lambda path: False,
        frame_extract_fn=extract,
        audio_prepare_fn=audio_prepare or (lambda *args, **kwargs: unavailable_audio(16_000, "no audio stream")),
        client_factory=lambda config: FakeGemma(),
    )
    services = WebServices(storage, env={"GOOGLE_API_KEY": "test-key"}, dependencies=dependencies)
    jobs = JobManager(storage, services, executor=executor or ImmediateExecutor())
    return storage, services, jobs


def _manual_client(tmp_path, *, executor=None, extract=_extract, audio_prepare=None):
    storage, services, jobs = _services(tmp_path, executor=executor, extract=extract, audio_prepare=audio_prepare)
    return TestClient(create_app(storage=storage, services=services, jobs=jobs, env={})), storage


def _create_manual(client):
    uploaded = client.post("/api/runs", files={"video": ("clip.mp4", b"video", "video/mp4")}).json()
    run_id = uploaded["id"]
    metadata = client.post(f"/api/runs/{run_id}/metadata", json={"preset": "balanced"})
    assert metadata.status_code == 200
    return run_id


def test_manual_lab_pipeline_uses_stage_jobs_and_real_snapshots(tmp_path):
    client, _ = _manual_client(tmp_path)
    with client:
        run_id = _create_manual(client)
        initial = client.get(f"/api/runs/{run_id}").json()
        assert initial["mode"] == "manual"
        assert initial["status"] == "pending"
        assert initial["stages"]["frames"] == "waiting"

        assert client.post(f"/api/runs/{run_id}/frames", json={"method": "hybrid", "totalFrames": 6, "anchorCount": 4, "highChangeCount": 2, "minSpacingSec": 1.0, "changeSensitivity": 0.5}).status_code == 202
        assert client.get(f"/api/runs/{run_id}").json()["stages"]["frames"] == "complete"
        assert client.post(f"/api/runs/{run_id}/audio", json={"mode": "automatic", "maxDurationSec": 30, "sampleRateHz": 16000, "minRmsEnergy": 0.01, "strategy": "highest-energy"}).status_code == 202
        assert client.get(f"/api/runs/{run_id}").json()["audio"]["segment"]["status"] == "unavailable"
        assert client.post(f"/api/runs/{run_id}/evidence", json={"route": "auto", "temperature": 0.0, "maxTokens": 2048}).status_code == 202
        evidence = client.get(f"/api/runs/{run_id}").json()
        assert evidence["stages"]["evidence"] == "complete"
        assert evidence["evidence"]["result"]["routeProvider"] == "google"
        assert evidence["evidence"]["result"]["routeModality"] == "visual"
        assert client.post(f"/api/runs/{run_id}/captions", json={"temperature": 0.4, "minWords": 18, "maxWords": 35, "strictGrounding": True, "audioEvidenceMode": "use-if-present", "focusedRepair": True, "styles": ["formal", "sarcastic", "humorous-tech", "humorous-non-tech"]}).status_code == 202
        assert len(client.get(f"/api/runs/{run_id}").json()["captions"]["results"]) == 4
        first = client.post(f"/api/runs/{run_id}/experiments", json={"label": "Experiment A", "captionStyle": "formal"})
        assert first.status_code == 200
        assert len(first.json()["experiments"]) == 1
        client.post(f"/api/runs/{run_id}/captions", json={"temperature": 0.8, "minWords": 18, "maxWords": 35, "strictGrounding": True, "audioEvidenceMode": "use-if-present", "focusedRepair": True, "styles": ["formal", "sarcastic", "humorous-tech", "humorous-non-tech"]})
        second = client.post(f"/api/runs/{run_id}/experiments", json={"label": "Experiment B", "captionStyle": "formal"})
        assert len(second.json()["experiments"]) == 2
        left, right = [item["id"] for item in second.json()["experiments"]]
        comparison = client.get(f"/api/runs/{run_id}/compare", params={"left": left, "right": right})
        assert comparison.status_code == 200
        assert comparison.json()["differences"]["captionTemperature"] == {"left": 0.4, "right": 0.8}


def test_quick_intermediate_stage_keeps_processing_status(tmp_path):
    storage, services, _ = _services(tmp_path)
    audio_started = threading.Event()
    release_audio = threading.Event()
    original_audio = services.analyze_run_audio

    def paused_audio(run_id, config):
        audio_started.set()
        assert release_audio.wait(5)
        return original_audio(run_id, config)

    services.analyze_run_audio = paused_audio
    jobs = JobManager(storage, services, executor=ThreadPoolExecutor(max_workers=1))
    client = TestClient(create_app(storage=storage, services=services, jobs=jobs, env={}))
    with client:
        run_id = _create_manual(client)
        assert client.post(f"/api/runs/{run_id}/quick-caption").status_code == 202
        assert audio_started.wait(5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            run = client.get(f"/api/runs/{run_id}").json()
            if run["stages"]["frames"] == "complete":
                break
            time.sleep(0.01)
        assert run["stages"]["frames"] == "complete"
        assert run["status"] == "processing"
        release_audio.set()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            run = client.get(f"/api/runs/{run_id}").json()
            if run["status"] == "ready":
                break
            time.sleep(0.02)
        assert run["status"] == "ready"
        assert run["stages"]["captions"] == "complete"
    jobs.close()


def test_quick_uses_one_decreasing_deadline_for_later_remote_attempts(tmp_path, monkeypatch):
    client, _ = _manual_client(tmp_path)
    clock = {"now": 0.0}
    monkeypatch.setattr("gemmaclip.web.services.time.monotonic", lambda: clock["now"])
    with client:
        run_id = _create_manual(client)
        services = client.app.state.services
        original_extract = services.extract_run_frames
        original_audio = services.analyze_run_audio

        def slow_frames(run_id, config):
            result = original_extract(run_id, config)
            clock["now"] += 300.0
            return result

        def slow_audio(run_id, config):
            result = original_audio(run_id, config)
            clock["now"] += 200.0
            return result

        services.extract_run_frames = slow_frames
        services.analyze_run_audio = slow_audio
        provider_calls = []

        class UnexpectedProvider:
            def chat_completion_text(self, *args, **kwargs):
                provider_calls.append((args, kwargs))
                raise AssertionError("the evidence provider should be skipped at 70 seconds remaining")

        services.dependencies.client_factory = lambda config: UnexpectedProvider()
        with pytest.raises(WebPipelineError, match="Evidence generation failed safely"):
            services.run_quick_caption(run_id)
        assert provider_calls == []


def test_frame_selection_is_persisted_and_invalidates_downstream(tmp_path):
    client, _ = _manual_client(tmp_path)
    with client:
        run_id = _create_manual(client)
        client.post(f"/api/runs/{run_id}/frames", json={"method": "uniform", "totalFrames": 6, "anchorCount": 0, "highChangeCount": 0, "minSpacingSec": 1.0, "changeSensitivity": 0.5})
        selected = ["frame_001.jpg", "frame_003.jpg", "frame_004.jpg", "frame_005.jpg", "frame_006.jpg", "frame_002.jpg"]
        response = client.patch(f"/api/runs/{run_id}/frames/selection", json={"includedFrameIds": selected})
        assert response.status_code == 200
        run = client.get(f"/api/runs/{run_id}").json()
        assert {frame["id"] for frame in run["frames"]["frames"] if frame["included"]} == set(selected)
        assert run["stages"]["evidence"] == "invalidated"
        assert client.patch(f"/api/runs/{run_id}/frames/selection", json={"includedFrameIds": ["frame_999.jpg", "frame_001.jpg"]}).status_code == 422


def test_invalid_stage_configuration_is_rejected_before_job_creation(tmp_path):
    client, _ = _manual_client(tmp_path)
    with client:
        run_id = _create_manual(client)
        invalid_frames = {"method": "hybrid", "totalFrames": 4, "anchorCount": 0, "highChangeCount": 0, "minSpacingSec": 1.0, "changeSensitivity": 0.5}
        invalid_caption = {"temperature": 0.4, "minWords": 40, "maxWords": 20, "strictGrounding": True, "audioEvidenceMode": "use-if-present", "focusedRepair": True, "styles": ["formal"]}
        invalid_audio = {"mode": "automatic", "maxDurationSec": 30, "sampleRateHz": 16000, "minRmsEnergy": 0.01, "strategy": "custom-range"}
        assert client.post(f"/api/runs/{run_id}/frames", json=invalid_frames).status_code == 422
        assert client.post(f"/api/runs/{run_id}/captions", json=invalid_caption).status_code == 422
        assert client.post(f"/api/runs/{run_id}/audio", json=invalid_audio).status_code == 422
        run = client.get(f"/api/runs/{run_id}").json()
        assert run["status"] == "pending"
        assert run["stages"]["frames"] == "waiting"


def test_conflicting_stage_job_returns_409_and_blocks_delete(tmp_path):
    client, storage = _manual_client(tmp_path, executor=BlockingExecutor())
    with client:
        run_id = _create_manual(client)
        payload = {"method": "hybrid", "totalFrames": 6, "anchorCount": 4, "highChangeCount": 2, "minSpacingSec": 1.0, "changeSensitivity": 0.5}
        assert client.post(f"/api/runs/{run_id}/frames", json=payload).status_code == 202
        assert client.post(f"/api/runs/{run_id}/frames", json=payload).status_code == 409
        assert client.delete(f"/api/runs/{run_id}").status_code == 409
        assert storage.run_dir(run_id).exists()


def test_synchronous_mutations_are_rejected_while_run_job_is_active(tmp_path):
    client, storage = _manual_client(tmp_path, executor=BlockingExecutor())
    with client:
        run_id = _create_manual(client)
        frame_config = {"method": "hybrid", "totalFrames": 6, "anchorCount": 4, "highChangeCount": 2, "minSpacingSec": 1.0, "changeSensitivity": 0.5}
        assert client.post(f"/api/runs/{run_id}/frames", json=frame_config).status_code == 202
        assert client.post(f"/api/runs/{run_id}/metadata", json={"preset": "balanced"}).status_code == 409

    client, storage = _manual_client(tmp_path / "evidence", executor=BlockingExecutor())
    with client:
        run_id = _create_manual(client)
        services = client.app.state.services
        services.extract_run_frames(run_id, frame_config)
        assert client.post(f"/api/runs/{run_id}/evidence", json={"route": "auto", "temperature": 0.0, "maxTokens": 2048}).status_code == 202
        frame_ids = [f"frame_{index:03d}.jpg" for index in range(1, 7)]
        assert client.patch(f"/api/runs/{run_id}/frames/selection", json={"includedFrameIds": frame_ids}).status_code == 409

    client, storage = _manual_client(tmp_path / "captions", executor=BlockingExecutor())
    with client:
        run_id = _create_manual(client)
        services = client.app.state.services
        services.extract_run_frames(run_id, frame_config)
        storage.update_run(run_id, lambda payload: payload["stages"].update(evidence="complete"))
        assert client.post(f"/api/runs/{run_id}/captions", json={"temperature": 0.4, "minWords": 18, "maxWords": 35, "strictGrounding": True, "audioEvidenceMode": "use-if-present", "focusedRepair": True, "styles": ["formal"]}).status_code == 202
        assert client.post(f"/api/runs/{run_id}/experiments", json={"label": "blocked", "captionStyle": "formal"}).status_code == 409


def test_audio_candidate_is_cleaned_after_stage_completion(tmp_path):
    def prepare(video_path, destination, *, settings):
        path = Path(destination) / "audio_selected.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(16000); handle.writeframes(b"\x00\x00" * 1600)
        return AudioEvidenceCandidate(path, True, True, False, 1.0, 0.1, 16000, 0.05, "highest-energy window selected")

    client, storage = _manual_client(tmp_path, audio_prepare=prepare)
    with client:
        run_id = _create_manual(client)
        client.post(f"/api/runs/{run_id}/audio", json={"mode": "automatic", "maxDurationSec": 30, "sampleRateHz": 16000, "minRmsEnergy": 0.01, "strategy": "highest-energy"})
        assert not list(storage.run_dir(run_id).rglob("audio_selected.wav"))
        assert client.get(f"/api/runs/{run_id}").json()["audio"]["segment"]["energyCandidateFound"] is True


def test_failed_stage_is_sanitized_and_active_registry_is_cleared(tmp_path):
    def failing_extract(*args, **kwargs):
        raise RuntimeError("private provider deployment URL")

    client, storage = _manual_client(tmp_path, extract=failing_extract)
    jobs = client.app.state.jobs
    with client:
        run_id = _create_manual(client)
        assert client.post(f"/api/runs/{run_id}/frames", json={"method": "hybrid", "totalFrames": 6, "anchorCount": 4, "highChangeCount": 2, "minSpacingSec": 1.0, "changeSensitivity": 0.5}).status_code == 202
        failed = client.get(f"/api/runs/{run_id}").json()
        assert failed["status"] == "error"
        assert failed["stages"]["frames"] == "error"
        assert "private provider" not in (failed["error"] or "")
        assert not jobs.is_active(run_id)
        assert storage.run_dir(run_id).exists()


def test_evidence_max_tokens_and_caption_policy_reach_provider_attempts(tmp_path):
    frames = []
    for index in range(6):
        path = tmp_path / f"frame_{index}.jpg"
        path.write_bytes(b"jpeg")
        frames.append(ExtractedFrame(path, float(index), "uniform", 0.1))
    config = load_routed_gemma_config({"GOOGLE_API_KEY": "test-key"})
    seen_max_tokens = []
    prompts = []

    class RecordingGemma:
        def chat_completion_text(self, messages, temperature, **kwargs):
            del temperature
            if kwargs.get("max_tokens") is not None:
                seen_max_tokens.append(kwargs["max_tokens"])
                return json.dumps({"scene": "a room", "main_subjects": ["person"], "actions": ["walking"]})
            prompts.append(messages[-1]["content"][-1]["text"])
            return json.dumps({"formal": "A person walks through a bright room while six chronological frames show the visible movement clearly."})

    evidence, _ = generate_routed_evidence(
        "interactive",
        frames,
        unavailable_audio(16_000, "no audio"),
        RouteDecision("visual", False, "test"),
        config=config,
        client_factory=lambda _: RecordingGemma(),
        remaining_time_fn=lambda: 500.0,
        max_tokens=512,
    )
    captions = generate_routed_captions_from_evidence(
        Task("interactive", "video", ("formal",)),
        frames,
        {**evidence, "audio": {"status": "unavailable", "allowed_caption_facts": []}},
        config=config,
        client_factory=lambda _: RecordingGemma(),
        remaining_time_fn=lambda: 500.0,
        min_words=8,
        max_words=16,
        audio_evidence_mode="ignore",
        focused_repair=False,
    )
    assert captions["formal"]
    assert seen_max_tokens == [512]
    assert "8-16 word caption" in prompts[0]
    assert "Ignore all audio evidence" in prompts[0]


def test_deterministic_caption_fallback_respects_word_bounds():
    captions = build_bounded_evidence_captions(
        ("formal", "sarcastic"),
        {"main_subjects": ["a person"], "actions": ["walking"], "setting": "a room"},
        min_words=8,
        max_words=10,
    )
    assert all(8 <= len(caption.split()) <= 10 for caption in captions.values())
    assert all(caption.endswith(".") and "frame" not in caption.lower() for caption in captions.values())
    long_caption = build_bounded_evidence_captions(
        ("formal",),
        {"main_subjects": ["a person"], "actions": ["walking"], "setting": "a room"},
        min_words=40,
        max_words=40,
    )["formal"]
    assert len(long_caption.split()) == 40
    assert long_caption.endswith(".") and "frame" not in long_caption.lower()


def test_routed_caption_fallback_respects_selected_word_bounds(tmp_path):
    frames = []
    for index in range(6):
        path = tmp_path / f"frame_{index}.jpg"
        path.write_bytes(b"jpeg")
        frames.append(ExtractedFrame(path, float(index), "uniform", 0.1))

    class FailingGemma:
        def chat_completion_text(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")

    captions = generate_routed_captions_from_evidence(
        Task("fallback", "video", ("formal",)),
        frames,
        {"scene": "a room", "main_subjects": ["person"], "actions": ["walking"], "audio": {"status": "unavailable", "allowed_caption_facts": []}},
        config=load_routed_gemma_config({"GOOGLE_API_KEY": "test-key"}),
        client_factory=lambda _: FailingGemma(),
        remaining_time_fn=lambda: 500.0,
        min_words=8,
        max_words=10,
    )
    assert 8 <= len(captions["formal"].split()) <= 10
