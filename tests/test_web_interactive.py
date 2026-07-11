from concurrent.futures import Executor, Future
import json
from pathlib import Path
import wave

from fastapi.testclient import TestClient

from gemmaclip.audio import AudioEvidenceCandidate, unavailable_audio
from gemmaclip.frames import ExtractedFrame
from gemmaclip.video import VideoMetadata
from gemmaclip.web.app import create_app
from gemmaclip.web.jobs import JobManager
from gemmaclip.web.services import PipelineDependencies, WebServices
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


def test_frame_selection_is_persisted_and_invalidates_downstream(tmp_path):
    client, _ = _manual_client(tmp_path)
    with client:
        run_id = _create_manual(client)
        client.post(f"/api/runs/{run_id}/frames", json={"method": "uniform", "totalFrames": 6, "anchorCount": 0, "highChangeCount": 0, "minSpacingSec": 1.0, "changeSensitivity": 0.5})
        selected = ["frame_001.jpg", "frame_003.jpg", "frame_004.jpg"]
        response = client.patch(f"/api/runs/{run_id}/frames/selection", json={"includedFrameIds": selected})
        assert response.status_code == 200
        run = client.get(f"/api/runs/{run_id}").json()
        assert [frame["id"] for frame in run["frames"]["frames"] if frame["included"]] == selected
        assert run["stages"]["evidence"] == "invalidated"
        assert client.patch(f"/api/runs/{run_id}/frames/selection", json={"includedFrameIds": ["frame_999.jpg", "frame_001.jpg"]}).status_code == 422


def test_conflicting_stage_job_returns_409_and_blocks_delete(tmp_path):
    client, storage = _manual_client(tmp_path, executor=BlockingExecutor())
    with client:
        run_id = _create_manual(client)
        payload = {"method": "hybrid", "totalFrames": 6, "anchorCount": 4, "highChangeCount": 2, "minSpacingSec": 1.0, "changeSensitivity": 0.5}
        assert client.post(f"/api/runs/{run_id}/frames", json=payload).status_code == 202
        assert client.post(f"/api/runs/{run_id}/frames", json=payload).status_code == 409
        assert client.delete(f"/api/runs/{run_id}").status_code == 409
        assert storage.run_dir(run_id).exists()


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
