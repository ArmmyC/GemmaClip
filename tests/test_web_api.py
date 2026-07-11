from concurrent.futures import Executor, Future

from fastapi.testclient import TestClient

from gemmaclip.web.app import create_app
from gemmaclip.web.jobs import JobManager
from gemmaclip.web.services import WebServices
from gemmaclip.web.storage import RunStorage


class NoopExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):
        future = Future(); future.set_result(None); return future


def client_for(tmp_path, *, max_bytes=64, env=None):
    storage = RunStorage(tmp_path, max_upload_bytes=max_bytes)
    services = WebServices(storage, env=env or {})
    jobs = JobManager(storage, services, executor=NoopExecutor())
    return TestClient(create_app(storage=storage, services=services, jobs=jobs, env={})), storage


def test_upload_config_media_delete_and_secret_filtering(tmp_path):
    client, storage = client_for(tmp_path, env={"FIREWORKS_API_KEY": "super-secret"})
    config = client.get("/api/config")
    assert config.status_code == 200 and config.json()["gemmaCredentialsConfigured"] is True
    assert "super-secret" not in config.text and "FIREWORKS" not in config.text
    response = client.post("/api/runs", files={"video": ("../clip.mp4", b"video", "video/mp4")})
    assert response.status_code == 201
    run_id = response.json()["id"]
    assert client.get(f"/api/runs/{run_id}/media/video").content == b"video"
    assert client.delete(f"/api/runs/{run_id}").status_code == 204
    assert client.get(f"/api/runs/{run_id}").status_code == 404


def test_health_metadata_status_and_frame_endpoints(tmp_path):
    client, storage = client_for(tmp_path)
    assert client.get("/api/health").json() == {"status": "ok"}
    uploaded = client.post("/api/runs", files={"video": ("clip.mp4", b"video", "video/mp4")}).json()
    run_id = uploaded["id"]
    client.app.state.services.probe_run_video = lambda value: storage.public_run(storage.read_run(value))
    metadata = client.post(f"/api/runs/{run_id}/metadata", json={"preset": "fast"})
    assert metadata.status_code == 200 and metadata.json()["preset"] == "fast"
    assert client.get(f"/api/runs/{run_id}/status").json()["status"] == "pending"
    frame = storage.run_dir(run_id) / "frames" / "frame_001.jpg"; frame.write_bytes(b"jpeg")
    assert client.get(f"/api/runs/{run_id}/frames/frame_001.jpg").content == b"jpeg"


def test_upload_rejects_size_extension_and_traversal(tmp_path):
    client, _ = client_for(tmp_path, max_bytes=4)
    assert client.post("/api/runs", files={"video": ("clip.exe", b"x", "application/octet-stream")}).status_code == 415
    assert client.post("/api/runs", files={"video": ("clip.mp4", b"12345", "video/mp4")}).status_code == 413
    assert client.get("/api/runs/..%2Fsecret").status_code in {400, 404}


def test_missing_credentials_is_clear_and_sanitized(tmp_path):
    client, _ = client_for(tmp_path)
    uploaded = client.post("/api/runs", files={"video": ("clip.mp4", b"x", "video/mp4")}).json()
    response = client.post(f"/api/runs/{uploaded['id']}/quick-caption")
    assert response.status_code == 503
    assert response.json() == {"detail": "Gemma credentials are not configured for routed captioning."}


def test_unexpected_api_errors_do_not_expose_internal_details(tmp_path):
    client, storage = client_for(tmp_path)
    client = TestClient(client.app, raise_server_exceptions=False)
    storage.read_run = lambda run_id: (_ for _ in ()).throw(RuntimeError("private-key-value"))
    response = client.get("/api/runs/run_aaaaaaaaaaaaaaaaaaaa")
    assert response.status_code == 500
    assert response.json() == {"detail": "The request failed safely. Please retry."}
    assert "private-key-value" not in response.text
