from __future__ import annotations

import json
import logging
from concurrent.futures import Executor, Future

from fastapi.testclient import TestClient

from gemmaclip.web.app import create_app
from gemmaclip.web.jobs import JobManager
from gemmaclip.web.observability import configure_event_logging, log_event
from gemmaclip.web.services import WebServices
from gemmaclip.web.storage import RunStorage


class NoopExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        future.set_result(None)
        return future


def _client(tmp_path, *, env=None, app_env=None):
    storage = RunStorage(tmp_path / "runs")
    services = WebServices(storage, env=env or {})
    jobs = JobManager(storage, services, executor=NoopExecutor())
    return TestClient(create_app(storage=storage, services=services, jobs=jobs, env=app_env or {}))


def test_health_is_degraded_without_credentials_and_contains_only_safe_state(tmp_path):
    client = _client(tmp_path)
    payload = client.get("/api/health").json()
    assert payload["status"] == "degraded"
    assert payload["providersConfigured"] is False
    assert payload["storage"] == "available"
    assert payload["jobManager"] == "available"
    assert "runs" not in client.get("/api/health").text
    assert "API_KEY" not in client.get("/api/health").text


def test_health_is_ok_when_core_services_and_a_provider_are_configured(tmp_path, monkeypatch):
    monkeypatch.setattr("gemmaclip.web.app.shutil.which", lambda name: f"/usr/bin/{name}")
    client = _client(tmp_path, env={"GOOGLE_API_KEY": "test-secret"})
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["mediaTools"] == {"ffmpeg": "available", "ffprobe": "available"}
    assert "test-secret" not in json.dumps(payload)


def test_health_reports_unavailable_storage(tmp_path):
    storage = RunStorage(tmp_path / "runs")
    storage.health_check = lambda: False
    services = WebServices(storage, env={})
    jobs = JobManager(storage, services, executor=NoopExecutor())
    client = TestClient(create_app(storage=storage, services=services, jobs=jobs, env={}))
    payload = client.get("/api/health").json()
    assert payload["status"] == "unavailable"
    assert payload["storage"] == "unavailable"


def test_static_frontend_fallback_excludes_api_and_unknown_assets(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text('<html><body><div id="root"></div></body></html>', encoding="utf-8")
    client = _client(tmp_path, app_env={"GEMMACLIP_WEB_STATIC_DIR": str(dist)})
    assert 'id="root"' in client.get("/").text
    assert 'id="root"' in client.get("/lab/example/video").text
    assert client.get("/api/unknown").status_code == 404
    assert client.get("/assets/unknown.js").status_code == 404
    assert "id=\"root\"" not in client.get("/api/docs").text


def test_lifecycle_logs_are_allowlisted_json_and_redact_secrets(caplog):
    secret = "super-secret-provider-key"
    private_url = "https://private.example.invalid/deployment"
    configure_event_logging({"GEMMACLIP_LOG_FORMAT": "json", "FIREWORKS_API_KEY": secret})
    caplog.set_level(logging.INFO, logger="gemmaclip.web.events")
    log_event(
        "stage_failed",
        run_id="run_aaaaaaaaaaaaaaaaaaaa",
        stage="evidence",
        provider="fireworks",
        model=private_url,
        error_category=private_url,
        secrets={"FIREWORKS_API_KEY": secret, "FIREWORKS_BASE_URL": private_url},
        raw_caption="must never be logged",
    )
    message = caplog.records[-1].getMessage()
    payload = json.loads(message)
    assert payload["event"] == "stage_failed"
    assert payload["stage"] == "evidence"
    assert "raw_caption" not in payload
    assert secret not in message
    assert private_url not in message
    assert "configured Gemma model" in message
