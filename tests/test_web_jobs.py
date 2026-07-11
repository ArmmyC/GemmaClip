from concurrent.futures import Executor, Future, ThreadPoolExecutor

import pytest

from gemmaclip.web.jobs import JobManager, JobManagerClosed
from gemmaclip.web.services import WebServices
from gemmaclip.web.services import WebPipelineError
from gemmaclip.web.storage import RunStorage


class TrackingExecutor(Executor):
    def __init__(self): self.shutdown_called = False
    def submit(self, fn, /, *args, **kwargs):
        future = Future(); future.set_result(None); return future
    def shutdown(self, wait=True, *, cancel_futures=False): self.shutdown_called = True


class ImmediateExecutor(Executor):
    def submit(self, fn, /, *args, **kwargs):
        fn(*args, **kwargs)
        future = Future(); future.set_result(None); return future


def test_job_manager_tracks_active_runs_and_does_not_close_injected_executor(tmp_path):
    storage = RunStorage(tmp_path)
    run = storage.create_run("clip.mp4", ".mp4", "video/mp4")
    executor = TrackingExecutor()
    manager = JobManager(storage, WebServices(storage, env={"GOOGLE_API_KEY": "configured"}), executor=executor)
    manager.start_quick_caption(run["id"])
    assert manager.is_active(run["id"])
    assert manager.active_run_ids() == {run["id"]}
    manager.close()
    assert executor.shutdown_called is False
    with pytest.raises(JobManagerClosed): manager.start_quick_caption(run["id"])


def test_owned_executor_shutdown_is_nonblocking(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ThreadPoolExecutor, "shutdown", lambda self, wait=True, *, cancel_futures=False: calls.append((wait, cancel_futures)))
    storage = RunStorage(tmp_path)
    manager = JobManager(storage, WebServices(storage, env={}))
    manager.close()
    assert manager._closed is True
    assert calls == [(False, True)]


def test_deterministic_fallback_job_finishes_as_error_with_outcome(tmp_path):
    storage = RunStorage(tmp_path)
    run = storage.create_run("clip.mp4", ".mp4", "video/mp4")
    class Services:
        def ensure_credentials(self): pass
        def run_quick_caption(self, run_id, progress=None):
            storage.update_run(run_id, lambda payload: payload.update(generationOutcome="deterministic_fallback"))
            raise WebPipelineError("Gemma could not produce grounded evidence for this video. Please retry or check the configured deployment.")
    JobManager(storage, Services(), executor=ImmediateExecutor()).start_quick_caption(run["id"])
    failed = storage.read_run(run["id"])
    assert failed["status"] == "error"
    assert failed["generationOutcome"] == "deterministic_fallback"
    assert failed["error"].startswith("Gemma could not produce grounded evidence")
