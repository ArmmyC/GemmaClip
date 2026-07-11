from __future__ import annotations

import threading
from concurrent.futures import Executor, ThreadPoolExecutor

from gemmaclip.web.services import WebConfigurationError, WebPipelineError, WebServices
from gemmaclip.web.storage import RunStorage


class JobAlreadyRunning(RuntimeError):
    pass


class JobManagerClosed(RuntimeError):
    pass


class JobManager:
    def __init__(self, storage: RunStorage, services: WebServices, *, executor: Executor | None = None) -> None:
        self.storage = storage
        self.services = services
        self.executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemmaclip-web")
        self._owns_executor = executor is None
        self._active: set[str] = set()
        self._closed = False
        self._lock = threading.Lock()

    def start_quick_caption(self, run_id: str) -> dict:
        with self._lock:
            if self._closed:
                raise JobManagerClosed("Caption processing is temporarily unavailable during shutdown.")
            self.services.ensure_credentials()
            if run_id in self._active:
                raise JobAlreadyRunning("This run is already processing.")
            self._active.add(run_id)
        try:
            run = self.storage.update_run(run_id, _mark_processing)
            self.executor.submit(self._execute, run_id)
        except Exception:
            with self._lock:
                self._active.discard(run_id)
            try:
                self.storage.update_run(run_id, lambda payload: _mark_failed(payload, "Caption processing could not start safely. Please retry."))
            except Exception:
                pass
            raise
        return run

    def is_active(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._active

    def active_run_ids(self) -> set[str]:
        with self._lock:
            return set(self._active)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        if self._owns_executor and isinstance(self.executor, ThreadPoolExecutor):
            self.executor.shutdown(wait=False, cancel_futures=True)

    def _execute(self, run_id: str) -> None:
        try:
            self.services.run_quick_caption(run_id, progress=lambda stage, message: self._progress(run_id, stage, message))
        except (WebConfigurationError, WebPipelineError) as exc:
            self.storage.update_run(run_id, lambda run: _mark_failed(run, str(exc)))
        except Exception:
            self.storage.update_run(run_id, lambda run: _mark_failed(run, "The Quick Caption job failed safely. Please retry."))
        finally:
            with self._lock:
                self._active.discard(run_id)

    def _progress(self, run_id: str, stage: str, message: str) -> None:
        self.storage.update_run(run_id, lambda run: _mark_progress(run, stage, message))


def _mark_processing(run: dict) -> None:
    run["status"] = "processing"
    run["activeStage"] = "video"
    run["progressMessage"] = "Inspecting video"
    run["error"] = None
    run["generationOutcome"] = None
    run["degraded"] = False


def _mark_progress(run: dict, stage: str, message: str) -> None:
    previous = run.get("activeStage")
    if previous in run["stages"] and previous != stage and run["stages"][previous] == "active":
        run["stages"][previous] = "complete"
    if stage in run["stages"]:
        run["stages"][stage] = "active"
    run["activeStage"] = stage
    run["progressMessage"] = message


def _mark_failed(run: dict, message: str) -> None:
    run["status"] = "error"
    active = run.get("activeStage")
    if active in run["stages"]:
        run["stages"][active] = "error"
    run["error"] = message
    run["progressMessage"] = "Processing failed"
