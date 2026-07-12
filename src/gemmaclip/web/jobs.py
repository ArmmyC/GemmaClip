from __future__ import annotations

import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Mapping

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

    def start_quick_caption(self, run_id: str) -> dict[str, Any]:
        self.services.ensure_credentials()
        return self._start(run_id, "quick_caption", None, _mark_quick_processing)

    def start_stage(self, run_id: str, stage: str, config: Mapping[str, Any]) -> dict[str, Any]:
        if stage not in {"frames", "audio", "evidence", "captions"}:
            raise ValueError("Unsupported processing stage.")
        return self._start(run_id, stage, dict(config), lambda run: _mark_stage_processing(run, stage))

    def _start(self, run_id: str, job_type: str, config: Mapping[str, Any] | None, marker) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise JobManagerClosed("Processing is temporarily unavailable during shutdown.")
            if run_id in self._active:
                raise JobAlreadyRunning("This run is already processing.")
            self._active.add(run_id)
        try:
            run = self.storage.update_run(run_id, marker)
            self.executor.submit(self._execute, run_id, job_type, config)
        except Exception:
            with self._lock:
                self._active.discard(run_id)
            try:
                self.storage.update_run(run_id, lambda payload: _mark_failed(payload, job_type, "Processing could not start safely. Please retry."))
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

    @contextmanager
    def mutation(self, run_id: str):
        """Serialize synchronous run mutations with asynchronous jobs."""
        with self._lock:
            if self._closed:
                raise JobManagerClosed("Processing is temporarily unavailable during shutdown.")
            if run_id in self._active:
                raise JobAlreadyRunning("This run is currently processing and cannot be changed.")
            self._active.add(run_id)
        try:
            yield
        finally:
            with self._lock:
                self._active.discard(run_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        if self._owns_executor and isinstance(self.executor, ThreadPoolExecutor):
            self.executor.shutdown(wait=False, cancel_futures=True)

    def _execute(self, run_id: str, job_type: str, config: Mapping[str, Any] | None) -> None:
        try:
            if job_type == "quick_caption":
                self.services.run_quick_caption(run_id, progress=lambda stage, message: self._progress(run_id, stage, message))
            elif job_type == "frames":
                self.services.extract_run_frames(run_id, config or {})
            elif job_type == "audio":
                self.services.analyze_run_audio(run_id, config or {})
            elif job_type == "evidence":
                self.services.generate_run_evidence(run_id, config or {})
            elif job_type == "captions":
                self.services.generate_run_captions(run_id, config or {})
        except (WebConfigurationError, WebPipelineError) as exc:
            self.storage.update_run(run_id, lambda run: _mark_failed(run, job_type, str(exc)))
        except Exception:
            self.storage.update_run(run_id, lambda run: _mark_failed(run, job_type, "The processing stage failed safely. Please retry."))
        finally:
            with self._lock:
                self._active.discard(run_id)

    def _progress(self, run_id: str, stage: str, message: str) -> None:
        self.storage.update_run(run_id, lambda run: _mark_progress(run, stage, message))


def _mark_quick_processing(run: dict[str, Any]) -> None:
    run["mode"] = "quick"
    run["status"] = "processing"
    run["activeStage"] = "video"
    run["progressMessage"] = "Inspecting video"
    run["error"] = None
    run["generationOutcome"] = None
    run["degraded"] = False
    run["stageErrors"] = {}


def _mark_stage_processing(run: dict[str, Any], stage: str) -> None:
    run["status"] = "processing"
    run["activeStage"] = stage
    run["progressMessage"] = {
        "frames": "Selecting important moments",
        "audio": "Checking audio",
        "evidence": "Building grounded evidence",
        "captions": "Writing captions",
    }[stage]
    run["error"] = None
    run["stageErrors"].pop(stage, None)
    run["stages"][stage] = "active"


def _mark_progress(run: dict[str, Any], stage: str, message: str) -> None:
    run["status"] = "processing"
    previous = run.get("activeStage")
    if previous in run["stages"] and previous != stage and run["stages"][previous] == "active":
        run["stages"][previous] = "complete"
    if stage in run["stages"]:
        run["stages"][stage] = "active"
    run["activeStage"] = stage
    run["progressMessage"] = message


def _mark_failed(run: dict[str, Any], job_type: str, message: str) -> None:
    stage = "captions" if job_type == "quick_caption" and run.get("activeStage") not in run["stages"] else (run.get("activeStage") if job_type == "quick_caption" else job_type)
    run["status"] = "error"
    if stage in run["stages"]:
        run["stages"][stage] = "error"
    run["stageErrors"][stage] = message
    run["error"] = message
    run["progressMessage"] = "Processing failed"
