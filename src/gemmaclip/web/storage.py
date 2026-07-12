from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RUN_ID_PATTERN = re.compile(r"run_[A-Za-z0-9_-]{20,80}\Z")
FRAME_ID_PATTERN = re.compile(r"frame_[0-9]{3}\.jpg\Z")
DEFAULT_RUNS_DIR = Path(".gemmaclip/runs")
DEFAULT_MAX_UPLOAD_BYTES = 200 * 1024 * 1024
DEFAULT_RUN_TTL_SECONDS = 86_400
LOGGER = logging.getLogger("gemmaclip.web.maintenance")


class StorageError(RuntimeError):
    pass


class InvalidRunId(StorageError):
    pass


class RunNotFound(StorageError):
    pass


class UnsafeAsset(StorageError):
    pass


class RunStorage:
    def __init__(self, root: str | Path = DEFAULT_RUNS_DIR, *, max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES, run_ttl_seconds: int = DEFAULT_RUN_TTL_SECONDS) -> None:
        self.root = Path(root).resolve()
        self.max_upload_bytes = max_upload_bytes
        self.run_ttl_seconds = max(0, run_ttl_seconds)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> RunStorage:
        root = env.get("GEMMACLIP_WEB_RUNS_DIR") or env.get("GEMMACLIP_WEB_RUN_DIR") or str(DEFAULT_RUNS_DIR)
        return cls(
            root,
            max_upload_bytes=_positive_int(env.get("GEMMACLIP_WEB_MAX_UPLOAD_BYTES"), DEFAULT_MAX_UPLOAD_BYTES),
            run_ttl_seconds=_nonnegative_int(env.get("GEMMACLIP_WEB_RUN_TTL_SECONDS"), DEFAULT_RUN_TTL_SECONDS),
        )

    def create_run(self, original_filename: str, suffix: str, content_type: str) -> dict[str, Any]:
        run_id = "run_" + secrets.token_urlsafe(18).replace("-", "_")
        run_dir = self.run_dir(run_id, require_exists=False)
        run_dir.mkdir(parents=True, exist_ok=False)
        for name in ("input", "frames", "audio", "debug", "results"):
            (run_dir / name).mkdir()
        run = _initial_run(run_id, sanitize_display_filename(original_filename), content_type)
        run["_uploadSuffix"] = suffix
        self.write_run(run_id, run)
        return self.public_run(run)

    def run_dir(self, run_id: str, *, require_exists: bool = True) -> Path:
        validate_run_id(run_id)
        path = (self.root / run_id).resolve()
        if not path.is_relative_to(self.root):
            raise InvalidRunId("Invalid run identifier.")
        if require_exists and not path.is_dir():
            raise RunNotFound("Run not found.")
        return path

    def input_path(self, run_id: str) -> Path:
        run_dir = self.run_dir(run_id)
        matches = [path for path in (run_dir / "input").iterdir() if path.is_file() and not path.name.endswith(".part")]
        if len(matches) != 1:
            raise RunNotFound("Uploaded video not found.")
        return matches[0]

    def upload_path(self, run_id: str, suffix: str) -> Path:
        if suffix not in {".mp4", ".webm", ".mov"}:
            raise UnsafeAsset("Unsupported video extension.")
        return self.run_dir(run_id) / "input" / f"video{suffix}"

    def frame_path(self, run_id: str, frame_id: str) -> Path:
        if not FRAME_ID_PATTERN.fullmatch(frame_id):
            raise UnsafeAsset("Invalid frame identifier.")
        frames_dir = (self.run_dir(run_id) / "frames").resolve()
        path = (frames_dir / frame_id).resolve()
        if not path.is_relative_to(frames_dir) or not path.is_file():
            raise UnsafeAsset("Frame not found.")
        return path

    def read_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            path = self.run_dir(run_id) / "run.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise RunNotFound("Run not found.") from exc
            if not isinstance(payload, dict):
                raise StorageError("Stored run metadata is invalid.")
            _ensure_run_shape(payload)
            return payload

    def public_run(self, run: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in run.items() if not key.startswith("_")}

    def write_run(self, run_id: str, run: Mapping[str, Any]) -> None:
        with self._lock:
            _atomic_write_json(self.run_dir(run_id, require_exists=False) / "run.json", dict(run))

    def update_run(self, run_id: str, updater: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with self._lock:
            run = self.read_run(run_id)
            updater(run)
            self.write_run(run_id, run)
            return self.public_run(run)

    def write_artifact_json(self, run_id: str, relative_path: str, payload: Any) -> Path:
        run_dir = self.run_dir(run_id)
        path = (run_dir / relative_path).resolve()
        if not path.is_relative_to(run_dir):
            raise UnsafeAsset("Unsafe artifact path.")
        _atomic_write_json(path, payload)
        return path

    def read_artifact_json(self, run_id: str, relative_path: str) -> Any:
        run_dir = self.run_dir(run_id)
        path = (run_dir / relative_path).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise UnsafeAsset("Artifact not found.")
        return json.loads(path.read_text(encoding="utf-8"))

    def clear_frames(self, run_id: str) -> Path:
        frames_dir = self.run_dir(run_id) / "frames"
        for path in frames_dir.iterdir():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        return frames_dir

    def delete_run(self, run_id: str) -> None:
        run_dir = self.run_dir(run_id)
        shutil.rmtree(run_dir)

    def recover_interrupted_runs(self) -> list[str]:
        recovered: list[str] = []
        for run_id in self._stored_run_ids():
            try:
                run = self.read_run(run_id)
                if run.get("status") != "processing":
                    continue
                def interrupt(payload: dict[str, Any]) -> None:
                    payload["status"] = "error"
                    payload["progressMessage"] = "Processing was interrupted by a server restart."
                    payload["error"] = "Processing was interrupted. Please start a new run."
                    active = payload.get("activeStage")
                    if active in payload.get("stages", {}):
                        payload["stages"][active] = "error"
                self.update_run(run_id, interrupt)
                recovered.append(run_id)
                LOGGER.info("run_id=%s operation=recovery status=interrupted", run_id)
            except Exception as exc:
                LOGGER.warning("operation=recovery status=skipped error=%s", type(exc).__name__)
        return recovered

    def cleanup_expired_runs(self, active_run_ids: set[str] | None = None, *, now: datetime | None = None) -> list[str]:
        if self.run_ttl_seconds <= 0:
            return []
        active = active_run_ids or set()
        current = now or datetime.now(UTC)
        deleted: list[str] = []
        for run_id in self._stored_run_ids():
            if run_id in active:
                continue
            try:
                run = self.read_run(run_id)
                if run.get("status") not in {"ready", "error", "pending"}:
                    continue
                created = _parse_created_at(run.get("createdAt"))
                age_seconds = (current - created).total_seconds()
                if age_seconds <= self.run_ttl_seconds:
                    continue
                self.delete_run(run_id)
                deleted.append(run_id)
                LOGGER.info("run_id=%s operation=cleanup status=deleted age_seconds=%.0f", run_id, age_seconds)
            except Exception as exc:
                LOGGER.warning("operation=cleanup status=skipped error=%s", type(exc).__name__)
        return deleted

    def _stored_run_ids(self) -> list[str]:
        return sorted(path.name for path in self.root.iterdir() if path.is_dir() and RUN_ID_PATTERN.fullmatch(path.name))


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise InvalidRunId("Invalid run identifier.")
    if "/" in run_id or "\\" in run_id or ".." in run_id or any(ord(char) < 32 for char in run_id):
        raise InvalidRunId("Invalid run identifier.")
    return run_id


def sanitize_display_filename(filename: str) -> str:
    normalized = str(filename or "video").replace("\\", "/").split("/")[-1]
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", normalized).strip(" ._")
    return (cleaned or "video")[:160]


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _initial_run(run_id: str, filename: str, content_type: str) -> dict[str, Any]:
    return {
        "id": run_id,
        "createdAt": datetime.now(UTC).isoformat(),
        "status": "pending",
        "video": {"filename": filename, "durationSec": 0.0, "width": 0, "height": 0, "fps": 0.0, "codec": "unknown", "sizeBytes": 0, "hasAudioStream": False, "contentType": content_type},
        "preset": "balanced",
        "frames": {"config": {"method": "hybrid", "totalFrames": 6, "anchorCount": 4, "highChangeCount": 2, "minSpacingSec": 1.0, "changeSensitivity": 0.5}, "frames": [], "changeSeries": []},
        "audio": {"config": {"mode": "automatic", "maxDurationSec": 30, "sampleRateHz": 16000, "minRmsEnergy": 0.01, "strategy": "highest-energy"}, "segment": {"startSec": 0.0, "endSec": 0.0, "rms": 0.0, "waveform": [], "hasAudioStream": False, "energyCandidateFound": False, "routeExplanation": "Audio has not been inspected yet.", "artifactAvailable": False, "status": "unavailable"}},
        "evidence": {"config": {"route": "auto", "temperature": 0.0, "maxTokens": 2048, "provider": "automatic", "showPromptStructure": False, "showRawJson": True}, "result": {}},
        "captions": {"config": {"model": "gemma-4-31b", "temperature": 0.4, "minWords": 18, "maxWords": 35, "strictGrounding": True, "audioEvidenceMode": "use-if-present", "focusedRepair": True, "styles": ["formal", "sarcastic", "humorous-tech", "humorous-non-tech"]}, "results": []},
        "experiments": [],
        "stages": {"video": "waiting", "frames": "waiting", "audio": "waiting", "evidence": "waiting", "captions": "waiting", "compare": "waiting"},
        "activeStage": "video",
        "progressMessage": "Video uploaded",
        "error": None,
        "generationOutcome": None,
        "degraded": False,
        "mode": "manual",
        "runtimes": {"video": 0.0, "frames": 0.0, "audio": 0.0, "evidence": 0.0, "captions": 0.0, "compare": 0.0},
        "stageErrors": {},
    }


def _ensure_run_shape(run: dict[str, Any]) -> None:
    """Supply safe defaults for runs written before interactive Lab support."""
    run.setdefault("mode", "quick")
    run.setdefault("runtimes", {stage: 0.0 for stage in ("video", "frames", "audio", "evidence", "captions", "compare")})
    run.setdefault("stageErrors", {})
    run.setdefault("activeStage", None)
    run.setdefault("progressMessage", None)
    run.setdefault("error", None)
    run.setdefault("generationOutcome", None)
    run.setdefault("degraded", False)
    stages = run.setdefault("stages", {})
    for stage in ("video", "frames", "audio", "evidence", "captions", "compare"):
        stages.setdefault(stage, "waiting")


def _positive_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: str | None, default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
    except ValueError:
        return default
    return parsed if parsed >= 0 else 0


def _parse_created_at(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Run creation time is invalid.")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
