from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse

from gemmaclip.web.adapters import backend_style_to_frontend
from gemmaclip.web.jobs import JobAlreadyRunning, JobManager, JobManagerClosed
from gemmaclip.web.media import UploadValidationError, media_type_for_path, validate_upload_name
from gemmaclip.web.models import (
    AudioRequest,
    CaptionRequest,
    ConfigResponse,
    EvidenceRequest,
    ExperimentRequest,
    FrameRequest,
    FrameSelectionRequest,
    HealthResponse,
    PresetRequest,
    RunResponse,
    StatusResponse,
)
from gemmaclip.web.services import DEFAULT_STYLES, WebConfigurationError, WebServices
from gemmaclip.web.storage import RunStorage
from gemmaclip.web.observability import log_event


UPLOAD_CHUNK_SIZE = 1024 * 1024


def create_api_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        from gemmaclip.web.app import build_health_response

        storage, services, jobs = _dependencies(request)
        return build_health_response(storage, services, jobs)

    @router.get("/config", response_model=ConfigResponse)
    def config(request: Request) -> ConfigResponse:
        storage, services, _ = _dependencies(request)
        return ConfigResponse(
            maximum_upload_size=storage.max_upload_bytes,
            supported_video_types=["mp4", "webm", "mov"],
            available_caption_styles=[backend_style_to_frontend(style) for style in DEFAULT_STYLES],
            gemma_credentials_configured=services.credentials_configured(),
            audio_mode_available=True,
        )

    @router.post("/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
    async def create_run(request: Request, video: UploadFile) -> dict[str, Any]:
        storage, _, jobs = _dependencies(request)
        storage.cleanup_expired_runs(jobs.active_run_ids())
        try:
            suffix = validate_upload_name(video.filename, video.content_type)
        except UploadValidationError as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        run = storage.create_run(video.filename or "video", suffix, video.content_type or "application/octet-stream")
        run_id = run["id"]
        destination = storage.upload_path(run_id, suffix)
        part_path = destination.with_suffix(destination.suffix + ".part")
        total = 0
        try:
            with part_path.open("wb") as handle:
                while True:
                    chunk = await video.read(UPLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > storage.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="The uploaded video exceeds the configured size limit.")
                    handle.write(chunk)
            if total <= 0:
                raise HTTPException(status_code=400, detail="The uploaded video is empty.")
            part_path.replace(destination)
            result = storage.update_run(run_id, lambda payload: _store_upload_size(payload, total))
            log_event("run_created", run_id=run_id, mode="manual", stage="video", status="pending", artifact_count=1, secrets=storage_env(request))
            return result
        except Exception:
            part_path.unlink(missing_ok=True)
            try:
                storage.delete_run(run_id)
            except Exception:
                pass
            raise
        finally:
            await video.close()

    @router.get("/runs/{run_id}", response_model=RunResponse)
    def get_run(run_id: str, request: Request) -> dict[str, Any]:
        storage, _, _ = _dependencies(request)
        return storage.public_run(storage.read_run(run_id))

    @router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_run(run_id: str, request: Request) -> Response:
        storage, _, jobs = _dependencies(request)
        try:
            with jobs.mutation(run_id):
                storage.delete_run(run_id)
        except JobAlreadyRunning as exc:
            raise HTTPException(status_code=409, detail="This run is currently processing and cannot be deleted.") from exc
        log_event("run_deleted", run_id=run_id, status="deleted", secrets=storage_env(request))
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post("/runs/{run_id}/metadata", response_model=RunResponse)
    def metadata(run_id: str, payload: PresetRequest, request: Request) -> dict[str, Any]:
        _, services, jobs = _dependencies(request)
        try:
            with jobs.mutation(run_id):
                services.apply_preset(run_id, payload.preset)
                return services.probe_run_video(run_id)
        except JobAlreadyRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/quick-caption", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
    def quick_caption(run_id: str, request: Request) -> dict[str, Any]:
        _, _, jobs = _dependencies(request)
        try:
            return jobs.start_quick_caption(run_id)
        except WebConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except JobAlreadyRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except JobManagerClosed as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/runs/{run_id}/frames", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
    def frames(run_id: str, payload: FrameRequest, request: Request) -> dict[str, Any]:
        return _start_stage(request, run_id, "frames", payload.model_dump(by_alias=True))

    @router.patch("/runs/{run_id}/frames/selection", response_model=RunResponse)
    def frame_selection(run_id: str, payload: FrameSelectionRequest, request: Request) -> dict[str, Any]:
        _, services, jobs = _dependencies(request)
        try:
            with jobs.mutation(run_id):
                return services.update_frame_selection(run_id, payload.included_frame_ids)
        except JobAlreadyRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/frames", response_model=RunResponse)
    def get_frames(run_id: str, request: Request) -> dict[str, Any]:
        return _get_public_run(request, run_id)

    @router.post("/runs/{run_id}/audio", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
    def audio(run_id: str, payload: AudioRequest, request: Request) -> dict[str, Any]:
        return _start_stage(request, run_id, "audio", payload.model_dump(by_alias=True))

    @router.get("/runs/{run_id}/audio", response_model=RunResponse)
    def get_audio(run_id: str, request: Request) -> dict[str, Any]:
        return _get_public_run(request, run_id)

    @router.post("/runs/{run_id}/evidence", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
    def evidence(run_id: str, payload: EvidenceRequest, request: Request) -> dict[str, Any]:
        return _start_stage(request, run_id, "evidence", payload.model_dump(by_alias=True))

    @router.get("/runs/{run_id}/evidence", response_model=RunResponse)
    def get_evidence(run_id: str, request: Request) -> dict[str, Any]:
        return _get_public_run(request, run_id)

    @router.post("/runs/{run_id}/captions", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
    def captions(run_id: str, payload: CaptionRequest, request: Request) -> dict[str, Any]:
        return _start_stage(request, run_id, "captions", payload.model_dump(by_alias=True))

    @router.get("/runs/{run_id}/captions", response_model=RunResponse)
    def get_captions(run_id: str, request: Request) -> dict[str, Any]:
        return _get_public_run(request, run_id)

    @router.post("/runs/{run_id}/experiments", response_model=RunResponse)
    def experiment(run_id: str, payload: ExperimentRequest, request: Request) -> dict[str, Any]:
        _, services, jobs = _dependencies(request)
        try:
            with jobs.mutation(run_id):
                return services.create_run_experiment(run_id, payload.label, payload.caption_style)
        except JobAlreadyRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/experiments", response_model=RunResponse)
    def experiments(run_id: str, request: Request) -> dict[str, Any]:
        return _get_public_run(request, run_id)

    @router.get("/runs/{run_id}/compare")
    def compare(run_id: str, left: str, right: str, request: Request) -> dict[str, Any]:
        _, services, _ = _dependencies(request)
        return services.compare_experiments(run_id, left, right)

    @router.get("/runs/{run_id}/status", response_model=StatusResponse)
    def run_status(run_id: str, request: Request) -> dict[str, Any]:
        storage, _, _ = _dependencies(request)
        run = storage.public_run(storage.read_run(run_id))
        return {
            "id": run["id"],
            "status": run["status"],
            "activeStage": run.get("activeStage"),
            "progressMessage": run.get("progressMessage"),
            "stages": run["stages"],
            "runtimes": run.get("runtimes", {}),
            "error": run.get("error"),
            "generationOutcome": run.get("generationOutcome"),
            "degraded": bool(run.get("degraded", False)),
        }

    @router.get("/runs/{run_id}/media/video")
    def video_media(run_id: str, request: Request) -> FileResponse:
        storage, _, _ = _dependencies(request)
        path = storage.input_path(run_id)
        return FileResponse(path, media_type=media_type_for_path(path), filename=None)

    @router.get("/runs/{run_id}/frames/{frame_id}")
    def frame_media(run_id: str, frame_id: str, request: Request) -> FileResponse:
        storage, _, _ = _dependencies(request)
        path = storage.frame_path(run_id, frame_id)
        return FileResponse(path, media_type="image/jpeg", filename=None)

    return router


def _start_stage(request: Request, run_id: str, stage: str, config: dict[str, Any]) -> dict[str, Any]:
    _, _, jobs = _dependencies(request)
    try:
        return jobs.start_stage(run_id, stage, config)
    except JobAlreadyRunning as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except JobManagerClosed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _get_public_run(request: Request, run_id: str) -> dict[str, Any]:
    storage, _, _ = _dependencies(request)
    return storage.public_run(storage.read_run(run_id))


def _dependencies(request: Request) -> tuple[RunStorage, WebServices, JobManager]:
    return request.app.state.storage, request.app.state.services, request.app.state.jobs


def storage_env(request: Request) -> Mapping[str, str]:
    return request.app.state.services.env


def _store_upload_size(run: dict[str, Any], total: int) -> None:
    run["video"]["sizeBytes"] = total
