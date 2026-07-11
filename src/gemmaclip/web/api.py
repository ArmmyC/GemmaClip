from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile, status
from fastapi.responses import FileResponse

from gemmaclip.web.jobs import JobAlreadyRunning, JobManager
from gemmaclip.web.media import UploadValidationError, media_type_for_path, validate_upload_name
from gemmaclip.web.models import ConfigResponse, HealthResponse, PresetRequest, RunResponse, StatusResponse
from gemmaclip.web.services import DEFAULT_STYLES, WebConfigurationError, WebPipelineError, WebServices
from gemmaclip.web.storage import RunStorage


UPLOAD_CHUNK_SIZE = 1024 * 1024


def create_api_router() -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @router.get("/config", response_model=ConfigResponse)
    def config(request: Request) -> ConfigResponse:
        storage, services, _ = _dependencies(request)
        return ConfigResponse(
            maximum_upload_size=storage.max_upload_bytes,
            supported_video_types=["mp4", "webm", "mov"],
            available_caption_styles=list(DEFAULT_STYLES),
            gemma_credentials_configured=services.credentials_configured(),
            audio_mode_available=True,
        )

    @router.post("/runs", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
    async def create_run(request: Request, video: UploadFile) -> dict:
        storage, _, _ = _dependencies(request)
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
            run = storage.update_run(run_id, lambda payload: _store_upload_size(payload, total))
            return run
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
    def get_run(run_id: str, request: Request) -> dict:
        storage, _, _ = _dependencies(request)
        return storage.public_run(storage.read_run(run_id))

    @router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_run(run_id: str, request: Request) -> Response:
        storage, _, _ = _dependencies(request)
        storage.delete_run(run_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @router.post("/runs/{run_id}/metadata", response_model=RunResponse)
    def metadata(run_id: str, payload: PresetRequest, request: Request) -> dict:
        _, services, _ = _dependencies(request)
        services.apply_preset(run_id, payload.preset)
        return services.probe_run_video(run_id)

    @router.post("/runs/{run_id}/quick-caption", response_model=RunResponse, status_code=status.HTTP_202_ACCEPTED)
    def quick_caption(run_id: str, request: Request) -> dict:
        _, _, jobs = _dependencies(request)
        try:
            return jobs.start_quick_caption(run_id)
        except WebConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except JobAlreadyRunning as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.get("/runs/{run_id}/status", response_model=StatusResponse)
    def run_status(run_id: str, request: Request) -> dict:
        storage, _, _ = _dependencies(request)
        run = storage.public_run(storage.read_run(run_id))
        return {
            "id": run["id"],
            "status": run["status"],
            "activeStage": run.get("activeStage"),
            "progressMessage": run.get("progressMessage"),
            "stages": run["stages"],
            "error": run.get("error"),
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


def _dependencies(request: Request) -> tuple[RunStorage, WebServices, JobManager]:
    return request.app.state.storage, request.app.state.services, request.app.state.jobs


def _store_upload_size(run: dict, total: int) -> None:
    run["video"]["sizeBytes"] = total
