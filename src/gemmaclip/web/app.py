from __future__ import annotations

import logging
import os
import shutil
from contextlib import asynccontextmanager
from collections.abc import Mapping
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from gemmaclip.web.api import create_api_router
from gemmaclip.web.jobs import JobManager
from gemmaclip.web.models import HealthResponse, MediaToolsResponse
from gemmaclip.web.observability import configure_event_logging
from gemmaclip.web.services import WebPipelineError, WebServices
from gemmaclip.web.storage import InvalidRunId, RunNotFound, RunStorage, StorageError, UnsafeAsset


LOGGER = logging.getLogger("gemmaclip.web")


def create_app(
    *,
    storage: RunStorage | None = None,
    services: WebServices | None = None,
    jobs: JobManager | None = None,
    env: Mapping[str, str] | None = None,
) -> FastAPI:
    values = dict(env if env is not None else os.environ)
    configure_event_logging(values)
    active_storage = storage or RunStorage.from_env(values)
    active_services = services or WebServices(active_storage, env=values)
    active_jobs = jobs or JobManager(active_storage, active_services)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            active_storage.recover_interrupted_runs()
            active_storage.cleanup_expired_runs(active_jobs.active_run_ids())
        except Exception:
            LOGGER.warning("Web startup maintenance failed safely.")
        try:
            yield
        finally:
            active_jobs.close()

    app = FastAPI(title="GemmaClip Web API", version="0.1.0", docs_url="/api/docs", redoc_url=None, lifespan=lifespan)
    app.state.storage = active_storage
    app.state.services = active_services
    app.state.jobs = active_jobs

    origins = [item.strip() for item in values.get("GEMMACLIP_WEB_CORS_ORIGINS", "").split(",") if item.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Content-Type"],
        )

    @app.exception_handler(InvalidRunId)
    async def invalid_run_handler(request: Request, exc: InvalidRunId) -> JSONResponse:
        del request
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RunNotFound)
    async def missing_run_handler(request: Request, exc: RunNotFound) -> JSONResponse:
        del request
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(UnsafeAsset)
    async def unsafe_asset_handler(request: Request, exc: UnsafeAsset) -> JSONResponse:
        del request
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(WebPipelineError)
    async def pipeline_handler(request: Request, exc: WebPipelineError) -> JSONResponse:
        del request
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(StorageError)
    async def storage_handler(request: Request, exc: StorageError) -> JSONResponse:
        del request, exc
        return JSONResponse(status_code=500, content={"detail": "Run storage failed safely."})

    @app.exception_handler(Exception)
    async def safe_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.error("Web request failed safely: method=%s path=%s exception=%s", request.method, request.url.path, exc.__class__.__name__)
        return JSONResponse(status_code=500, content={"detail": "The request failed safely. Please retry."})

    app.include_router(create_api_router())
    static_root = Path(values["GEMMACLIP_WEB_STATIC_DIR"]).resolve() if values.get("GEMMACLIP_WEB_STATIC_DIR") else None
    _mount_frontend(app, static_root)
    return app


def build_health_response(storage: RunStorage, services: WebServices, jobs: JobManager) -> HealthResponse:
    storage_ok = storage.health_check()
    media = MediaToolsResponse(
        ffmpeg="available" if shutil.which("ffmpeg") else "unavailable",
        ffprobe="available" if shutil.which("ffprobe") else "unavailable",
    )
    job_ok = jobs.health_available()
    if not storage_ok or not job_ok:
        status = "unavailable"
    elif media.ffmpeg != "available" or media.ffprobe != "available" or not services.credentials_configured():
        status = "degraded"
    else:
        status = "ok"
    return HealthResponse(
        status=status,
        storage="available" if storage_ok else "unavailable",
        media_tools=media,
        providers_configured=services.credentials_configured(),
        job_manager="available" if job_ok else "unavailable",
    )


def _mount_frontend(app: FastAPI, static_root: Path | None) -> None:
    if static_root is None or not static_root.is_dir():
        return
    assets = static_root / "assets"
    if assets.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/assets", StaticFiles(directory=assets, check_dir=True), name="frontend-assets")
    index_path = static_root / "index.html"
    if not index_path.is_file():
        return

    @app.get("/{frontend_path:path}", include_in_schema=False)
    async def frontend_fallback(frontend_path: str):
        if frontend_path == "" or frontend_path == "index.html":
            return FileResponse(index_path, media_type="text/html")
        if frontend_path == "api" or frontend_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        if frontend_path == "assets" or frontend_path.startswith("assets/"):
            raise HTTPException(status_code=404, detail="Frontend asset not found.")
        if ".." in Path(frontend_path).parts:
            raise HTTPException(status_code=404, detail="Frontend route not found.")
        return FileResponse(index_path, media_type="text/html")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "gemmaclip.web.app:create_app",
        factory=True,
        host=os.environ.get("GEMMACLIP_WEB_HOST", "127.0.0.1"),
        port=int(os.environ.get("GEMMACLIP_WEB_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
