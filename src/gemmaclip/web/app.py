from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gemmaclip.web.api import create_api_router
from gemmaclip.web.jobs import JobManager
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
    active_storage = storage or RunStorage.from_env(values)
    active_services = services or WebServices(active_storage, env=values)
    active_jobs = jobs or JobManager(active_storage, active_services)

    app = FastAPI(title="GemmaClip Web API", version="0.1.0", docs_url="/api/docs", redoc_url=None)
    app.state.storage = active_storage
    app.state.services = active_services
    app.state.jobs = active_jobs

    origins = [item.strip() for item in values.get("GEMMACLIP_WEB_CORS_ORIGINS", "").split(",") if item.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
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
    return app


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
