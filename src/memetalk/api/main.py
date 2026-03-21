from __future__ import annotations

from contextlib import nullcontext

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from memetalk.app.container import AppContainer, build_container
from memetalk.app.query_image import decode_base64_image, temporary_query_image
from memetalk.config import AppSettings
from memetalk.core.models import SearchRequest


def create_app(settings: AppSettings | None = None, container: AppContainer | None = None) -> FastAPI:
    app = FastAPI(title="MemeTalk API", version="0.1.0")
    app.state.settings = settings or AppSettings.from_env()
    app.state.container = container

    def get_container(request: Request) -> AppContainer:
        if request.app.state.container is None:
            request.app.state.container = build_container(request.app.state.settings)
        return request.app.state.container

    @app.get("/api/v1/health")
    def health(request: Request):
        try:
            active_container = get_container(request)
            return {"status": "ok", "providers": active_container.providers.trace()}
        except Exception as exc:
            return JSONResponse(status_code=503, content={"status": "degraded", "error": str(exc)})

    @app.post("/api/v1/search")
    def search(payload: SearchRequest, request: Request):
        active_container = get_container(request)
        try:
            if payload.query_image_base64 is None:
                image_context = nullcontext(None)
            else:
                image_bytes, detected_media_type = decode_base64_image(payload.query_image_base64)
                image_context = temporary_query_image(
                    image_bytes,
                    filename=payload.query_image_filename,
                    media_type=payload.query_image_media_type or detected_media_type,
                )
            with image_context as query_image_path:
                response = active_container.search_service.search(
                    query=payload.query,
                    top_n=payload.top_n,
                    candidate_k=max(payload.candidate_k, payload.top_n),
                    mode=payload.mode,
                    preferred_tone=payload.preferred_tone,
                    query_image_path=query_image_path,
                )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return response.model_dump()

    @app.get("/api/v1/assets/{image_id}")
    def asset(image_id: str, request: Request):
        active_container = get_container(request)
        asset_record = active_container.repository.get_asset_by_id(image_id)
        if not asset_record:
            raise HTTPException(status_code=404, detail="Image not found.")
        return FileResponse(asset_record.file_path)

    return app


app = create_app()
