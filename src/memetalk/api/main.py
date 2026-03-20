from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from memetalk.app.container import AppContainer, build_container
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
        response = active_container.search_service.search(
            query=payload.query,
            top_n=payload.top_n,
            candidate_k=max(payload.candidate_k, payload.top_n),
            mode=payload.mode,
        )
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
