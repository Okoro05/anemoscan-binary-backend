"""AnemoScan API: scans, patient details, analysis, dataset export."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import eye_state_inference, inference
from .admin_routes import router as admin_router
from .config import settings
from .db import init_db
from .routes import router


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    # Keep the existing anemia model independent from the eye-state model.
    if settings.model_path.exists():
        inference.get_session()
    # Eye-state inference is optional during deployment; its endpoint returns
    # a clear 503 response if the separate artifact has not been deployed yet.
    if settings.eye_state_model_path.exists():
        eye_state_inference.get_session()
    yield


app = FastAPI(title="AnemoScan API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(router)
app.include_router(admin_router)

web_dist = Path(settings.web_dist_dir) if settings.web_dist_dir else None
if web_dist and web_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")
else:

    @app.get("/")
    def root():
        return {"service": "AnemoScan API", "docs": "/docs", "health": "/api/health"}
