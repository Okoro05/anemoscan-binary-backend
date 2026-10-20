"""AnemoScan API: scans, patient details, analysis, dataset export."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import inference
from .config import settings
from .db import init_db
from .routes import router


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if settings.model_path.exists():
        inference.get_session()
    yield


app = FastAPI(title="AnemoScan API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

web_dist = Path(settings.web_dist_dir) if settings.web_dist_dir else None
if web_dist and web_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")
else:

    @app.get("/")
    def root():
        return {"service": "AnemoScan API", "docs": "/docs", "health": "/api/health"}
