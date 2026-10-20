"""Settings, read from environment variables so the same code runs anywhere."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


class Settings:
    def __init__(self) -> None:
        self.database_url = normalize_database_url(
            os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'anemoscan.db'}")
        )
        self.model_path = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "model" / "model.onnx")))
        origins = os.getenv("CORS_ORIGINS", "*")
        self.cors_origins = [item.strip() for item in origins.split(",") if item.strip()]
        self.max_image_bytes = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
        self.web_dist_dir = os.getenv("WEB_DIST_DIR", "")


settings = Settings()
