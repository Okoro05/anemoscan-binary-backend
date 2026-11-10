"""Settings, read from environment variables so the same code runs anywhere."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model" / "eye_state"


def normalize_database_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


class Settings:
    def __init__(self) -> None:
        self.database_url = normalize_database_url(os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'anemoscan.db'}"))
        # Existing binary anemia model. Do not point this at the eye-state artifact.
        self.model_path = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "model" / "model.onnx")))
        # Independent open-eye / closed-eye model and its training metadata.
        self.eye_state_model_path = Path(
            os.getenv("EYE_STATE_MODEL_PATH", str(MODEL_DIR / "anemoscan_eye_state.onnx"))
        )
        self.eye_state_metadata_path = Path(
            os.getenv("EYE_STATE_METADATA_PATH", str(MODEL_DIR / "model_metadata.json"))
        )
        origins = os.getenv("CORS_ORIGINS", "*")
        self.cors_origins = [item.strip() for item in origins.split(",") if item.strip()]
        self.max_image_bytes = int(os.getenv("MAX_IMAGE_BYTES", str(8 * 1024 * 1024)))
        self.web_dist_dir = os.getenv("WEB_DIST_DIR", "")
        self.environment = os.getenv("APP_ENV", "production").strip().lower()
        self.admin_username = os.getenv("ADMIN_USERNAME", "").strip()
        self.admin_password = os.getenv("ADMIN_PASSWORD", "")


settings = Settings()
