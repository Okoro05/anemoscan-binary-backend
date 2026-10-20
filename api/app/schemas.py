import re
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

PATIENT_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


def normalize_gender(value: object) -> object:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("m", "male"):
            return "male"
        if text in ("f", "female"):
            return "female"
    raise ValueError("Gender must be male or female.")


class MetricsIn(CamelModel):
    brightness: float = 0
    sharpness: float = 0
    contrast: float = 0
    overall: float = 0


class CaptureCreate(CamelModel):
    image_data: str
    patient_id: str
    age: int | None = Field(default=None, ge=0, le=120)
    gender: Literal["male", "female"] | None = None
    metrics: MetricsIn = MetricsIn()
    threshold: float = 70
    camera_label: str = "BACK"
    status: str = "LOW QUALITY"

    @field_validator("patient_id")
    @classmethod
    def check_patient_id(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if len(cleaned) > 40 or not PATIENT_ID_RE.match(cleaned):
            raise ValueError("Patient ID may only use lowercase letters, numbers and hyphens (e.g. patient-042).")
        return cleaned

    @field_validator("gender", mode="before")
    @classmethod
    def check_gender(cls, value: object) -> object:
        return normalize_gender(value)


class RenameIn(CamelModel):
    name: str

    @field_validator("name")
    @classmethod
    def check_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not NAME_RE.match(cleaned):
            raise ValueError("Name must be lowercase with no spaces (letters, numbers, - _ . only, up to 80 characters).")
        return cleaned


class CaptureOut(CamelModel):
    id: int
    patient_id: str
    scan_number: int
    name: str
    status: str
    camera_label: str
    threshold: float
    brightness: float
    sharpness: float
    contrast: float
    overall: float
    age: int | None
    gender: str | None
    image_url: str
    anemia_detected: bool | None
    anemia_probability: float | None
    anemia_status: str | None
    analyzed_at: str | None
    created_at: str


class CaptureList(CamelModel):
    captures: list[CaptureOut]
    has_more: bool


def iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()
