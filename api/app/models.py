from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Capture(Base):
    """One scan with image-quality data and the final binary anemia result."""

    __tablename__ = "captures"
    __table_args__ = (UniqueConstraint("patient_id", "scan_number", name="uq_patient_scan"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[str] = mapped_column(String(64), index=True)
    scan_number: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    status: Mapped[str] = mapped_column(String(32))
    camera_label: Mapped[str] = mapped_column(String(16), default="BACK")
    threshold: Mapped[float] = mapped_column(Float, default=70.0)
    brightness: Mapped[float] = mapped_column(Float, default=0.0)
    sharpness: Mapped[float] = mapped_column(Float, default=0.0)
    contrast: Mapped[float] = mapped_column(Float, default=0.0)
    overall: Mapped[float] = mapped_column(Float, default=0.0)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(16), nullable=True)
    image_bytes: Mapped[int] = mapped_column(Integer, default=0)
    anemia_detected: Mapped[bool | None] = mapped_column(nullable=True)
    anemia_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    anemia_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    analyzed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    image: Mapped["CaptureImage | None"] = relationship(
        back_populates="capture", uselist=False, cascade="all, delete-orphan"
    )


class CaptureImage(Base):
    __tablename__ = "capture_images"

    capture_id: Mapped[int] = mapped_column(
        ForeignKey("captures.id", ondelete="CASCADE"), primary_key=True
    )
    content_type: Mapped[str] = mapped_column(String(32))
    data: Mapped[bytes] = mapped_column(LargeBinary)

    capture: Mapped[Capture] = relationship(back_populates="image")
