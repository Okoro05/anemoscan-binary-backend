import base64
import binascii
import csv
import io
import re
import zipfile

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from PIL import Image
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import inference
from .config import settings
from .db import get_db
from .models import Capture, CaptureImage, utcnow
from .schemas import CaptureCreate, CaptureList, CaptureOut, RenameIn, iso

router = APIRouter(prefix="/api")

DATA_URL_RE = re.compile(r"^data:(image/(?:jpeg|jpg|png|webp));base64,(.+)$", re.DOTALL)
ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}


def to_out(capture: Capture) -> CaptureOut:
    return CaptureOut(
        id=capture.id,
        patient_id=capture.patient_id,
        scan_number=capture.scan_number,
        name=capture.name,
        status=capture.status,
        camera_label=capture.camera_label,
        threshold=capture.threshold,
        brightness=capture.brightness,
        sharpness=capture.sharpness,
        contrast=capture.contrast,
        overall=capture.overall,
        age=capture.age,
        gender=capture.gender,
        image_url=f"/api/captures/{capture.id}/image",
        anemia_detected=capture.anemia_detected,
        anemia_probability=capture.anemia_probability,
        anemia_status=capture.anemia_status,
        analyzed_at=iso(capture.analyzed_at),
        created_at=iso(capture.created_at) or "",
    )


def decode_image(data_url: str) -> tuple[str, bytes]:
    match = DATA_URL_RE.match(data_url)
    if not match:
        raise HTTPException(422, "imageData must be a base64 image data URL (jpeg, png or webp).")
    try:
        data = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, "imageData is not valid base64.") from exc
    if len(data) > settings.max_image_bytes:
        raise HTTPException(413, "The image is too large.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            content_type = ALLOWED_FORMATS.get(image.format or "")
            image.verify()
    except Exception as exc:
        raise HTTPException(422, "The file is not a readable image.") from exc
    if content_type is None:
        raise HTTPException(422, "Only jpeg, png or webp images are accepted.")
    return content_type, data


def get_capture_or_404(db: Session, capture_id: int) -> Capture:
    capture = db.get(Capture, capture_id)
    if capture is None:
        raise HTTPException(404, "Scan not found.")
    return capture


@router.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        database_ok = True
    except Exception:
        database_ok = False
    return {"status": "ok" if database_ok else "degraded", "database": database_ok, **inference.model_info()}


@router.get("/captures", response_model=CaptureList)
def list_captures(limit: int = Query(8, ge=1, le=24), offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    rows = db.scalars(
        select(Capture).order_by(Capture.created_at.desc(), Capture.id.desc()).limit(limit).offset(offset)
    ).all()
    return CaptureList(captures=[to_out(row) for row in rows], has_more=len(rows) == limit)


@router.post("/captures", response_model=CaptureOut, status_code=201)
def create_capture(payload: CaptureCreate, db: Session = Depends(get_db)):
    content_type, data = decode_image(payload.image_data)
    base = (
        db.scalar(select(func.coalesce(func.max(Capture.scan_number), 0)).where(Capture.patient_id == payload.patient_id))
        or 0
    ) + 1

    for attempt in range(6):
        scan_number = base + attempt
        capture = Capture(
            patient_id=payload.patient_id,
            scan_number=scan_number,
            name=f"{payload.patient_id}-scan-{scan_number}",
            status="READY" if payload.status == "READY" else "LOW QUALITY",
            camera_label=payload.camera_label if payload.camera_label in ("FRONT", "BACK", "IMPORT") else "BACK",
            threshold=payload.threshold,
            brightness=payload.metrics.brightness,
            sharpness=payload.metrics.sharpness,
            contrast=payload.metrics.contrast,
            overall=payload.metrics.overall,
            age=payload.age,
            gender=payload.gender,
            image_bytes=len(data),
        )
        capture.image = CaptureImage(content_type=content_type, data=data)
        db.add(capture)
        try:
            db.commit()
            return to_out(capture)
        except IntegrityError:
            db.rollback()

    raise HTTPException(409, "Could not allocate a scan number. Please try again.")


@router.get("/captures/{capture_id}/image")
def get_image(capture_id: int, db: Session = Depends(get_db)):
    capture = get_capture_or_404(db, capture_id)
    if capture.image is None:
        raise HTTPException(404, "Image not found.")
    return Response(content=capture.image.data, media_type=capture.image.content_type, headers={"Cache-Control": "public, max-age=31536000, immutable"})


@router.patch("/captures/{capture_id}", response_model=CaptureOut)
def rename_capture(capture_id: int, payload: RenameIn, db: Session = Depends(get_db)):
    capture = get_capture_or_404(db, capture_id)
    clash = db.scalar(select(Capture.id).where(Capture.name == payload.name, Capture.id != capture_id))
    if clash is not None:
        raise HTTPException(409, "This name is already used by another scan.")
    capture.name = payload.name
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(409, "This name is already used by another scan.") from exc
    return to_out(capture)


@router.delete("/captures/{capture_id}", status_code=204)
def delete_capture(capture_id: int, db: Session = Depends(get_db)):
    capture = get_capture_or_404(db, capture_id)
    db.delete(capture)
    db.commit()
    return Response(status_code=204)


@router.post("/captures/{capture_id}/analyze", response_model=CaptureOut)
def analyze_capture(capture_id: int, db: Session = Depends(get_db)):
    capture = get_capture_or_404(db, capture_id)
    if capture.image is None:
        raise HTTPException(404, "Image not found.")
    try:
        prediction = inference.predict(capture.image.data)
    except inference.InferenceError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc

    capture.anemia_detected = prediction.anemia_detected
    capture.anemia_probability = prediction.anemia_probability
    capture.anemia_status = prediction.status
    capture.analyzed_at = utcnow()
    db.commit()
    return to_out(capture)


@router.get("/export")
def export_dataset(db: Session = Depends(get_db)):
    rows = db.scalars(select(Capture).order_by(Capture.patient_id, Capture.scan_number)).all()
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["patient_id", "image", "gender", "age", "scan_number", "quality_score", "anemia_detected", "anemia_probability", "anemia_status"])

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as bundle:
        for capture in rows:
            if capture.image is None:
                continue
            filename = f"{capture.name}.{EXTENSIONS.get(capture.image.content_type, 'jpg')}"
            bundle.writestr(f"images/{filename}", capture.image.data)
            writer.writerow([
                capture.patient_id,
                filename,
                capture.gender or "",
                "" if capture.age is None else capture.age,
                capture.scan_number,
                round(capture.overall, 1),
                "" if capture.anemia_detected is None else capture.anemia_detected,
                "" if capture.anemia_probability is None else capture.anemia_probability,
                capture.anemia_status or "",
            ])
        bundle.writestr("anemia_scans.csv", csv_buffer.getvalue())

    return Response(content=archive.getvalue(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="anemoscan-dataset.zip"'})
