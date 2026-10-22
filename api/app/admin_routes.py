import csv
import io
import zipfile

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .admin_auth import CSRF_COOKIE, SESSION_COOKIE, clear_admin_session, create_admin_session, require_admin_session, require_admin_write, validate_credentials
from .admin_schemas import AdminLogin
from .db import get_db
from .models import AdminSession, Capture, utcnow
from .routes import EXTENSIONS, get_capture_or_404, to_out

router = APIRouter(prefix="/api/admin")


@router.post("/login")
def admin_login(payload: AdminLogin, request: Request, response: Response, db: Session = Depends(get_db)):
    if not validate_credentials(payload.username, payload.password):
        raise HTTPException(401, "Invalid admin username or password.")
    create_admin_session(response, request, db, payload.username)
    return {"authenticated": True, "username": payload.username}


@router.post("/logout")
def admin_logout(
    request: Request,
    response: Response,
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    db: Session = Depends(get_db),
):
    clear_admin_session(response, request, db, session_cookie)
    return {"authenticated": False}


@router.get("/me")
def admin_me(session: AdminSession = Depends(require_admin_session)):
    return {"authenticated": True, "username": session.username, "expiresAt": session.expires_at.isoformat()}


@router.get("/stats")
def admin_stats(_: AdminSession = Depends(require_admin_session), db: Session = Depends(get_db)):
    total = db.scalar(select(func.count(Capture.id))) or 0
    anemia = db.scalar(select(func.count(Capture.id)).where(Capture.anemia_detected.is_(True))) or 0
    not_anemia = db.scalar(select(func.count(Capture.id)).where(Capture.anemia_detected.is_(False))) or 0
    unanalyzed = db.scalar(select(func.count(Capture.id)).where(Capture.anemia_detected.is_(None))) or 0
    patients = db.scalar(select(func.count(func.distinct(Capture.patient_id)))) or 0
    avg_probability = db.scalar(select(func.avg(Capture.anemia_probability)).where(Capture.anemia_probability.is_not(None)))
    return {
        "totalScans": total,
        "uniquePatients": patients,
        "anemiaDetected": anemia,
        "noAnemia": not_anemia,
        "unanalyzed": unanalyzed,
        "averageAnemiaProbability": round(float(avg_probability), 4) if avg_probability is not None else None,
    }


@router.get("/captures")
def admin_captures(
    q: str = "",
    result: str = "all",
    limit: int = 24,
    offset: int = 0,
    _: AdminSession = Depends(require_admin_session),
    db: Session = Depends(get_db),
):
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    stmt = select(Capture).order_by(Capture.created_at.desc(), Capture.id.desc())
    q = q.strip().lower()
    if q:
        pattern = f"%{q}%"
        stmt = stmt.where((func.lower(Capture.patient_id).like(pattern)) | (func.lower(Capture.name).like(pattern)))
    if result == "anemia":
        stmt = stmt.where(Capture.anemia_detected.is_(True))
    elif result == "no-anemia":
        stmt = stmt.where(Capture.anemia_detected.is_(False))
    elif result == "unanalyzed":
        stmt = stmt.where(Capture.anemia_detected.is_(None))

    rows = db.scalars(stmt.limit(limit).offset(offset)).all()
    total_stmt = select(func.count(Capture.id))
    if q:
        pattern = f"%{q}%"
        total_stmt = total_stmt.where((func.lower(Capture.patient_id).like(pattern)) | (func.lower(Capture.name).like(pattern)))
    if result == "anemia":
        total_stmt = total_stmt.where(Capture.anemia_detected.is_(True))
    elif result == "no-anemia":
        total_stmt = total_stmt.where(Capture.anemia_detected.is_(False))
    elif result == "unanalyzed":
        total_stmt = total_stmt.where(Capture.anemia_detected.is_(None))

    total = db.scalar(total_stmt) or 0
    captures = []
    for row in rows:
        item = to_out(row).model_dump(by_alias=True)
        item["imageUrl"] = f"/api/admin/captures/{row.id}/image"
        captures.append(item)
    return {"captures": captures, "total": total, "hasMore": offset + len(rows) < total}


@router.get("/captures/{capture_id}/image")
def admin_image(capture_id: int, _: AdminSession = Depends(require_admin_session), db: Session = Depends(get_db)):
    capture = get_capture_or_404(db, capture_id)
    if capture.image is None:
        raise HTTPException(404, "Image not found.")
    return Response(content=capture.image.data, media_type=capture.image.content_type, headers={"Cache-Control": "private, no-store"})


@router.delete("/captures/{capture_id}", status_code=204)
def admin_delete_capture(capture_id: int, _: AdminSession = Depends(require_admin_write), db: Session = Depends(get_db)):
    capture = get_capture_or_404(db, capture_id)
    db.delete(capture)
    db.commit()
    return Response(status_code=204)


@router.get("/export")
def admin_export(_: AdminSession = Depends(require_admin_session), db: Session = Depends(get_db)):
    rows = db.scalars(select(Capture).order_by(Capture.patient_id, Capture.scan_number)).all()
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["patient_id", "image", "gender", "age", "scan_number", "quality_score", "anemia_detected", "anemia_probability", "anemia_status", "created_at"])
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as bundle:
        for capture in rows:
            if capture.image is None:
                continue
            filename = f"{capture.name}.{EXTENSIONS.get(capture.image.content_type, 'jpg')}"
            bundle.writestr(f"images/{filename}", capture.image.data)
            writer.writerow([capture.patient_id, filename, capture.gender or "", "" if capture.age is None else capture.age, capture.scan_number, round(capture.overall, 1), capture.anemia_detected, capture.anemia_probability, capture.anemia_status or "", capture.created_at.isoformat()])
        bundle.writestr("anemia_scans.csv", csv_buffer.getvalue())
    return Response(content=archive.getvalue(), media_type="application/zip", headers={"Content-Disposition": 'attachment; filename="anemoscan-admin-export.zip"', "Cache-Control": "private, no-store"})
