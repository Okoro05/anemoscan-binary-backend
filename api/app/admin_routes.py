import csv
import io
import re
import zipfile

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as PdfImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.utils import ImageReader
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .admin_auth import CSRF_COOKIE, SESSION_COOKIE, clear_admin_session, create_admin_session, require_admin_session, require_admin_write, validate_credentials
from .admin_schemas import AdminLogin
from .db import get_db
from .models import AdminSession, Capture
from .routes import get_capture_or_404, to_out

router = APIRouter(prefix="/api/admin")
EXTENSIONS = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return cleaned or "anemoscan-report"


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


@router.get("/captures/{capture_id}/report")
def admin_report(capture_id: int, _: AdminSession = Depends(require_admin_session), db: Session = Depends(get_db)):
    capture = get_capture_or_404(db, capture_id)
    if capture.image is None:
        raise HTTPException(404, "Image not found.")

    archive = io.BytesIO()
    document = SimpleDocTemplate(
        archive,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"AnemoScan report - {capture.name}",
        author="AnemoScan",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AnemoTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#00a99d"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "AnemoSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#4b5563"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "AnemoHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "AnemoBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    story = [
        Paragraph("AnemoScan", title_style),
        Paragraph("AI-powered anemia screening report", subtitle_style),
    ]

    try:
        image_reader = ImageReader(io.BytesIO(capture.image.data))
        width, height = image_reader.getSize()
        max_width = 165 * mm
        max_height = 100 * mm
        scale = min(max_width / width, max_height / height, 1)
        image = PdfImage(image_reader, width=width * scale, height=height * scale)
        story.extend([image, Spacer(1, 10)])
    except Exception:
        story.append(Paragraph("Stored scan image could not be embedded in this report.", body_style))

    result_text = "Not analyzed"
    if capture.anemia_detected is True:
        result_text = "Anemia detected"
    elif capture.anemia_detected is False:
        result_text = "No anemia"

    probability = "—" if capture.anemia_probability is None else f"{capture.anemia_probability * 100:.0f}%"
    analyzed = capture.analyzed_at.isoformat() if capture.analyzed_at else "Not analyzed"
    created = capture.created_at.isoformat() if capture.created_at else "—"

    story.append(Paragraph("Screening details", heading_style))
    data = [
        ["Patient ID", capture.patient_id],
        ["Scan", f"{capture.name} (#{capture.scan_number})"],
        ["Age / gender", f"{capture.age if capture.age is not None else '—'} / {capture.gender or '—'}"],
        ["Image quality", f"{capture.overall:.0f}/100"],
        ["AI result", result_text],
        ["AI probability", probability],
        ["AI threshold", "50%"],
        ["Created", created],
        ["Analyzed", analyzed],
    ]
    table = Table(data, colWidths=[42 * mm, 125 * mm], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8fffb")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1f2937")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([table, Spacer(1, 12)])
    story.append(Paragraph(
        "Prototype screening result only. This AI result is not a diagnosis and should be confirmed with an appropriate clinical assessment and hemoglobin/laboratory testing.",
        body_style,
    ))
    document.build(story)
    return Response(
        content=archive.getvalue(),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{_safe_filename(capture.name)}-report.pdf"',
            "Cache-Control": "private, no-store",
        },
    )


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
