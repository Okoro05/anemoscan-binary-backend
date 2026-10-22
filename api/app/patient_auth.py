"""Short-lived browser session that grants access to the patient's current scan only."""

import hashlib
import secrets
from datetime import timedelta

from fastapi import Cookie, Depends, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import Capture, PatientSession, utcnow

SESSION_COOKIE = "anemoscan_patient_session"
SESSION_TTL_HOURS = 12


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cookie_secure(request: Request) -> bool:
    return request.url.scheme == "https" or settings.environment == "production"


def create_patient_session(response: Response, request: Request, db: Session, capture: Capture) -> None:
    raw_token = secrets.token_urlsafe(48)
    session = PatientSession(
        capture_id=capture.id,
        token_hash=_hash_token(raw_token),
        expires_at=utcnow() + timedelta(hours=SESSION_TTL_HOURS),
    )
    db.add(session)
    db.commit()

    secure = _cookie_secure(request)
    response.set_cookie(
        SESSION_COOKIE,
        raw_token,
        max_age=SESSION_TTL_HOURS * 3600,
        httponly=True,
        secure=secure,
        samesite="none" if secure else "lax",
        path="/",
    )


def require_patient_capture(
    capture_id: int,
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    db: Session = Depends(get_db),
) -> Capture:
    if not session_cookie:
        raise HTTPException(401, "Patient scan session required.")

    session = db.scalar(
        select(PatientSession).where(
            PatientSession.token_hash == _hash_token(session_cookie),
            PatientSession.capture_id == capture_id,
            PatientSession.expires_at > utcnow(),
        )
    )
    if session is None:
        raise HTTPException(403, "This scan is not available in the current patient session.")

    capture = db.get(Capture, capture_id)
    if capture is None:
        raise HTTPException(404, "Scan not found.")
    return capture
