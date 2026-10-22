"""Cookie-based admin authentication for the AnemoScan administration portal."""

import hashlib
import hmac
import secrets
from datetime import timedelta

from fastapi import Cookie, Depends, Header, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import AdminSession, utcnow

SESSION_COOKIE = "anemoscan_admin_session"
CSRF_COOKIE = "anemoscan_admin_csrf"
SESSION_TTL_HOURS = 12


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cookie_secure(request: Request) -> bool:
    return request.url.scheme == "https" or settings.environment == "production"


def validate_credentials(username: str, password: str) -> bool:
    configured_user = settings.admin_username
    configured_password = settings.admin_password
    if not configured_user or not configured_password:
        return False
    return hmac.compare_digest(username, configured_user) and hmac.compare_digest(password, configured_password)


def create_admin_session(response: Response, request: Request, db: Session, username: str) -> None:
    raw_session = secrets.token_urlsafe(48)
    raw_csrf = secrets.token_urlsafe(32)
    session = AdminSession(
        token_hash=_hash_token(raw_session),
        csrf_token=raw_csrf,
        username=username,
        expires_at=utcnow() + timedelta(hours=SESSION_TTL_HOURS),
    )
    db.add(session)
    db.commit()

    secure = _cookie_secure(request)
    response.set_cookie(SESSION_COOKIE, raw_session, max_age=SESSION_TTL_HOURS * 3600, httponly=True, secure=secure, samesite="none" if secure else "lax", path="/")
    response.set_cookie(CSRF_COOKIE, raw_csrf, max_age=SESSION_TTL_HOURS * 3600, httponly=False, secure=secure, samesite="none" if secure else "lax", path="/")


def require_admin_session(
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    db: Session = Depends(get_db),
) -> AdminSession:
    if not session_cookie:
        raise HTTPException(401, "Admin authentication required.")
    session = db.scalar(select(AdminSession).where(AdminSession.token_hash == _hash_token(session_cookie), AdminSession.expires_at > utcnow()))
    if session is None:
        raise HTTPException(401, "Admin session expired or invalid.")
    return session


def require_admin_write(
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
    csrf_cookie: str | None = Cookie(default=None, alias=CSRF_COOKIE),
    session: AdminSession = Depends(require_admin_session),
) -> AdminSession:
    if not csrf_header or not csrf_cookie or not hmac.compare_digest(csrf_header, csrf_cookie) or not hmac.compare_digest(csrf_header, session.csrf_token):
        raise HTTPException(403, "CSRF validation failed.")
    return session


def clear_admin_session(response: Response, request: Request, db: Session, session_cookie: str | None) -> None:
    if session_cookie:
        session = db.scalar(select(AdminSession).where(AdminSession.token_hash == _hash_token(session_cookie)))
        if session is not None:
            db.delete(session)
            db.commit()
    secure = _cookie_secure(request)
    response.delete_cookie(SESSION_COOKIE, secure=secure, samesite="none" if secure else "lax", path="/")
    response.delete_cookie(CSRF_COOKIE, secure=secure, samesite="none" if secure else "lax", path="/")
