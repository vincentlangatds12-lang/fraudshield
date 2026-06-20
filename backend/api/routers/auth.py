"""
Authentication router — simple JWT session cookie (username/password).
Matches the IVR pattern: HttpOnly cookie, require_session dependency.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Cookie, HTTPException, Response
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from pydantic import BaseModel

from config.settings import SESSION_SECRET_KEY, SESSION_EXPIRE_SECONDS, FRONTEND_ORIGIN

router = APIRouter(tags=["Auth"])

_ALGORITHM   = "HS256"
_COOKIE_NAME = "fraud_session"

# Demo users — in production, this would hit a real user store
_DEMO_USERS = {
    "analyst@umba.com": {"password": "umba2026", "name": "Fraud Analyst", "role": "analyst"},
    "admin@umba.com":   {"password": "admin2026", "name": "Admin",         "role": "admin"},
    "demo@umba.com":    {"password": "demo",       "name": "Demo User",    "role": "viewer"},
}


class LoginRequest(BaseModel):
    email:    str
    password: str


def _create_token(email: str, name: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(seconds=SESSION_EXPIRE_SECONDS)
    return jwt.encode(
        {"sub": email, "name": name, "role": role,
         "exp": expire, "iat": datetime.now(timezone.utc)},
        SESSION_SECRET_KEY, algorithm=_ALGORITHM,
    )


def _decode_token(token: str) -> dict:
    return jwt.decode(token, SESSION_SECRET_KEY, algorithms=[_ALGORITHM])


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=_COOKIE_NAME, value=token,
        httponly=True, secure=False, samesite="lax",
        max_age=SESSION_EXPIRE_SECONDS, path="/",
    )


@router.post("/auth/login")
def login(body: LoginRequest, response: Response):
    user = _DEMO_USERS.get(body.email.lower().strip())
    if not user or user["password"] != body.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = _create_token(body.email, user["name"], user["role"])
    resp  = JSONResponse({"email": body.email, "name": user["name"], "role": user["role"]})
    _set_cookie(resp, token)
    return resp


@router.get("/auth/me")
def get_me(fraud_session: Optional[str] = Cookie(default=None)):
    if not fraud_session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = _decode_token(fraud_session)
        return {"email": payload["sub"], "name": payload.get("name"), "role": payload.get("role")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Session expired")


@router.post("/auth/logout")
def logout():
    resp = JSONResponse({"status": "logged out"})
    resp.delete_cookie(key=_COOKIE_NAME, path="/")
    return resp


def require_session(fraud_session: Optional[str] = Cookie(default=None)) -> dict:
    """FastAPI dependency — injected at router level (same pattern as IVR)."""
    if not fraud_session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        return _decode_token(fraud_session)
    except JWTError:
        raise HTTPException(status_code=401, detail="Session expired")
