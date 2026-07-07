from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "user_accounts.sqlite3"
SESSION_COOKIE = "ionmix_session"
SESSION_DAYS = 30

_init_lock = threading.Lock()
_initialised = False


def database_path() -> Path:
    configured = os.getenv("IONMIX_AUTH_DB")
    if configured:
        return Path(configured)
    return DEFAULT_DB_PATH


def connect() -> sqlite3.Connection:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    global _initialised
    if _initialised:
        return
    with _init_lock:
        if _initialised:
            return
        with connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS saved_formulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    formula TEXT NOT NULL,
                    score REAL,
                    confidence REAL,
                    recommendation_json TEXT NOT NULL,
                    request_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_saved_formulations_user_created
                    ON saved_formulations(user_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_sessions_expires_at
                    ON sessions(expires_at);
                """
            )
        _initialised = True


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return f"pbkdf2_sha256$260000${salt.hex()}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:
        return False


def public_user(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "display_name": row["display_name"],
        "created_at": row["created_at"],
    }


def create_user(email: str, password: str, display_name: str | None = None) -> dict[str, Any]:
    init_db()
    clean_email = email.strip().lower()
    if len(password) < 8:
        raise ValueError("密码至少需要 8 位。")
    if "@" not in clean_email or "." not in clean_email:
        raise ValueError("请输入有效邮箱。")
    name = (display_name or clean_email.split("@")[0]).strip()[:40] or clean_email
    created_at = now_iso()
    try:
        with connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (email, display_name, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (clean_email, name, hash_password(password), created_at),
            )
            row = conn.execute(
                "SELECT id, email, display_name, created_at FROM users WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()
            return public_user(row)
    except sqlite3.IntegrityError as exc:
        raise ValueError("这个邮箱已经注册过了。") from exc


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    init_db()
    clean_email = email.strip().lower()
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (clean_email,)).fetchone()
    if not row or not verify_password(password, row["password_hash"]):
        return None
    return public_user(row)


def create_session(user_id: int) -> dict[str, str]:
    init_db()
    token = secrets.token_urlsafe(36)
    created_at = datetime.now(UTC).replace(microsecond=0)
    expires_at = created_at + timedelta(days=SESSION_DAYS)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (token, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, user_id, created_at.isoformat(), expires_at.isoformat()),
        )
    return {"token": token, "expires_at": expires_at.isoformat()}


def delete_session(token: str | None) -> None:
    if not token:
        return
    init_db()
    with connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def user_from_session(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None
    init_db()
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    with connect() as conn:
        row = conn.execute(
            """
            SELECT users.id, users.email, users.display_name, users.created_at
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ? AND sessions.expires_at > ?
            """,
            (token, now),
        ).fetchone()
        conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
    return public_user(row) if row else None


def formula_from_recommendation(recommendation: dict[str, Any]) -> str:
    components = recommendation.get("components") or []
    if components:
        return " + ".join(
            f"{component.get('code', '?')} {component.get('ratio', '?')}%"
            for component in components
        )
    return " + ".join(
        filter(None, [recommendation.get("solvent_a"), recommendation.get("solvent_b")])
    )


def save_formulation(
    *,
    user_id: int,
    name: str,
    recommendation: dict[str, Any],
    request_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    init_db()
    clean_name = name.strip()[:80]
    if not clean_name:
        raise ValueError("请给这个配方起一个名字。")
    formula = formula_from_recommendation(recommendation)
    created_at = now_iso()
    with connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO saved_formulations
                (user_id, name, formula, score, confidence, recommendation_json, request_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                clean_name,
                formula,
                recommendation.get("score"),
                recommendation.get("confidence"),
                json.dumps(recommendation, ensure_ascii=False),
                json.dumps(request_context or {}, ensure_ascii=False),
                created_at,
            ),
        )
    return {
        "id": int(cursor.lastrowid),
        "name": clean_name,
        "formula": formula,
        "score": recommendation.get("score"),
        "confidence": recommendation.get("confidence"),
        "created_at": created_at,
        "recommendation": recommendation,
        "request_context": request_context or {},
    }


def list_formulations(user_id: int, limit: int = 80) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, name, formula, score, confidence, recommendation_json, request_json, created_at
            FROM saved_formulations
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "name": row["name"],
            "formula": row["formula"],
            "score": row["score"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
            "recommendation": json.loads(row["recommendation_json"]),
            "request_context": json.loads(row["request_json"] or "{}"),
        }
        for row in rows
    ]


def delete_formulation(user_id: int, formulation_id: int) -> bool:
    init_db()
    with connect() as conn:
        cursor = conn.execute(
            "DELETE FROM saved_formulations WHERE user_id = ? AND id = ?",
            (user_id, formulation_id),
        )
    return cursor.rowcount > 0
