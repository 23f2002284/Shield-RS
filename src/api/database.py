"""
src/api/database.py
====================
SQLite-backed user management with real auth for Shield v3.

Tables:
  users         — id, name, email, password_hash, location, language,
                   preferred_topics, content_strictness, dark_mode,
                   preferences, created_at, last_login
  watch_history — user_id, video_id, title, channel, thumbnail,
                   duration_seconds, watched_at, watch_pct, agent_scores
"""

import sqlite3
import json
import uuid
import os
import hashlib
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional

# bcrypt-like hashing using hashlib (no extra dependency)
# For production, install and use bcrypt. This is a secure fallback.


def _hash_password(password: str) -> str:
    """Hash a password with a random salt using PBKDF2-SHA256."""
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return f"{salt}${dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    try:
        salt, hash_hex = stored_hash.split("$", 1)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return dk.hex() == hash_hex
    except Exception:
        return False


DB_PATH = Path(os.getenv("SHIELD_DB_PATH", "data/shield.db"))


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id                  TEXT PRIMARY KEY,
            name                TEXT NOT NULL,
            email               TEXT UNIQUE NOT NULL,
            password_hash       TEXT NOT NULL,
            location            TEXT NOT NULL DEFAULT '',
            language            TEXT NOT NULL DEFAULT 'en',
            preferred_topics    TEXT NOT NULL DEFAULT '[]',
            content_strictness  TEXT NOT NULL DEFAULT 'balanced',
            dark_mode           INTEGER NOT NULL DEFAULT 0,
            preferences         TEXT NOT NULL DEFAULT '{}',
            created_at          TEXT NOT NULL,
            last_login          TEXT
        );

        CREATE TABLE IF NOT EXISTS watch_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         TEXT NOT NULL,
            video_id        TEXT NOT NULL,
            title           TEXT,
            channel         TEXT,
            thumbnail       TEXT,
            duration_seconds INTEGER DEFAULT 0,
            watched_at      TEXT NOT NULL,
            watch_pct       REAL DEFAULT 0.0,
            agent_scores    TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE INDEX IF NOT EXISTS idx_history_user
            ON watch_history(user_id, watched_at DESC);
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def register_user(
    name: str,
    email: str,
    password: str,
    location: str = "",
    language: str = "en",
    preferred_topics: list[str] = None,
    content_strictness: str = "balanced",
) -> dict:
    """Register a new user. Raises ValueError if email already exists."""
    conn = _get_conn()

    # Check if email exists
    existing = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        conn.close()
        raise ValueError("Email already registered")

    user_id = str(uuid.uuid4())[:8]
    pw_hash = _hash_password(password)
    topics_json = json.dumps(preferred_topics or [])
    now = datetime.utcnow().isoformat()

    conn.execute(
        """INSERT INTO users
           (id, name, email, password_hash, location, language,
            preferred_topics, content_strictness, created_at, last_login)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, name, email, pw_hash, location, language,
         topics_json, content_strictness, now, now),
    )
    conn.commit()
    conn.close()

    return {
        "id": user_id,
        "name": name,
        "email": email,
        "location": location,
        "language": language,
        "preferred_topics": preferred_topics or [],
        "content_strictness": content_strictness,
        "dark_mode": False,
        "created_at": now,
    }


def login_user(email: str, password: str) -> Optional[dict]:
    """Authenticate a user. Returns user dict or None."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not row:
        conn.close()
        return None

    if not _verify_password(password, row["password_hash"]):
        conn.close()
        return None

    # Update last_login
    now = datetime.utcnow().isoformat()
    conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, row["id"]))
    conn.commit()
    conn.close()

    return _row_to_user(row)


def _row_to_user(row) -> dict:
    """Convert a DB row to a user dict (never includes password_hash)."""
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "location": row["location"],
        "language": row["language"],
        "preferred_topics": json.loads(row["preferred_topics"]),
        "content_strictness": row["content_strictness"],
        "dark_mode": bool(row["dark_mode"]),
        "preferences": json.loads(row["preferences"]),
        "created_at": row["created_at"],
        "last_login": row["last_login"],
    }


# ---------------------------------------------------------------------------
# Users (CRUD)
# ---------------------------------------------------------------------------

def create_user(name: str, preferences: Optional[dict] = None) -> dict:
    """Legacy create user (for backward compat). Use register_user instead."""
    return register_user(
        name=name,
        email=f"{name.lower().replace(' ', '.')}@shield.local",
        password="shield123",
        preferred_topics=[],
    )


def get_user(user_id: str) -> Optional[dict]:
    """Get a user by ID."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return _row_to_user(row)


def update_user_preferences(user_id: str, preferences: dict) -> bool:
    """Update a user's preferences."""
    conn = _get_conn()
    result = conn.execute(
        "UPDATE users SET preferences = ? WHERE id = ?",
        (json.dumps(preferences), user_id),
    )
    conn.commit()
    conn.close()
    return result.rowcount > 0


def update_user_settings(
    user_id: str,
    name: str = None,
    location: str = None,
    language: str = None,
    preferred_topics: list[str] = None,
    content_strictness: str = None,
    dark_mode: bool = None,
) -> Optional[dict]:
    """Update user settings. Only updates fields that are not None."""
    conn = _get_conn()
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return None

    updates = []
    params = []

    if name is not None:
        updates.append("name = ?")
        params.append(name)
    if location is not None:
        updates.append("location = ?")
        params.append(location)
    if language is not None:
        updates.append("language = ?")
        params.append(language)
    if preferred_topics is not None:
        updates.append("preferred_topics = ?")
        params.append(json.dumps(preferred_topics))
    if content_strictness is not None:
        updates.append("content_strictness = ?")
        params.append(content_strictness)
    if dark_mode is not None:
        updates.append("dark_mode = ?")
        params.append(int(dark_mode))

    if updates:
        params.append(user_id)
        conn.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()

    # Return updated user
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return _row_to_user(row)


def list_users() -> list[dict]:
    """List all users."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return [_row_to_user(r) for r in rows]


# ---------------------------------------------------------------------------
# Watch History
# ---------------------------------------------------------------------------

def log_watch(
    user_id: str,
    video_id: str,
    title: str = "",
    channel: str = "",
    thumbnail: str = "",
    duration_seconds: int = 0,
    watch_pct: float = 1.0,
    agent_scores: Optional[dict] = None,
) -> dict:
    """Log a video watch event for a user."""
    now = datetime.utcnow().isoformat()
    scores = json.dumps(agent_scores or {})

    conn = _get_conn()
    conn.execute(
        """INSERT INTO watch_history
           (user_id, video_id, title, channel, thumbnail,
            duration_seconds, watched_at, watch_pct, agent_scores)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, video_id, title, channel, thumbnail,
         duration_seconds, now, watch_pct, scores),
    )
    conn.commit()
    conn.close()
    return {
        "user_id": user_id,
        "video_id": video_id,
        "title": title,
        "watched_at": now,
    }


def get_history(user_id: str, limit: int = 50) -> list[dict]:
    """Get a user's watch history, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        """SELECT * FROM watch_history
           WHERE user_id = ?
           ORDER BY watched_at DESC
           LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    conn.close()
    return [
        {
            "video_id": r["video_id"],
            "title": r["title"],
            "channel": r["channel"],
            "thumbnail": r["thumbnail"],
            "duration_seconds": r["duration_seconds"],
            "watched_at": r["watched_at"],
            "watch_pct": r["watch_pct"],
            "agent_scores": json.loads(r["agent_scores"]),
        }
        for r in rows
    ]


def get_user_topic_stats(user_id: str) -> dict:
    """
    Compute topic distribution from a user's watch history.
    Returns counts of videos watched per inferred topic.
    """
    history = get_history(user_id, limit=200)
    topics: dict[str, int] = {}
    for h in history:
        scores = h.get("agent_scores", {})
        topic = scores.get("topic", "general")
        topics[topic] = topics.get(topic, 0) + 1
    return topics


# Initialize on import
init_db()
