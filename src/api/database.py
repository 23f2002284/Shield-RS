"""
src/api/database.py
====================
SQLite-backed user management and watch history for Shield.

Tables:
  users         — id, name, preferences (JSON), created_at
  watch_history — user_id, video_id, title, channel, thumbnail, 
                  duration_seconds, watched_at, watch_pct, agent_scores (JSON)
"""

import sqlite3
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

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
            id            TEXT PRIMARY KEY,
            name          TEXT NOT NULL,
            preferences   TEXT NOT NULL DEFAULT '{}',
            created_at    TEXT NOT NULL
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
# Users
# ---------------------------------------------------------------------------

def create_user(name: str, preferences: Optional[dict] = None) -> dict:
    """Create a new user profile. Returns the user dict."""
    user_id = str(uuid.uuid4())[:8]
    prefs = json.dumps(preferences or {})
    now = datetime.utcnow().isoformat()

    conn = _get_conn()
    conn.execute(
        "INSERT INTO users (id, name, preferences, created_at) VALUES (?, ?, ?, ?)",
        (user_id, name, prefs, now),
    )
    conn.commit()
    conn.close()
    return {"id": user_id, "name": name, "preferences": preferences or {}, "created_at": now}


def get_user(user_id: str) -> Optional[dict]:
    """Get a user by ID."""
    conn = _get_conn()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "preferences": json.loads(row["preferences"]),
        "created_at": row["created_at"],
    }


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


def list_users() -> list[dict]:
    """List all users."""
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "preferences": json.loads(r["preferences"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]


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
        # Use agent_scores topic if available, else "unknown"
        scores = h.get("agent_scores", {})
        topic = scores.get("topic", "general")
        topics[topic] = topics.get(topic, 0) + 1
    return topics


# Initialize on import
init_db()
