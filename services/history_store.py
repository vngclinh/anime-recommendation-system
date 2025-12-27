from typing import Optional, Set
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from supabase import engine

# ===============================
# CONFIG
# ===============================
RATING_BY_ACTION = {
    "watched": 10.0,
    "disliked": 0.0,
}

_TABLES_READY = False


# ===============================
# INIT TABLES (SAFE)
# ===============================
def _ensure_tables() -> None:
    """
    Create tables if not exists.
    This function is SAFE:
    - Only runs once per process
    - Does NOT crash the app if DB is unavailable
    """
    global _TABLES_READY
    if _TABLES_READY:
        return

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS user_ratings (
                        username TEXT NOT NULL,
                        anime_id INTEGER NOT NULL,
                        rating REAL NOT NULL,
                        status TEXT NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (username, anime_id)
                    );
                    """
                )
            )
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS user_events (
                        id BIGSERIAL PRIMARY KEY,
                        username TEXT NOT NULL,
                        action TEXT NOT NULL,
                        anime_id INTEGER,
                        mood TEXT,
                        seed_anime_id INTEGER,
                        score REAL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
            )

        _TABLES_READY = True
        print("✅ DB tables ready")

    except SQLAlchemyError as e:
        # IMPORTANT: never crash Streamlit app
        print("⚠️ DB table init skipped:", e)


# ===============================
# LOGGING EVENTS (FAIL-SAFE)
# ===============================
def log_event(
    username: str,
    action: str,
    anime_id: Optional[int] = None,
    mood: Optional[str] = None,
    seed_anime_id: Optional[int] = None,
    score: Optional[float] = None,
) -> None:
    try:
        _ensure_tables()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO user_events (
                        username, action, anime_id, mood, seed_anime_id, score
                    ) VALUES (
                        :username, :action, :anime_id, :mood, :seed_anime_id, :score
                    );
                    """
                ),
                {
                    "username": username,
                    "action": action,
                    "anime_id": anime_id,
                    "mood": mood,
                    "seed_anime_id": seed_anime_id,
                    "score": score,
                },
            )
    except SQLAlchemyError as e:
        print("⚠️ log_event skipped:", e)


def log_search(username: str, seed_anime_id: int, mood: Optional[str] = None) -> None:
    log_event(
        username=username,
        action="search",
        seed_anime_id=seed_anime_id,
        mood=mood,
    )


# ===============================
# USER ACTIONS
# ===============================
def set_item_state(username: str, anime_id: int, action: str) -> None:
    if action not in RATING_BY_ACTION:
        raise ValueError(f"Unknown action: {action}")

    rating = RATING_BY_ACTION[action]

    try:
        _ensure_tables()
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO user_ratings (username, anime_id, rating, status)
                    VALUES (:username, :anime_id, :rating, :status)
                    ON CONFLICT (username, anime_id)
                    DO UPDATE SET
                        rating = EXCLUDED.rating,
                        status = EXCLUDED.status,
                        updated_at = NOW();
                    """
                ),
                {
                    "username": username,
                    "anime_id": anime_id,
                    "rating": rating,
                    "status": action,
                },
            )
    except SQLAlchemyError as e:
        print("⚠️ set_item_state skipped:", e)


# ===============================
# READ DATA
# ===============================
def get_blocked_ids(username: str) -> Set[int]:
    try:
        _ensure_tables()
        df = pd.read_sql(
            text(
                """
                SELECT anime_id
                FROM user_ratings
                WHERE username = :username
                  AND status IN ('watched', 'disliked');
                """
            ),
            engine,
            params={"username": username},
        )
        return set(df["anime_id"].astype(int).tolist())

    except SQLAlchemyError as e:
        print("⚠️ get_blocked_ids failed:", e)
        return set()
