from typing import Optional, Set

import pandas as pd
from sqlalchemy import text
from supabase import engine

RATING_BY_ACTION = {
    "watched": 10.0,
    "disliked": 0.0,
}

_TABLES_READY = False


def _ensure_tables() -> None:
    global _TABLES_READY
    if _TABLES_READY:
        return

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                create table if not exists user_ratings (
                    username text not null,
                    anime_id integer not null,
                    rating real not null,
                    status text not null,
                    updated_at timestamptz not null default now(),
                    primary key (username, anime_id)
                );
                """
            )
        )
        conn.execute(
            text(
                """
                create table if not exists user_events (
                    id bigserial primary key,
                    username text not null,
                    action text not null,
                    anime_id integer,
                    mood text,
                    seed_anime_id integer,
                    score real,
                    created_at timestamptz not null default now()
                );
                """
            )
        )

    _TABLES_READY = True


def log_event(
    username: str,
    action: str,
    anime_id: Optional[int] = None,
    mood: Optional[str] = None,
    seed_anime_id: Optional[int] = None,
    score: Optional[float] = None,
) -> None:
    _ensure_tables()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into user_events (
                    username, action, anime_id, mood, seed_anime_id, score
                ) values (
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


def log_search(username: str, seed_anime_id: int, mood: Optional[str] = None) -> None:
    log_event(
        username=username,
        action="search",
        seed_anime_id=seed_anime_id,
        mood=mood,
    )


def set_item_state(username: str, anime_id: int, action: str) -> None:
    if action not in RATING_BY_ACTION:
        raise ValueError(f"Unknown action: {action}")

    rating = RATING_BY_ACTION[action]
    _ensure_tables()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into user_ratings (username, anime_id, rating, status)
                values (:username, :anime_id, :rating, :status)
                on conflict (username, anime_id)
                do update set
                    rating = excluded.rating,
                    status = excluded.status,
                    updated_at = now();
                """
            ),
            {
                "username": username,
                "anime_id": anime_id,
                "rating": rating,
                "status": action,
            },
        )


def get_blocked_ids(username: str) -> Set[int]:
    _ensure_tables()
    df = pd.read_sql(
        text(
            """
            select anime_id
            from user_ratings
            where username = :username
              and status in ('watched', 'disliked');
            """
        ),
        engine,
        params={"username": username},
    )
    return set(df["anime_id"].astype(int).tolist())
