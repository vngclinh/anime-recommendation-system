from typing import Optional, Set, List
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

RATING_BY_ACTION = {"watched": 10.0, "disliked": 0.0}

_sb: Optional[Client] = None


def _get_sb() -> Client:
    global _sb
    if _sb is not None:
        return _sb

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY).")

    _sb = create_client(url, key)
    return _sb


def log_event(
    username: str,
    action: str,
    anime_id: Optional[int] = None,
    mood: Optional[str] = None,
    seed_anime_id: Optional[int] = None,
    score: Optional[float] = None,
) -> None:
    try:
        sb = _get_sb()
        sb.table("user_events").insert({
            "username": username,
            "action": action,
            "anime_id": anime_id,
            "mood": mood,
            "seed_anime_id": seed_anime_id,
            "score": score,
        }).execute()
    except Exception as e:
        print("⚠️ log_event skipped:", e)


def log_search(username: str, seed_anime_id: int, mood: Optional[str] = None) -> None:
    log_event(username=username, action="search", seed_anime_id=int(seed_anime_id), mood=mood)


def set_item_state(username: str, anime_id: int, action: str) -> None:
    if action not in RATING_BY_ACTION:
        raise ValueError(f"Unknown action: {action}")

    try:
        sb = _get_sb()
        sb.table("user_ratings").upsert(
            {
                "username": username,
                "anime_id": int(anime_id),
                "rating": float(RATING_BY_ACTION[action]),
                "status": action,
            },
            on_conflict="username,anime_id",
        ).execute()
    except Exception as e:
        print("⚠️ set_item_state skipped:", e)


def get_blocked_ids(username: str) -> Set[int]:
    try:
        sb = _get_sb()
        resp = (
            sb.table("user_ratings")
            .select("anime_id,status")
            .eq("username", username)
            .in_("status", ["watched", "disliked"])
            .execute()
        )
        rows = resp.data or []
        return set(int(r["anime_id"]) for r in rows if r.get("anime_id") is not None)
    except Exception as e:
        print("⚠️ get_blocked_ids failed:", e)
        return set()


def get_recent_searches(username: str, limit: int = 8) -> List[int]:
    """
    DISTINCT seed_anime_id by latest created_at (giống GROUP BY MAX).
    """
    try:
        sb = _get_sb()
        resp = (
            sb.table("user_events")
            .select("seed_anime_id,created_at")
            .eq("username", username)
            .eq("action", "search")
            .not_.is_("seed_anime_id", "null")
            .order("created_at", desc=True)
            .limit(200)  # lấy nhiều rồi unique theo thứ tự
            .execute()
        )
        rows = resp.data or []
        out, seen = [], set()
        for r in rows:
            aid = r.get("seed_anime_id")
            if aid is None:
                continue
            aid = int(aid)
            if aid in seen:
                continue
            seen.add(aid)
            out.append(aid)
            if len(out) >= int(limit):
                break
        return out
    except Exception as e:
        print("⚠️ get_recent_searches failed:", e)
        return []


def delete_search(username: str, seed_anime_id: int) -> None:
    try:
        sb = _get_sb()
        (
            sb.table("user_events")
            .delete()
            .eq("username", username)
            .eq("action", "search")
            .eq("seed_anime_id", int(seed_anime_id))
            .execute()
        )
    except Exception as e:
        print("⚠️ delete_search failed:", e)
