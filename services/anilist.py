import requests
import streamlit as st

ANILIST_URL = "https://graphql.anilist.co"

QUERY = """
query ($search: String) {
  Media(search: $search, type: ANIME) {
    id
    title { romaji english native }
    coverImage { large medium }
  }
}
"""

@st.cache_data(ttl=24*3600, show_spinner=False)
def anilist_cover_url(title: str) -> str | None:
    title = (title or "").strip()
    if not title:
        return None

    try:
        resp = requests.post(
            ANILIST_URL,
            json={"query": QUERY, "variables": {"search": title}},
            timeout=10
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        media = (data.get("data") or {}).get("Media")
        if not media:
            return None

        cover = media.get("coverImage") or {}
        return cover.get("large") or cover.get("medium")

    except Exception:
        return None
