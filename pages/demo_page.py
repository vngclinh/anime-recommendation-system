import streamlit as st
from recommender import recommend_for_anime_cached, load_artifacts, init_shared_cache
from components.anime_card import anime_card
from services.anilist import anilist_cover_url
from services.history_store import (
    set_item_state,
    log_event,
    get_blocked_ids,
    log_search,
    get_recent_searches,
    get_disliked_ids,
)
from recommender import apply_anti_cluster_penalty

def inject_css():
    st.markdown("""
    <style>
      .stApp {
        background: radial-gradient(1200px 600px at 50% -10%, #6b4bbd 0%, #2a1f55 35%, #121225 100%);
      }
      header {visibility: hidden;}
      .block-container {padding-top: 2.0rem;}

      .hero h1{
        text-align:center;
        font-weight: 800;
        letter-spacing: 0.3px;
        color: #ffffff;
        margin-bottom: 0.25rem;
      }
      .hero p{
        text-align:center;
        color: rgba(255,255,255,0.75);
        margin-top: 0;
      }

      .section-title{
        text-align:center;
        font-size: 2.0rem;
        font-weight: 800;
        color: rgba(255,255,255,0.92);
        margin: 1.5rem 0 1.0rem 0;
      }

      /* --- General Buttons --- */
      div.stButton > button {
        width: 100%;
        border-radius: 999px !important;
        border: 1px solid rgba(155, 115, 255, 0.35) !important;
        background: rgba(120, 88, 230, 0.25) !important;
        color: rgba(255,255,255,0.92) !important;
        font-weight: 700 !important;
        padding: 0.55rem 0.9rem !important;
      }
      div.stButton > button:hover{
        border-color: rgba(180, 140, 255, 0.7) !important;
        background: rgba(120, 88, 230, 0.35) !important;
      }

      .recent-title{
        font-size: 0.85rem;
        font-weight: 600;
        color: rgba(255,255,255,0.55);
        margin: 0.6rem 0 0.4rem 0;
      }

      div[data-testid="stVerticalBlock"]:has(.recent-searches) div[data-testid="stButton"] > button{
        width: 200px !important;
        max-width: 200px !important;
        min-height: 36px !important;
        border-radius: 999px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        background: linear-gradient(90deg, #4c57d6, #6a5ce7) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        padding: 0.45rem 0.9rem !important;
        box-shadow: 0 6px 14px rgba(15, 12, 40, 0.35);
        display: inline-flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
      }
      div[data-testid="stVerticalBlock"]:has(.recent-searches) div[data-testid="stButton"] > button > div{
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
      }
      div[data-testid="stVerticalBlock"]:has(.recent-searches) div[data-testid="stButton"] > button:hover{
        background: linear-gradient(90deg, #5b66f0, #7b6cff) !important;
      }
      div[data-testid="stVerticalBlock"]:has(.recent-searches) div[data-testid="stHorizontalBlock"]{
        gap: 5px;
      }
      div[data-testid="stVerticalBlock"]:has(.recent-searches) div[data-testid="stButton"]{
        margin-bottom: 5px;
      }

      /* --- Inputs --- */
      .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(15, 16, 34, 0.55);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
      }
      .stTextInput input{
        background: rgba(15, 16, 34, 0.55);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        color: rgba(255,255,255,0.92);
      }

      hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(160,120,255,0.6), transparent);
        margin: 1.2rem 0;
      }
/* ---- RECENT SEARCH CHIPS (Compact like Image 2) ---- */
/* ===== RECENT SEARCH – PILL STYLE (MATCH IMAGE 2) ===== */

    </style>
    """, unsafe_allow_html=True)


def _init_demo_state():
    st.session_state.setdefault("demo_query", "")
    st.session_state.setdefault("demo_mood", "Normal")
    st.session_state.setdefault("demo_selected_name", "")
    st.session_state.setdefault("demo_recs_key", None)
    st.session_state.setdefault("demo_recs", None)


def _apply_history_selection(name: str, anime_id: int) -> None:
    # safe: set text input value + internal query
    st.session_state["demo_query"] = name
    st.session_state["demo_query_input"] = name

    # keep selected name for default index
    st.session_state["demo_selected_name"] = name

    # IMPORTANT: remove selectbox stored state to avoid "value not in options" error
    st.session_state.pop("demo_selected_input", None)

    # shared for debug
    st.session_state["shared_search_query"] = name
    st.session_state["shared_selected_anime_id"] = int(anime_id)


def _render_recent_searches_legacy(items, username: str, limit: int = 8):
    recent_ids = get_recent_searches(username, limit=limit)
    if not recent_ids:
        return

    id_to_name = dict(zip(items["anime_id"].astype(int), items["name"].astype(str)))

    st.markdown(
        "<div style='font-size:0.85rem;font-weight:600;color:rgba(255,255,255,0.55);margin:0.6rem 0;'>Recent Searches</div>",
        unsafe_allow_html=True
    )

    html = "<div class='recent-wrap'>"

    for aid in recent_ids:
        aid = int(aid)
        name = id_to_name.get(aid, "")
        safe = name.replace('"', "&quot;")

        html += f"""
        <div class="recent-pill"
             onclick="window.parent.postMessage({{type:'pick', aid:{aid}}}, '*')">
          <span>{safe}</span>
          <div class="pill-x"
               onclick="event.stopPropagation();
                        window.parent.postMessage({{type:'del', aid:{aid}}}, '*')">
            ✕
          </div>
        </div>
        """

    html += "</div>"


def _render_recent_searches(items, username: str, limit: int = 8):
    recent_ids = get_recent_searches(username, limit=limit)
    if not recent_ids:
        return

    id_to_name = dict(zip(items["anime_id"].astype(int), items["name"].astype(str)))

    with st.container():
        st.markdown("<div class='recent-searches'></div>", unsafe_allow_html=True)
        st.markdown("<div class='recent-title'>Recent Searches</div>", unsafe_allow_html=True)

        per_row = 4
        max_rows = 2
        display_ids = recent_ids[: per_row * max_rows]

        for start in range(0, len(display_ids), per_row):
            row_ids = display_ids[start:start + per_row]
            row_cols = st.columns(len(row_ids))
            for col, aid in zip(row_cols, row_ids):
                name = id_to_name.get(int(aid))
                if not name:
                    continue

                with col:
                    if st.button(name, key=f"recent_pick_{aid}"):
                        _apply_history_selection(name, int(aid))
                        st.rerun()


def render():
    inject_css()
    init_shared_cache()

    username = st.session_state.get("username", "").strip()
    if not username:
        st.warning("Enter a username in the sidebar to continue.")
        st.stop()

    _init_demo_state()

    pack = load_artifacts()
    items = pack["items"]

    # ===== HERO =====
    st.markdown("""
      <div class="hero">
        <h1>Just one episode...</h1>
        <p>Discover your next favorite anime</p>
      </div>
    """, unsafe_allow_html=True)

    # ===== MOVED: Recent searches call removed from here =====

    st.markdown("<div class='section-title'>Recommended for You</div>", unsafe_allow_html=True)

    # ===== SEARCH + FILTER BAR =====
    c1, c2 = st.columns([3, 1.3])
    with c1:
        query = st.text_input(
            "Search anime",
            placeholder="Example: Naruto / One Piece / Death Note ...",
            label_visibility="collapsed",
            value=st.session_state["demo_query"],
            key="demo_query_input",
        )
        st.session_state["demo_query"] = query
        st.session_state["shared_search_query"] = query

        # ===== UPDATED: Render Recent Searches HERE (Under search bar) =====
        _render_recent_searches(items, username, limit=4) 
        # ===================================================================

        subset = (
            items[items["name"].str.contains(query, case=False, na=False)].head(20)
            if query.strip()
            else items.head(20)
        )

        if subset.empty:
            if query.strip():
                st.warning("No matching anime found.")
            # If empty query, we just wait (or show nothing special)
        else:
            options = subset["name"].tolist()
            selected_name = st.session_state.get("demo_selected_name", "")
            selected_index = options.index(selected_name) if selected_name in options else 0

            selected = st.selectbox(
                "Choose anime",
                options,
                index=selected_index,
                label_visibility="collapsed",
                key="demo_selected_input",
            )
            st.session_state["demo_selected_name"] = selected
            anime_id = int(subset.loc[subset["name"] == selected, "anime_id"].iloc[0])
            st.session_state["shared_selected_anime_id"] = int(anime_id)

    with c2:
        mood_options = [
            "Normal",
            "Happy / Funny",
            "Sad / Reflective",
            "Hyped / Intense",
            "Relaxed / Chill",
            "Curious / Mind-bending",
        ]
        saved_mood = st.session_state.get("demo_mood", mood_options[0])
        mood_index = mood_options.index(saved_mood) if saved_mood in mood_options else 0
        mood = st.selectbox(
            "Mood",
            mood_options,
            index=mood_index,
            label_visibility="collapsed",
            key="demo_mood_input",
        )
        st.session_state["demo_mood"] = mood

        st.session_state["shared_mood"] = mood
        st.session_state["shared_alpha"] = 0.4
        st.session_state["shared_top_k"] = 10
        st.session_state["shared_cf_candidates"] = 200
        st.session_state["shared_cb_candidates"] = 200

    # Safety check if no anime selected yet
    if 'anime_id' not in locals():
        return

    # ===== RECOMMEND =====
    alpha = 0.4
    top_k = 10
    top_k_raw = max(60, top_k * 8)   # lấy nhiều để rerank sau khi phạt
    cf_candidates = 200
    cb_candidates = 200

    recs_key = (int(anime_id), str(mood), float(alpha), int(top_k_raw), int(cf_candidates), int(cb_candidates))

    base_recs = None
    if st.session_state.get("demo_recs_key") == recs_key:
        base_recs = st.session_state.get("demo_recs")

    if base_recs is None:
        with st.spinner("Finding anime that fits you..."):
          base_recs = recommend_for_anime_cached(
              query_anime_id=int(anime_id),
              mood=str(mood),
              top_k=int(top_k_raw),        # <<< dùng raw
              alpha=float(alpha),
              cf_candidates=int(cf_candidates),
              cb_candidates=int(cb_candidates),
          )
        st.session_state["demo_recs_key"] = recs_key
        st.session_state["demo_recs"] = base_recs

    st.session_state["shared_recs_key"] = recs_key
    st.session_state["shared_recs_df"] = base_recs

    # log search (fail-safe in history_store now)
    search_log_key = f"{username}:{anime_id}:{mood}"
    if st.session_state.get("last_search_key") != search_log_key:
        log_search(username=username, seed_anime_id=int(anime_id), mood=str(mood))
        st.session_state["last_search_key"] = search_log_key

    blocked = get_blocked_ids(username)
    if base_recs is None or base_recs.empty:
        st.info("No recommendations found.")
        return

    recs = base_recs[~base_recs["anime_id"].isin(blocked)]
    # --- Anti-Cluster Penalization (Dislike penalty) ---
    disliked_ids = get_disliked_ids(username, limit=50)

    beta = 0.15  # có thể làm slider sau
    recs = apply_anti_cluster_penalty(
        pack=pack,
        recs=recs,
        disliked_ids=disliked_ids,
        beta=float(beta),
    )

    # giữ đúng top_k sau khi rerank
    recs = recs.head(top_k)

    if recs.empty:
        st.info("No recommendations found.")
        return

    # ===== CARDS LIST =====
    for _, r in recs.iterrows():
        title = str(r["name"])
        img_url = anilist_cover_url(title) or "https://placehold.co/240x340?text=No+Cover"
        score_for_ui = float(r["final_score"]) if "final_score" in r else float(r["hybrid_score"])
        match_score = min(score_for_ui, 1.0) * 100

        action = anime_card(
            img_url=img_url,
            title=title,
            genres=[g.strip() for g in str(r["genre_overlap"]).split(",")] if str(r["genre_overlap"]) != "None" else [],
            like_pct=match_score,
            anime_id=int(r["anime_id"]),
        )

        if action in ("watched", "disliked"):
            aid = int(r["anime_id"])
            set_item_state(username, aid, action)
            log_event(
                username=username,
                anime_id=aid,
                action=action,
                mood=mood,
                seed_anime_id=int(anime_id),
                score=float(r["hybrid_score"]),
            )
            st.toast(f"Saved: {action}")
            st.rerun()

        st.divider()
