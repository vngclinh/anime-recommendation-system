import streamlit as st
from recommender import recommend_for_anime, load_artifacts
from components.anime_card import anime_card
from services.anilist import anilist_cover_url
from services.history_store import set_item_state, log_event, get_blocked_ids, log_search


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
    </style>
    """, unsafe_allow_html=True)


def _init_demo_state():
    st.session_state.setdefault("demo_query", "")
    st.session_state.setdefault("demo_mood", "Normal")
    st.session_state.setdefault("demo_selected_name", "")
    st.session_state.setdefault("demo_recs_key", None)
    st.session_state.setdefault("demo_recs", None)


def render():
    inject_css()

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

        subset = (
            items[items["name"].str.contains(query, case=False, na=False)].head(20)
            if query.strip()
            else items.head(20)
        )

        if subset.empty:
            st.warning("No matching anime found.")
            return

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

    # ===== RECOMMEND (auto-updates, no Apply button needed) =====
    recs_key = (anime_id, mood, 10, 0.4, 200, 200)
    base_recs = None
    if st.session_state.get("demo_recs_key") == recs_key:
        base_recs = st.session_state.get("demo_recs")
    if base_recs is None:
        with st.spinner("Finding anime that fits you..."):
            base_recs = recommend_for_anime(
                pack=pack,
                query_anime_id=anime_id,
                mood=mood,
                top_k=10,
                alpha=0.4,
                cf_candidates=200,
                cb_candidates=200,
            )
        st.session_state["demo_recs_key"] = recs_key
        st.session_state["demo_recs"] = base_recs

    search_log_key = f"{username}:{anime_id}:{mood}"
    if st.session_state.get("last_search_key") != search_log_key:
        log_search(username=username, seed_anime_id=anime_id, mood=mood)
        st.session_state["last_search_key"] = search_log_key

    blocked = get_blocked_ids(username)
    if base_recs is None or base_recs.empty:
        st.info("No recommendations found.")
        return
    recs = base_recs[~base_recs["anime_id"].isin(blocked)]
    if recs.empty:
        st.info("No recommendations found.")
        return

    # ===== CARDS LIST =====
    for _, r in recs.iterrows():
        title = str(r["name"])
        img_url = anilist_cover_url(title) or "https://placehold.co/240x340?text=No+Cover"

        # Choose hybrid_score or like_pct_est for the match percentage.
        # match_score = float(r["like_pct_est"])
        match_score = min(float(r["hybrid_score"]), 1.0) * 100

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
                seed_anime_id=anime_id,
                score=float(r["hybrid_score"]),
            )
            st.toast(f"Saved: {action}")
            st.rerun()

        st.divider()
