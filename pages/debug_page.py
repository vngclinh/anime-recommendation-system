import streamlit as st
from recommender import recommend_for_anime, load_artifacts


def _init_debug_state():
    st.session_state.setdefault("debug_query", "")
    st.session_state.setdefault("debug_mood", "Normal")
    st.session_state.setdefault("debug_selected_name", "")
    st.session_state.setdefault("debug_alpha", 0.4)
    st.session_state.setdefault("debug_top_k", 10)
    st.session_state.setdefault("debug_cf_candidates", 200)
    st.session_state.setdefault("debug_recs_key", None)
    st.session_state.setdefault("debug_recs", None)


def render():
    st.title("Debug Recommender")

    _init_debug_state()

    pack = load_artifacts()
    items = pack["items"]

    # ===== FILTER BAR =====
    with st.container():
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            mood_options = [
                "Normal",
                "Happy / Funny",
                "Sad / Reflective",
                "Hyped / Intense",
                "Relaxed / Chill",
                "Curious / Mind-bending",
            ]
            saved_mood = st.session_state.get("debug_mood", mood_options[0])
            mood_index = mood_options.index(saved_mood) if saved_mood in mood_options else 0
            mood = st.selectbox(
                "Mood",
                mood_options,
                index=mood_index,
                key="debug_mood_input",
            )
            st.session_state["debug_mood"] = mood

        with c2:
            alpha = st.slider(
                "alpha",
                0.0,
                1.0,
                value=st.session_state["debug_alpha"],
                step=0.05,
            )
            st.session_state["debug_alpha"] = alpha

        with c3:
            top_k = st.slider("Top-K", 5, 30, value=st.session_state["debug_top_k"])
            st.session_state["debug_top_k"] = top_k

        with c4:
            cf_candidates = st.slider(
                "CF candidates",
                50,
                1000,
                value=st.session_state["debug_cf_candidates"],
                step=50,
            )
            st.session_state["debug_cf_candidates"] = cf_candidates

    st.divider()

    # ===== SEARCH =====
    query = st.text_input(
        "Search anime",
        value=st.session_state["debug_query"],
        key="debug_query_input",
    )
    st.session_state["debug_query"] = query

    subset = items[items["name"].str.contains(query, case=False, na=False)].head(50)
    label = subset.apply(
        lambda r: f"{r['name']} | {r['type']}", axis=1
    )

    label_options = label.tolist()
    if not label_options:
        st.info("No matching anime found.")
        return

    selected_name = st.session_state.get("debug_selected_name", "")
    selected_index = label_options.index(selected_name) if selected_name in label_options else 0
    selected = st.selectbox("Select anime", label_options, index=selected_index, key="debug_selected_input")
    st.session_state["debug_selected_name"] = selected
    anime_id = int(subset.iloc[label_options.index(selected)]["anime_id"])

    # ===== RECOMMEND =====
    recs_key = (anime_id, mood, alpha, top_k, cf_candidates)
    recs = None
    if st.session_state.get("debug_recs_key") == recs_key:
        recs = st.session_state.get("debug_recs")
    if recs is None:
        recs = recommend_for_anime(
            pack=pack,
            query_anime_id=anime_id,
            mood=mood,
            alpha=alpha,
            top_k=top_k,
            cf_candidates=cf_candidates,
        )
        st.session_state["debug_recs_key"] = recs_key
        st.session_state["debug_recs"] = recs

    st.dataframe(recs, use_container_width=True)
