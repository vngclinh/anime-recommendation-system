import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from recommender import init_shared_cache, recommend_for_anime, load_artifacts


def _init_debug_state():
    init_shared_cache()

    # defaults from shared (so Debug opens with Demo state)
    st.session_state.setdefault("debug_query", st.session_state.get("shared_search_query", ""))
    st.session_state.setdefault("debug_selected_anime_id", st.session_state.get("shared_selected_anime_id", None))
    st.session_state.setdefault("debug_mood", st.session_state.get("shared_mood", "Normal"))
    st.session_state.setdefault("debug_alpha", st.session_state.get("shared_alpha", 0.4))
    st.session_state.setdefault("debug_top_k", st.session_state.get("shared_top_k", 10))
    st.session_state.setdefault("debug_cf_candidates", st.session_state.get("shared_cf_candidates", 200))

    # local cache for debug
    st.session_state.setdefault("debug_recs_key", None)
    st.session_state.setdefault("debug_recs", None)

    # explain UI state (avoid KeyError)
    st.session_state.setdefault("debug_show_explain", True)
    st.session_state.setdefault("debug_explain_top_n", 10)


def render_explain_charts(recs: pd.DataFrame, top_n: int = 10):
    needed = {"name", "cf_part", "content_part", "mood_bonus", "hybrid_score"}
    if recs is None or recs.empty:
        st.info("No recommendations to explain yet.")
        return

    if not needed.issubset(set(recs.columns)):
        st.warning(
            "Explain charts cannot render because required columns are missing.\n"
            "Please update recommender.py to return: "
            + ", ".join(sorted(list(needed)))
        )
        return

    tmp = recs.copy().head(top_n).copy()

    for col in ["cf_part", "content_part", "mood_bonus", "hybrid_score"]:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    tmp = tmp.dropna(subset=["cf_part", "content_part", "mood_bonus", "hybrid_score"])
    if tmp.empty:
        st.warning("All rows became NaN after numeric conversion; cannot plot.")
        return

    tmp = tmp.sort_values("hybrid_score", ascending=True)

    # 1) stacked bar contributions
    fig_h = plt.figure(figsize=(14, max(4, 0.6 * len(tmp))))
    y = np.arange(len(tmp))

    plt.barh(y, tmp["cf_part"].values, label="CF contribution")
    plt.barh(y, tmp["content_part"].values, left=tmp["cf_part"].values, label="Content contribution")
    left2 = (tmp["cf_part"] + tmp["content_part"]).values
    plt.barh(y, tmp["mood_bonus"].values, left=left2, label="Mood bonus")

    plt.yticks(y, tmp["name"].astype(str).values)
    plt.xlabel("Contribution to final score")
    plt.title("Score Breakdown: CF vs Content vs Mood")
    plt.legend()
    st.pyplot(fig_h, clear_figure=True)

    # 2) scatter optional
    if {"cf_norm", "content_norm"}.issubset(set(tmp.columns)):
        tmp["cf_norm"] = pd.to_numeric(tmp["cf_norm"], errors="coerce")
        tmp["content_norm"] = pd.to_numeric(tmp["content_norm"], errors="coerce")
        tmp2 = tmp.dropna(subset=["cf_norm", "content_norm"])

        if not tmp2.empty and len(tmp2) > 1:
            fig_s = plt.figure(figsize=(9, 6))
            plt.scatter(tmp2["cf_norm"].values, tmp2["content_norm"].values, s=70)
            for _, r in tmp2.iterrows():
                plt.text(float(r["cf_norm"]), float(r["content_norm"]), str(r["name"])[:18], fontsize=8)

            plt.xlabel("CF norm (used in score)")
            plt.ylabel("Content norm (used in score)")
            plt.title("CF vs Content (Top Recommendations)")
            st.pyplot(fig_s, clear_figure=True)

    # 3) sanity check
    tmp["final_check"] = tmp["cf_part"] + tmp["content_part"] + tmp["mood_bonus"]
    gap = float(np.mean(np.abs(tmp["final_check"] - tmp["hybrid_score"])))
    st.caption(f"Sanity check: mean(|(cf_part+content_part+mood_bonus) - hybrid_score|) = {gap:.6f}")


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
            mood = st.selectbox("Mood", mood_options, index=mood_index, key="debug_mood_input")
            st.session_state["debug_mood"] = mood

        with c2:
            alpha = st.slider("alpha", 0.0, 1.0, value=st.session_state["debug_alpha"], step=0.05)
            st.session_state["debug_alpha"] = alpha

        with c3:
            top_k = st.slider("Top-K", 5, 30, value=st.session_state["debug_top_k"])
            st.session_state["debug_top_k"] = top_k

        with c4:
            cf_candidates = st.slider("CF candidates", 50, 1000, value=st.session_state["debug_cf_candidates"], step=50)
            st.session_state["debug_cf_candidates"] = cf_candidates

    st.divider()

    # ===== SEARCH =====
    query = st.text_input("Search anime", value=st.session_state["debug_query"], key="debug_query_input")
    st.session_state["debug_query"] = query

    subset = (
        items[items["name"].str.contains(query, case=False, na=False)].head(50).copy()
        if query.strip()
        else items.head(50).copy()
    )

    if subset.empty:
        st.info("No matching anime found.")
        return

    subset["label"] = subset.apply(lambda r: f"{r['name']} | {r['type']}", axis=1)
    labels = subset["label"].tolist()
    label_to_id = dict(zip(labels, subset["anime_id"].astype(int).tolist()))

    # default selection based on debug_selected_anime_id (from shared)
    default_aid = st.session_state.get("debug_selected_anime_id", None)
    default_label = None
    if default_aid is not None:
        for lab, aid in label_to_id.items():
            if int(aid) == int(default_aid):
                default_label = lab
                break

    selected_index = labels.index(default_label) if default_label in labels else 0
    selected = st.selectbox("Select anime", labels, index=selected_index, key="debug_selected_input")
    anime_id = int(label_to_id[selected])

    st.session_state["debug_selected_anime_id"] = int(anime_id)

    # ===== RECOMMEND =====
    # keep cb_candidates same as Demo so shared cache matches
    cb_candidates = 200

    # ✅ canonical key MUST match Demo: (anime_id, mood, alpha, top_k, cf_candidates, cb_candidates)
    recs_key = (int(anime_id), str(mood), float(alpha), int(top_k), int(cf_candidates), int(cb_candidates))

    # 1) if Demo already computed exactly this key -> reuse instantly
    shared_key = st.session_state.get("shared_recs_key")
    shared_df = st.session_state.get("shared_recs_df")

    recs = None
    if shared_key == recs_key and shared_df is not None:
        recs = shared_df
    else:
        # 2) otherwise use Debug local cache
        if st.session_state.get("debug_recs_key") == recs_key:
            recs = st.session_state.get("debug_recs")

        if recs is None:
            recs = recommend_for_anime(
                pack=pack,
                query_anime_id=int(anime_id),
                mood=mood,
                alpha=float(alpha),
                top_k=int(top_k),
                cf_candidates=int(cf_candidates),
                cb_candidates=int(cb_candidates),
            )
            st.session_state["debug_recs_key"] = recs_key
            st.session_state["debug_recs"] = recs

    prefer_cols = [
        "anime_id", "name", "type", "genre_overlap",
        "cf_sim", "content_sim", "cf_norm", "content_norm",
        "cf_part", "content_part", "mood_bonus",
        "base_score", "hybrid_score", "like_pct_est", "mood_overlap", "why"
    ]
    cols = [c for c in prefer_cols if c in recs.columns] + [c for c in recs.columns if c not in prefer_cols]
    st.dataframe(recs[cols], use_container_width=True)

    # ===== EXPLAIN SECTION =====
    st.divider()
    st.subheader("Explain: Which features drive the recommendations?")

    show_explain = st.checkbox("Show explanation charts", value=st.session_state["debug_show_explain"])
    st.session_state["debug_show_explain"] = show_explain

    if show_explain:
        max_top = min(30, len(recs)) if recs is not None else 10
        top_n = st.slider(
            "Explain Top-N items",
            5,
            max(5, max_top),
            value=min(st.session_state["debug_explain_top_n"], max_top),
        )
        st.session_state["debug_explain_top_n"] = top_n
        render_explain_charts(recs, top_n=top_n)
