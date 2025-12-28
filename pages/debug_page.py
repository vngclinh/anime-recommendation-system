import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from recommender import init_shared_cache, recommend_for_anime, load_artifacts, apply_anti_cluster_penalty
from services.history_store import get_disliked_ids


# -----------------------------
# State init
# -----------------------------
def _init_debug_state():
    init_shared_cache()

    st.session_state.setdefault("debug_query", st.session_state.get("shared_search_query", ""))
    st.session_state.setdefault("debug_selected_anime_id", st.session_state.get("shared_selected_anime_id", None))
    st.session_state.setdefault("debug_mood", st.session_state.get("shared_mood", "Normal"))
    st.session_state.setdefault("debug_alpha", st.session_state.get("shared_alpha", 0.4))
    st.session_state.setdefault("debug_top_k", st.session_state.get("shared_top_k", 10))
    st.session_state.setdefault("debug_cf_candidates", st.session_state.get("shared_cf_candidates", 200))

    # debug caches
    st.session_state.setdefault("debug_recs_key", None)
    st.session_state.setdefault("debug_recs_raw", None)  # raw before penalty

    # explain UI state
    st.session_state.setdefault("debug_show_explain", True)
    st.session_state.setdefault("debug_explain_top_n", 10)

    # penalty UI state
    st.session_state.setdefault("debug_use_penalty", True)
    st.session_state.setdefault("debug_beta", 0.15)
    st.session_state.setdefault("debug_dislike_limit", 50)


# -----------------------------
# Helpers: compute "most similar disliked item" for each candidate
# -----------------------------
def _compute_best_disliked_for_candidates(pack, cand_ids: np.ndarray, disliked_ids: list[int]):
    """
    Returns:
      max_sim_all: (len(cand_ids),) float32
      best_disliked_id: (len(cand_ids),) int32 (or -1 if none)
    """
    id2row = pack["id2row"]
    emb = pack["item_emb_norm"]  # normalized (cosine = dot)

    n = len(cand_ids)
    max_sim_all = np.zeros(n, dtype=np.float32)
    best_disliked_id = np.full(n, -1, dtype=np.int32)

    if n == 0 or not disliked_ids:
        return max_sim_all, best_disliked_id

    # rows for disliked that exist
    dis_ids_valid = [int(d) for d in disliked_ids if int(d) in id2row]
    if not dis_ids_valid:
        return max_sim_all, best_disliked_id

    dis_rows = np.array([id2row[int(d)] for d in dis_ids_valid], dtype=np.int32)
    dis_vecs = emb[dis_rows]  # (D, dim)

    # candidate rows (mark missing)
    cand_rows = np.full(n, -1, dtype=np.int32)
    for i, aid in enumerate(cand_ids):
        aid = int(aid)
        if aid in id2row:
            cand_rows[i] = id2row[aid]

    valid_mask = cand_rows >= 0
    if not np.any(valid_mask):
        return max_sim_all, best_disliked_id

    cand_vecs = emb[cand_rows[valid_mask]]  # (M, dim)

    # cosine sim = dot
    sim = cand_vecs @ dis_vecs.T  # (M, D)
    sim = np.clip(sim, 0.0, 1.0)  # only positive similarity penalized

    argmax = sim.argmax(axis=1)
    maxv = sim[np.arange(sim.shape[0]), argmax].astype(np.float32)

    max_sim_all[valid_mask] = maxv
    best_ids = np.array(dis_ids_valid, dtype=np.int32)[argmax]
    best_disliked_id[valid_mask] = best_ids

    return max_sim_all, best_disliked_id


# -----------------------------
# Explain charts (extended: supports final_score + penalty)
# -----------------------------
def render_explain_charts(recs: pd.DataFrame, top_n: int = 10, score_col: str = "hybrid_score"):
    if recs is None or recs.empty:
        st.info("No recommendations to explain yet.")
        return

    needed = {"name", "cf_part", "content_part", "mood_bonus"}
    if not needed.issubset(set(recs.columns)):
        st.warning(
            "Explain charts cannot render because required columns are missing.\n"
            "Please update recommender.py to return: "
            + ", ".join(sorted(list(needed)))
        )
        return

    score_col = score_col if score_col in recs.columns else "hybrid_score"
    tmp = recs.copy().head(top_n).copy()

    for col in ["cf_part", "content_part", "mood_bonus", score_col]:
        if col in tmp.columns:
            tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    tmp = tmp.dropna(subset=["cf_part", "content_part", "mood_bonus", score_col])
    if tmp.empty:
        st.warning("All rows became NaN after numeric conversion; cannot plot.")
        return

    tmp = tmp.sort_values(score_col, ascending=True)

    # 1) stacked bar contributions (CF + Content + Mood)
    fig_h = plt.figure(figsize=(14, max(4, 0.6 * len(tmp))))
    y = np.arange(len(tmp))

    plt.barh(y, tmp["cf_part"].values, label="CF contribution")
    plt.barh(y, tmp["content_part"].values, left=tmp["cf_part"].values, label="Content contribution")
    left2 = (tmp["cf_part"] + tmp["content_part"]).values
    plt.barh(y, tmp["mood_bonus"].values, left=left2, label="Mood bonus")

    plt.yticks(y, tmp["name"].astype(str).values)
    plt.xlabel("Contribution to score")
    plt.title("Score Breakdown: CF vs Content vs Mood")
    plt.legend()
    st.pyplot(fig_h, clear_figure=True)

    # 2) penalty chart if available
    if "penalty" in tmp.columns and "final_score" in tmp.columns:
        tmp["penalty"] = pd.to_numeric(tmp["penalty"], errors="coerce").fillna(0.0)

        fig_p = plt.figure(figsize=(14, max(4, 0.55 * len(tmp))))
        y2 = np.arange(len(tmp))
        plt.barh(y2, tmp["penalty"].values, label="Penalty (beta * max_sim)")
        plt.yticks(y2, tmp["name"].astype(str).values)
        plt.xlabel("Penalty")
        plt.title("Dislike Penalty Applied to Candidates")
        plt.legend()
        st.pyplot(fig_p, clear_figure=True)

    # 3) scatter optional
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

    # 4) sanity check
    base_sum = tmp["cf_part"] + tmp["content_part"] + tmp["mood_bonus"]
    if "penalty" in tmp.columns and score_col == "final_score":
        tmp["final_check"] = base_sum - pd.to_numeric(tmp["penalty"], errors="coerce").fillna(0.0)
    else:
        tmp["final_check"] = base_sum

    gap = float(np.mean(np.abs(tmp["final_check"] - tmp[score_col])))
    st.caption(f"Sanity check: mean(|(cf_part+content_part+mood_bonus{'-penalty' if score_col=='final_score' else ''}) - {score_col}|) = {gap:.6f}")


# -----------------------------
# Main render
# -----------------------------
def render():
    st.title("Debug Recommender")
    _init_debug_state()

    pack = load_artifacts()
    items = pack["items"]

    # ----- Sidebar: penalty controls -----
    with st.sidebar:
        st.markdown("## Dislike Penalty Debug")

        # default username from main app state if exists
        default_user = st.session_state.get("username", "")
        username = st.text_input("Username (Supabase)", value=default_user).strip()
        st.session_state["debug_username"] = username

        use_penalty = st.checkbox("Apply Anti-Cluster Penalty", value=st.session_state["debug_use_penalty"])
        st.session_state["debug_use_penalty"] = use_penalty

        beta = st.slider("beta (penalty factor)", 0.0, 0.5, float(st.session_state["debug_beta"]), 0.01)
        st.session_state["debug_beta"] = beta

        dislike_limit = st.slider("Max disliked used", 0, 200, int(st.session_state["debug_dislike_limit"]), 10)
        st.session_state["debug_dislike_limit"] = dislike_limit

        st.caption("Formula: final_score = hybrid_score - beta * max_sim(candidate, disliked)")

    # ----- Filter bar -----
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
            alpha = st.slider("alpha", 0.0, 1.0, value=float(st.session_state["debug_alpha"]), step=0.05)
            st.session_state["debug_alpha"] = alpha

        with c3:
            top_k = st.slider("Top-K", 5, 30, value=int(st.session_state["debug_top_k"]))
            st.session_state["debug_top_k"] = top_k

        with c4:
            cf_candidates = st.slider(
                "CF candidates", 50, 1000, value=int(st.session_state["debug_cf_candidates"]), step=50
            )
            st.session_state["debug_cf_candidates"] = cf_candidates

    st.divider()

    # ----- Search -----
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

    # ----- Recommend (RAW) -----
    cb_candidates = 200  # keep same as Demo

    top_k_raw = max(120, int(top_k) * 8)  # IMPORTANT: enough to rerank after penalty
    recs_key = (int(anime_id), str(mood), float(alpha), int(top_k_raw), int(cf_candidates), int(cb_candidates))

    recs_raw = None
    if st.session_state.get("debug_recs_key") == recs_key:
        recs_raw = st.session_state.get("debug_recs_raw")

    if recs_raw is None:
        with st.spinner("Computing raw recommendations..."):
            recs_raw = recommend_for_anime(
                pack=pack,
                query_anime_id=int(anime_id),
                mood=mood,
                alpha=float(alpha),
                top_k=int(top_k_raw),
                cf_candidates=int(cf_candidates),
                cb_candidates=int(cb_candidates),
            )
        st.session_state["debug_recs_key"] = recs_key
        st.session_state["debug_recs_raw"] = recs_raw

    if recs_raw is None or recs_raw.empty:
        st.info("No recommendations found.")
        return

    # ----- Fetch disliked ids (if available) -----
    disliked_ids = []
    if username and dislike_limit > 0:
        disliked_ids = get_disliked_ids(username, limit=int(dislike_limit))

    # ----- Apply penalty (optional) -----
    recs_before = recs_raw.copy()
    recs_after = recs_raw.copy()

    if use_penalty and disliked_ids:
        recs_after = apply_anti_cluster_penalty(
            pack=pack,
            recs=recs_after,
            disliked_ids=disliked_ids,
            beta=float(beta),
        )

        # add "best disliked name" for explanation (vectorized)
        cand_ids = recs_after["anime_id"].astype(int).values
        max_sim, best_disliked = _compute_best_disliked_for_candidates(pack, cand_ids, disliked_ids)

        id2name = dict(zip(items["anime_id"].astype(int), items["name"].astype(str)))
        best_name = [id2name.get(int(x), "Unknown") if int(x) != -1 else "None" for x in best_disliked]

        # overwrite / ensure columns
        recs_after["dislike_max_sim"] = max_sim.astype(np.float32)
        recs_after["best_disliked_id"] = best_disliked.astype(np.int32)
        recs_after["best_disliked_name"] = best_name

        # recompute penalty if missing
        if "penalty" not in recs_after.columns:
            recs_after["penalty"] = float(beta) * recs_after["dislike_max_sim"].astype(float)
        if "final_score" not in recs_after.columns:
            recs_after["final_score"] = recs_after["hybrid_score"].astype(float) - recs_after["penalty"].astype(float)
            recs_after["final_score"] = recs_after["final_score"].clip(lower=0.0)

    # cut to top_k for display
    recs = recs_after.head(int(top_k)).copy()

    # ----- Summary section -----
    st.markdown("### Dislike penalty status")
    if not username:
        st.info("Nhập Username (sidebar) để Debug lấy disliked từ Supabase.")
    elif not disliked_ids:
        st.info("User chưa có disliked (hoặc query DB lỗi) → không áp penalty.")
    else:
        st.write(f"- Disliked used: **{len(disliked_ids)}** (limit={int(dislike_limit)})")
        st.write(f"- beta = **{float(beta):.2f}**")
        if use_penalty:
            st.success("Penalty is ON: final_score is used for ranking.")
        else:
            st.warning("Penalty is OFF: using hybrid_score only.")

    # ----- Main dataframe -----
    prefer_cols = [
        "anime_id", "name", "type", "genre_overlap",
        "cf_sim", "content_sim", "cf_norm", "content_norm",
        "cf_part", "content_part", "mood_bonus",
        "base_score", "hybrid_score",
        "dislike_max_sim", "penalty", "final_score",
        "best_disliked_name",
        "like_pct_est", "mood_overlap", "why"
    ]
    cols = [c for c in prefer_cols if c in recs.columns] + [c for c in recs.columns if c not in prefer_cols]
    st.dataframe(recs[cols], use_container_width=True)

    # ----- Penalty effect visualization -----
    if use_penalty and disliked_ids:
        st.divider()
        st.subheader("Penalty effect (Before vs After)")

        # compute delta on raw list for better insight
        raw_after = recs_after.copy()
        raw_after = raw_after.copy()

        # ensure columns
        if "final_score" in raw_after.columns and "hybrid_score" in raw_after.columns:
            raw_after["delta"] = raw_after["final_score"].astype(float) - raw_after["hybrid_score"].astype(float)
        else:
            raw_after["delta"] = 0.0

        # most penalized
        top_pen = raw_after.sort_values("delta", ascending=True).head(15).copy()

        # chart: hybrid vs final
        fig = plt.figure(figsize=(12, max(4, 0.45 * len(top_pen))))
        y = np.arange(len(top_pen))
        plt.barh(y, top_pen["hybrid_score"].astype(float).values, label="hybrid_score")
        plt.barh(y, top_pen["final_score"].astype(float).values, label="final_score")
        plt.yticks(y, top_pen["name"].astype(str).values)
        plt.xlabel("Score")
        plt.title("Most penalized items (hybrid_score vs final_score)")
        plt.legend()
        st.pyplot(fig, clear_figure=True)

        show_cols = [c for c in ["anime_id", "name", "hybrid_score", "final_score", "penalty", "dislike_max_sim", "best_disliked_name", "delta"] if c in top_pen.columns]
        st.dataframe(top_pen[show_cols], use_container_width=True)

    # ----- Explain charts -----
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
            value=min(int(st.session_state["debug_explain_top_n"]), max_top),
        )
        st.session_state["debug_explain_top_n"] = top_n

        # choose score column for sorting in explain
        score_col = "final_score" if (use_penalty and "final_score" in recs.columns and disliked_ids) else "hybrid_score"
        render_explain_charts(recs, top_n=top_n, score_col=score_col)
