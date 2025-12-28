import os
import math
import numpy as np
import pandas as pd
import streamlit as st

ART_DIR = "artifacts"
REQUIRED_FILES = [
    "items.csv", "df_small.csv", "item_emb.npy",
    "U.npy", "V.npy", "V_norm.npy", "user_mean.npy",
    "u_ids.npy", "i_ids.npy"
]
def init_shared_cache():
    st.session_state.setdefault("shared_search_query", "")
    st.session_state.setdefault("shared_selected_anime_id", None)

    # cache recs chung: chỉ lưu 1 key gần nhất cho gọn
    st.session_state.setdefault("shared_recs_key", None)
    st.session_state.setdefault("shared_recs_df", None)

    # lưu params mặc định từ Demo để Debug mở lên khớp
    st.session_state.setdefault("shared_mood", "Normal")
    st.session_state.setdefault("shared_alpha", 0.4)
    st.session_state.setdefault("shared_top_k", 10)
    st.session_state.setdefault("shared_cf_candidates", 200)

def like_probability(pred_rating: float, threshold: float = 8.0, scale: float = 1.0) -> float:
    x = (pred_rating - threshold) / max(scale, 1e-9)
    return 1.0 / (1.0 + math.exp(-x))


def safe_split_genres(s: str) -> set:
    s = "" if s is None else str(s)
    parts = [p.strip().lower() for p in s.split(",")]
    return set([p for p in parts if p])


def predict_cf(pack, user_id: int, anime_id: int) -> float:
    """
    CF prediction using matrix factorization artifacts (U, V, user_mean).
    If user/anime is missing from the mapping, return the global mean.
    """
    df_small = pack["df_small"]
    global_mean = float(df_small["user_rating"].mean())

    u2i, i2i = pack["u2i"], pack["i2i"]
    if user_id not in u2i or anime_id not in i2i:
        return global_mean

    ui = u2i[user_id]
    ii = i2i[anime_id]
    return float(pack["U"][ui] @ pack["V"][ii]) + float(pack["user_mean"][ui])


@st.cache_data(show_spinner=False)
def build_query_users_map(df_small: pd.DataFrame) -> dict:
    """
    Map anime_id -> np.array(user_ids) to avoid regrouping on every recommend.
    """
    mp = {}
    for aid, g in df_small.groupby("anime_id"):
        mp[int(aid)] = g["user_id"].values.astype(int)
    return mp


def check_artifacts():
    if not os.path.isdir(ART_DIR):
        return False, f"Artifacts directory not found: '{ART_DIR}'."
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(ART_DIR, f))]
    if missing:
        return False, "Missing artifacts:\n- " + "\n- ".join(missing)
    return True, ""


@st.cache_resource(show_spinner=False)
def load_artifacts():
    ok, msg = check_artifacts()
    if not ok:
        raise FileNotFoundError(msg)

    items = pd.read_csv(os.path.join(ART_DIR, "items.csv"))
    df_small = pd.read_csv(os.path.join(ART_DIR, "df_small.csv"))

    item_emb = np.load(os.path.join(ART_DIR, "item_emb.npy"))
    U = np.load(os.path.join(ART_DIR, "U.npy"))
    V = np.load(os.path.join(ART_DIR, "V.npy"))
    V_norm = np.load(os.path.join(ART_DIR, "V_norm.npy"))
    user_mean = np.load(os.path.join(ART_DIR, "user_mean.npy"))

    u_ids = np.load(os.path.join(ART_DIR, "u_ids.npy"), allow_pickle=True)
    i_ids = np.load(os.path.join(ART_DIR, "i_ids.npy"), allow_pickle=True)

    id2row = {int(aid): i for i, aid in enumerate(items["anime_id"].values)}
    row2id = items["anime_id"].values.astype(int)

    u2i = {int(u): i for i, u in enumerate(u_ids)}
    i2i = {int(a): i for i, a in enumerate(i_ids)}

    # types
    items["anime_id"] = items["anime_id"].astype(int)
    items["name"] = items["name"].astype(str)
    items["genre"] = items.get("genre", "").fillna("").astype(str)
    items["type"] = items.get("type", "Unknown").fillna("Unknown").astype(str)

    df_small["anime_id"] = df_small["anime_id"].astype(int)
    df_small["user_id"] = df_small["user_id"].astype(int)
    df_small["user_rating"] = df_small["user_rating"].astype(float)

    return {
        "items": items, "df_small": df_small,
        "item_emb": item_emb, "U": U, "V": V, "V_norm": V_norm,
        "user_mean": user_mean, "u_ids": u_ids, "i_ids": i_ids,
        "u2i": u2i, "i2i": i2i, "id2row": id2row, "row2id": row2id
    }


def recommend_for_anime(
    pack,
    query_anime_id: int,
    mood: str = "Normal",
    top_k: int = 10,
    alpha: float = 0.8,
    cf_candidates: int = 200,
    cb_candidates: int = 200,
    like_threshold: float = 8.0,
    like_scale: float = 1.0,
    max_q_users: int = 200,
    seed: int = 42,
):
    """
    Recommend similar anime for a given query anime_id using hybrid (CF + Content)
    plus context-aware bonus based on user's mood.

    Returns:
        pd.DataFrame with columns:
        anime_id, name, type, genre_overlap, content_sim, cf_sim, hybrid_score,
        pred_rating_mean, like_pct_est, why
    """

    # 1) Context rules: mood -> boost genres
    MOOD_RULES = {
        "Normal": [],
        "Happy / Funny": ["comedy", "adventure", "shounen", "slice of life"],
        "Sad / Reflective": ["drama", "tragedy", "romance", "psychological"],
        "Hyped / Intense": ["action", "thriller", "horror", "mecha", "sports"],
        "Relaxed / Chill": ["slice of life", "iyashikei", "fantasy", "music"],
        "Curious / Mind-bending": ["mystery", "psychological", "sci-fi", "police"],
    }

    items = pack["items"]
    df_small = pack["df_small"]
    item_emb = pack["item_emb"]
    id2row = pack["id2row"]
    row2id = pack["row2id"]

    i2i = pack["i2i"]
    i_ids = pack["i_ids"]
    V_norm = pack["V_norm"]

    if query_anime_id not in id2row or query_anime_id not in i2i:
        raise ValueError("Query anime_id not found in artifacts/model.")

    rng = np.random.default_rng(seed)

    # 2) Compute similarities (raw)
    q_row = id2row[query_anime_id]
    q_emb = item_emb[q_row:q_row + 1]
    cb_sim = (item_emb @ q_emb.T).ravel()

    q_i = i2i[query_anime_id]
    q_v = V_norm[q_i:q_i + 1]
    cf_sim = (V_norm @ q_v.T).ravel()

    # 3) Candidate set
    cb_take = min(cb_candidates + 1, len(cb_sim))
    cf_take = min(cf_candidates + 1, len(cf_sim))

    cb_top = np.argpartition(-cb_sim, cb_take - 1)[:cb_take]
    cf_top = np.argpartition(-cf_sim, cf_take - 1)[:cf_take]

    cand = set()

    # candidates from CB
    for idx in cb_top:
        aid = int(row2id[idx])
        if aid in i2i:  # ensure exists in CF index too
            cand.add(aid)

    # candidates from CF
    for idx in cf_top:
        aid = int(i_ids[idx])
        if aid in id2row:  # ensure exists in CB index too
            cand.add(aid)

    cand.discard(query_anime_id)
    cand_list = list(cand)

    if not cand_list:
        return pd.DataFrame()

    # 4) Collect scores + normalize + context
    # Cache: users who rated the query anime (for like_pct_est)
    q_users_map = build_query_users_map(df_small)
    q_users = q_users_map.get(query_anime_id, np.array([], dtype=int))

    if len(q_users) > max_q_users:
        q_users = rng.choice(q_users, max_q_users, replace=False)

    q_genres = safe_split_genres(items.loc[id2row[query_anime_id], "genre"])
    global_mean = float(df_small["user_rating"].mean())

    target_genres = set(MOOD_RULES.get(mood, []))

    # raw scores arrays
    cs_list = []
    cfs_list = []

    for aid in cand_list:
        cs = float(cb_sim[id2row[aid]]) if aid in id2row else 0.0
        cfs = float(cf_sim[i2i[aid]]) if aid in i2i else 0.0
        cs_list.append(cs)
        cfs_list.append(cfs)

    cs_arr = np.array(cs_list, dtype=float)
    cfs_arr = np.array(cfs_list, dtype=float)

    def minmax_scale(arr: np.ndarray) -> np.ndarray:
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn == 0:
            return np.zeros_like(arr) + 0.5
        return (arr - mn) / (mx - mn)

    cs_norm = minmax_scale(cs_arr)
    cfs_norm = minmax_scale(cfs_arr)

    # 5) Final scoring
    results = []

    for i, aid in enumerate(cand_list):
        # 5.1 base hybrid score (using normalized sims)
        cf_norm_val = float(cfs_norm[i])
        content_norm_val = float(cs_norm[i])
        base_score = alpha * cf_norm_val + (1.0 - alpha) * content_norm_val

        # 5.2 context bonus (mood)
        rec_genres = safe_split_genres(items.loc[id2row[aid], "genre"])
        overlap_mood = set()
        context_bonus = 0.0

        if mood != "Normal":
            overlap_mood = rec_genres.intersection(target_genres)
            if overlap_mood:
                context_bonus = 0.15  # giữ nguyên logic của bạn

        final_score = float(base_score + context_bonus)

        # score contribution parts (for explain charts)
        cf_part = float(alpha * cf_norm_val)
        content_part = float((1.0 - alpha) * content_norm_val)
        mood_bonus = float(context_bonus)

        # 5.3 predict rating mean -> like pct estimate
        if len(q_users) > 0:
            preds = [predict_cf(pack, int(u), int(aid)) for u in q_users]
            pred_mean = float(np.mean(preds))
        else:
            pred_mean = global_mean

        like_pct = like_probability(
            pred_mean, threshold=like_threshold, scale=like_scale
        ) * 100.0

        # Overlap with query anime genres (for explanation)
        overlap_q = sorted(list(rec_genres.intersection(q_genres)))[:5]

        why_text = f"Hybrid={base_score:.2f}"
        if context_bonus > 0:
            why_text += f" + Mood boost ({mood})"
        why_text += f"; Genres: {', '.join(overlap_q)}" if overlap_q else "; Genres: None"

        results.append({
            "anime_id": int(aid),
            "name": items.loc[id2row[aid], "name"],
            "type": items.loc[id2row[aid], "type"],
            "genre_overlap": ", ".join(overlap_q) if overlap_q else "None",

            # raw sims (to observe)
            "content_sim": float(cs_arr[i]),
            "cf_sim": float(cfs_arr[i]),

            # normalized sims (USED in scoring)
            "content_norm": float(content_norm_val),
            "cf_norm": float(cf_norm_val),

            # score breakdown (feature influence)
            "cf_part": float(cf_part),
            "content_part": float(content_part),
            "mood_bonus": float(mood_bonus),
            "base_score": float(base_score),
            "hybrid_score": float(final_score),   # giữ tên cũ để UI khỏi sửa nhiều
            "final_score": float(final_score),    # thêm alias cho rõ nghĩa

            "pred_rating_mean": float(pred_mean),
            "like_pct_est": float(like_pct),
            "mood_overlap": ", ".join(sorted(list(overlap_mood))[:6]) if overlap_mood else "None",
            "why": why_text,
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)
    return out
