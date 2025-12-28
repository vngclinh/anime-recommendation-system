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

    st.session_state.setdefault("shared_recs_key", None)
    st.session_state.setdefault("shared_recs_df", None)

    st.session_state.setdefault("shared_mood", "Normal")
    st.session_state.setdefault("shared_alpha", 0.4)
    st.session_state.setdefault("shared_top_k", 10)
    st.session_state.setdefault("shared_cf_candidates", 200)
    st.session_state.setdefault("shared_cb_candidates", 200)  # <<< THÊM


def like_probability(pred_rating: float, threshold: float = 8.0, scale: float = 1.0) -> float:
    x = (pred_rating - threshold) / max(scale, 1e-9)
    return 1.0 / (1.0 + math.exp(-x))


def safe_split_genres(s: str) -> set:
    s = "" if s is None else str(s)
    parts = [p.strip().lower() for p in s.split(",")]
    return set([p for p in parts if p])


def check_artifacts():
    if not os.path.isdir(ART_DIR):
        return False, f"Artifacts directory not found: '{ART_DIR}'."
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(ART_DIR, f))]
    if missing:
        return False, "Missing artifacts:\n- " + "\n- ".join(missing)
    return True, ""


def _build_query_users_map(df_small: pd.DataFrame) -> dict[int, np.ndarray]:
    """
    anime_id -> np.array(user_ids)
    IMPORTANT: KHÔNG @st.cache_data theo df_small nữa (hash df lớn rất chậm trên cloud).
    Ta build 1 lần trong load_artifacts() (đã cache_resource).
    """
    mp: dict[int, np.ndarray] = {}
    # groupby 1 lần (chạy lúc load artifacts)
    for aid, g in df_small.groupby("anime_id", sort=False):
        mp[int(aid)] = g["user_id"].values.astype(np.int32, copy=False)
    return mp


@st.cache_resource(show_spinner=False)
def load_artifacts():
    ok, msg = check_artifacts()
    if not ok:
        raise FileNotFoundError(msg)

    items = pd.read_csv(os.path.join(ART_DIR, "items.csv"))
    df_small = pd.read_csv(os.path.join(ART_DIR, "df_small.csv"))

    # numpy arrays
    item_emb = np.load(os.path.join(ART_DIR, "item_emb.npy"))
    U = np.load(os.path.join(ART_DIR, "U.npy"))
    V = np.load(os.path.join(ART_DIR, "V.npy"))
    V_norm = np.load(os.path.join(ART_DIR, "V_norm.npy"))
    user_mean = np.load(os.path.join(ART_DIR, "user_mean.npy"))

    u_ids = np.load(os.path.join(ART_DIR, "u_ids.npy"), allow_pickle=True)
    i_ids = np.load(os.path.join(ART_DIR, "i_ids.npy"), allow_pickle=True)

    # normalize dtypes (giảm RAM + tăng tốc)
    if item_emb.dtype != np.float32:
        item_emb = item_emb.astype(np.float32)
    if U.dtype != np.float32:
        U = U.astype(np.float32)
    if V.dtype != np.float32:
        V = V.astype(np.float32)
    if V_norm.dtype != np.float32:
        V_norm = V_norm.astype(np.float32)
    if user_mean.dtype != np.float32:
        user_mean = user_mean.astype(np.float32)
    norms = np.linalg.norm(item_emb, axis=1, keepdims=True)
    item_emb_norm = item_emb / np.maximum(norms, 1e-12)

    # types
    items["anime_id"] = items["anime_id"].astype(int)
    items["name"] = items["name"].astype(str)
    items["genre"] = items.get("genre", "").fillna("").astype(str)
    items["type"] = items.get("type", "Unknown").fillna("Unknown").astype(str)

    df_small["anime_id"] = df_small["anime_id"].astype(int)
    df_small["user_id"] = df_small["user_id"].astype(int)
    df_small["user_rating"] = df_small["user_rating"].astype(float)

    id2row = {int(aid): i for i, aid in enumerate(items["anime_id"].values)}
    row2id = items["anime_id"].values.astype(int)

    u2i = {int(u): i for i, u in enumerate(u_ids)}
    i2i = {int(a): i for i, a in enumerate(i_ids)}

    # precompute global mean + query users map + genre sets
    global_mean = float(df_small["user_rating"].mean())
    q_users_map = _build_query_users_map(df_small)

    # genre sets precompute (giảm gọi safe_split_genres liên tục)
    genre_sets = [safe_split_genres(g) for g in items["genre"].tolist()]

    return {
        "items": items,
        "df_small": df_small,
        "item_emb": item_emb,
        "U": U,
        "V": V,
        "V_norm": V_norm,
        "user_mean": user_mean,
        "u_ids": u_ids,
        "i_ids": i_ids,
        "u2i": u2i,
        "i2i": i2i,
        "id2row": id2row,
        "row2id": row2id,
        "global_mean": global_mean,
        "q_users_map": q_users_map,
        "genre_sets": genre_sets,
        "item_emb_norm": item_emb_norm
    }


def _minmax_scale(arr: np.ndarray) -> np.ndarray:
    mn = float(arr.min())
    mx = float(arr.max())
    if mx - mn == 0:
        return np.zeros_like(arr, dtype=np.float32) + 0.5
    return ((arr - mn) / (mx - mn)).astype(np.float32, copy=False)


def _predict_mean_for_candidates(pack, q_users: np.ndarray, cand_ids: np.ndarray) -> np.ndarray:
    """
    Vectorized:
      pred(u, item) = U[u] @ V[item] + user_mean[u]
    Return mean over users for each candidate item.
    """
    if q_users.size == 0 or cand_ids.size == 0:
        return np.array([], dtype=np.float32)

    u2i = pack["u2i"]
    i2i = pack["i2i"]
    U = pack["U"]
    V = pack["V"]
    user_mean = pack["user_mean"]

    # map ids -> indices (vector)
    ui = np.fromiter((u2i[int(u)] for u in q_users), dtype=np.int32, count=q_users.size)
    ii = np.fromiter((i2i[int(a)] for a in cand_ids), dtype=np.int32, count=cand_ids.size)

    U_sub = U[ui]                              # (n_users, k)
    V_sub = V[ii]                              # (n_items, k)
    b = user_mean[ui].reshape(-1, 1)            # (n_users, 1)

    # (n_users, k) @ (k, n_items) => (n_users, n_items)
    pred = U_sub @ V_sub.T
    pred = pred + b

    return pred.mean(axis=0).astype(np.float32, copy=False)


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
    MOOD_RULES = {
        "Normal": [],
        "Happy / Funny": ["comedy", "adventure", "shounen", "slice of life"],
        "Sad / Reflective": ["drama", "tragedy", "romance", "psychological"],
        "Hyped / Intense": ["action", "thriller", "horror", "mecha", "sports"],
        "Relaxed / Chill": ["slice of life", "iyashikei", "fantasy", "music"],
        "Curious / Mind-bending": ["mystery", "psychological", "sci-fi", "police"],
    }

    items = pack["items"]
    item_emb = pack["item_emb"]
    id2row = pack["id2row"]
    row2id = pack["row2id"]

    i2i = pack["i2i"]
    i_ids = pack["i_ids"]
    V_norm = pack["V_norm"]

    if query_anime_id not in id2row or query_anime_id not in i2i:
        raise ValueError("Query anime_id not found in artifacts/model.")

    rng = np.random.default_rng(seed)

    # 1) Similarities
    q_row = id2row[query_anime_id]
    q_emb = item_emb[q_row:q_row + 1]              # (1, d)
    cb_sim = (item_emb @ q_emb.T).ravel()          # (N,)

    q_i = i2i[query_anime_id]
    q_v = V_norm[q_i:q_i + 1]                      # (1, k)
    cf_sim = (V_norm @ q_v.T).ravel()              # (M,)

    # 2) Candidate set
    cb_take = min(cb_candidates + 1, cb_sim.shape[0])
    cf_take = min(cf_candidates + 1, cf_sim.shape[0])

    cb_top = np.argpartition(-cb_sim, cb_take - 1)[:cb_take]
    cf_top = np.argpartition(-cf_sim, cf_take - 1)[:cf_take]

    cand = set()

    for idx in cb_top:
        aid = int(row2id[idx])
        if aid in i2i:
            cand.add(aid)

    for idx in cf_top:
        aid = int(i_ids[idx])
        if aid in id2row:
            cand.add(aid)

    cand.discard(query_anime_id)
    if not cand:
        return pd.DataFrame()

    cand_ids = np.array(list(cand), dtype=np.int32)

    # 3) Collect sim values vectorized
    cand_rows = np.fromiter((id2row[int(a)] for a in cand_ids), dtype=np.int32, count=cand_ids.size)
    cand_iis = np.fromiter((i2i[int(a)] for a in cand_ids), dtype=np.int32, count=cand_ids.size)

    cs_arr = cb_sim[cand_rows].astype(np.float32, copy=False)
    cfs_arr = cf_sim[cand_iis].astype(np.float32, copy=False)

    cs_norm = _minmax_scale(cs_arr)
    cfs_norm = _minmax_scale(cfs_arr)

    base_score = alpha * cfs_norm + (1.0 - alpha) * cs_norm

    # 4) Mood bonus (loop nhẹ vì set intersection)
    genre_sets = pack["genre_sets"]
    q_genres = genre_sets[id2row[query_anime_id]]
    target_genres = set(MOOD_RULES.get(mood, []))

    mood_bonus_arr = np.zeros_like(base_score, dtype=np.float32)
    mood_overlap_text = []

    if mood != "Normal" and target_genres:
        for a in cand_ids:
            gs = genre_sets[id2row[int(a)]]
            overlap = gs.intersection(target_genres)
            if overlap:
                mood_bonus_arr[np.where(cand_ids == a)[0][0]] = 0.15
                mood_overlap_text.append(", ".join(sorted(list(overlap))[:6]))
            else:
                mood_overlap_text.append("None")
    else:
        mood_overlap_text = ["None"] * cand_ids.size

    final_score = base_score + mood_bonus_arr

    # 5) Like % estimate (vectorized)
    q_users_map = pack["q_users_map"]
    q_users = q_users_map.get(query_anime_id, np.array([], dtype=np.int32))

    if q_users.size > max_q_users:
        q_users = rng.choice(q_users, max_q_users, replace=False)

    global_mean = float(pack["global_mean"])
    if q_users.size > 0:
        pred_mean_arr = _predict_mean_for_candidates(pack, q_users.astype(np.int32, copy=False), cand_ids)
        if pred_mean_arr.size == 0:
            pred_mean_arr = np.full((cand_ids.size,), global_mean, dtype=np.float32)
    else:
        pred_mean_arr = np.full((cand_ids.size,), global_mean, dtype=np.float32)

    like_pct_arr = (1.0 / (1.0 + np.exp(-((pred_mean_arr - like_threshold) / max(like_scale, 1e-9))))) * 100.0

    # 6) Build output rows
    results = []
    for i, aid in enumerate(cand_ids):
        aid = int(aid)
        rec_genres = genre_sets[id2row[aid]]

        overlap_q = sorted(list(rec_genres.intersection(q_genres)))[:5]
        why_text = f"Hybrid={float(base_score[i]):.2f}"
        if float(mood_bonus_arr[i]) > 0:
            why_text += f" + Mood boost ({mood})"
        why_text += f"; Genres: {', '.join(overlap_q)}" if overlap_q else "; Genres: None"

        results.append({
            "anime_id": aid,
            "name": items.loc[id2row[aid], "name"],
            "type": items.loc[id2row[aid], "type"],
            "genre_overlap": ", ".join(overlap_q) if overlap_q else "None",

            "content_sim": float(cs_arr[i]),
            "cf_sim": float(cfs_arr[i]),

            "content_norm": float(cs_norm[i]),
            "cf_norm": float(cfs_norm[i]),

            "cf_part": float(alpha * cfs_norm[i]),
            "content_part": float((1.0 - alpha) * cs_norm[i]),
            "mood_bonus": float(mood_bonus_arr[i]),
            "base_score": float(base_score[i]),
            "hybrid_score": float(final_score[i]),
            "final_score": float(final_score[i]),

            "pred_rating_mean": float(pred_mean_arr[i]),
            "like_pct_est": float(like_pct_arr[i]),
            "mood_overlap": mood_overlap_text[i],
            "why": why_text,
        })

    out = pd.DataFrame(results)
    if out.empty:
        return out

    out = out.sort_values("hybrid_score", ascending=False).head(top_k).reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False, ttl=3600)
def recommend_for_anime_cached(
    query_anime_id: int,
    mood: str,
    top_k: int,
    alpha: float,
    cf_candidates: int,
    cb_candidates: int,
    like_threshold: float = 8.0,
    like_scale: float = 1.0,
    max_q_users: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Cache theo input nhỏ gọn (KHÔNG đưa pack/df vào key).
    """
    pack = load_artifacts()
    return recommend_for_anime(
        pack=pack,
        query_anime_id=int(query_anime_id),
        mood=str(mood),
        top_k=int(top_k),
        alpha=float(alpha),
        cf_candidates=int(cf_candidates),
        cb_candidates=int(cb_candidates),
        like_threshold=float(like_threshold),
        like_scale=float(like_scale),
        max_q_users=int(max_q_users),
        seed=int(seed),
    )

def apply_anti_cluster_penalty(
    pack,
    recs: pd.DataFrame,
    disliked_ids: list[int],
    beta: float = 0.15,
    sim_clip_min: float = 0.0,   # chỉ phạt similarity dương
) -> pd.DataFrame:
    if recs is None or recs.empty or not disliked_ids:
        return recs

    id2row = pack["id2row"]
    item_emb_norm = pack["item_emb_norm"]

    # lọc disliked có trong embedding
    dis_rows = [id2row[d] for d in disliked_ids if d in id2row]
    if not dis_rows:
        recs = recs.copy()
        recs["dislike_max_sim"] = 0.0
        recs["penalty"] = 0.0
        recs["final_score"] = recs["hybrid_score"].astype(float)
        return recs

    # candidate rows
    cand_ids = recs["anime_id"].astype(int).values
    cand_rows = [id2row[a] for a in cand_ids if a in id2row]

    # nếu có candidate không có row thì vẫn giữ, sim=0
    recs = recs.copy()
    max_sim_all = np.zeros(len(recs), dtype=np.float32)

    cand_vecs = item_emb_norm[np.array(cand_rows, dtype=np.int32)]  # (m, dim)
    dis_vecs  = item_emb_norm[np.array(dis_rows, dtype=np.int32)]   # (d, dim)

    # cosine sim = dot vì đã normalize
    sim = cand_vecs @ dis_vecs.T  # (m, d)
    sim = np.clip(sim, sim_clip_min, 1.0)
    max_sim = sim.max(axis=1).astype(np.float32)  # (m,)

    # gán về đúng thứ tự recs
    ptr = 0
    for i, aid in enumerate(cand_ids):
        if aid in id2row:
            max_sim_all[i] = float(max_sim[ptr])
            ptr += 1

    penalty = beta * max_sim_all
    recs["dislike_max_sim"] = max_sim_all
    recs["penalty"] = penalty
    recs["final_score"] = recs["hybrid_score"].astype(float) - penalty
    recs["final_score"] = recs["final_score"].clip(lower=0.0)

    # sort theo final_score mới
    recs = recs.sort_values("final_score", ascending=False).reset_index(drop=True)
    return recs
