import os
import re
import streamlit as st

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Anime EDA Visualizations",
    layout="wide",
    page_icon="📊"
)

PLOT_DIR = "artifacts/plots"

# Optional: mapping đẹp hơn cho một số file bạn đã tạo
PLOT_META = {
    "top_anime_community.png": {
        "title": "Top Anime Community",
        "desc": "Most popular anime ranked by member count."
    },
    "anime_categories_distribution.png": {
        "title": "Anime Categories Distribution",
        "desc": "Share of anime types (TV, OVA, Movie, Special, ONA, Music)."
    },
    "anime_categories_hub.png": {
        "title": "Anime Categories Hub",
        "desc": "Count of anime titles by type."
    },
    "ratings_distribution_overall.png": {
        "title": "Ratings Distribution (Overall)",
        "desc": "Distribution of average anime ratings and user ratings."
    },
    "top_anime_ratings.png": {
        "title": "Top Anime by Average Rating",
        "desc": "Highest-rated anime titles based on average rating."
    },
    "ratings_distribution_tv.png": {
        "title": "Ratings Distribution (TV)",
        "desc": "Rating distributions for the TV category (average vs. user ratings)."
    },
    "ratings_distribution_ova.png": {
        "title": "Ratings Distribution (OVA)",
        "desc": "Rating distributions for the OVA category (average vs. user ratings)."
    },
    "anime_name_wordcloud.png": {
        "title": "Anime Title Word Cloud",
        "desc": "Word cloud of the most frequently rated anime titles."
    },
    "type_rating_episode_violin_box.png": {
        "title": "Ratings & Episodes by Type",
        "desc": "Rating distribution (violin) and episode count distribution (boxplot, log scale) by type."
    },
    "type_genre_rating_heatmap.png": {
        "title": "Average Rating Heatmap by Type & Main Genre",
        "desc": "Heatmap of mean ratings across anime type and the primary genre."
    },
}

def filename_to_title(fname: str) -> str:
    base = os.path.splitext(fname)[0]
    base = base.replace("-", " ").replace("_", " ").strip()
    base = re.sub(r"\s+", " ", base)
    # Title case but keep common acronyms nicer
    title = base.title().replace("Tv", "TV").replace("Ova", "OVA").replace("Ona", "ONA")
    return title

def filename_to_desc(fname: str) -> str:
    # English fallback description
    title = filename_to_title(fname)
    return f"Visualization: {title}."

def load_plot_files(plot_dir: str):
    if not os.path.exists(plot_dir):
        return []
    files = [f for f in os.listdir(plot_dir) if f.lower().endswith(".png")]
    files.sort()
    return files

# ===============================
# UI
# ===============================
st.title("📊 Anime EDA Visualizations")
st.caption("This page automatically displays all saved plots from `artifacts/plots/`.")

st.divider()

plot_files = load_plot_files(PLOT_DIR)

if not plot_files:
    st.warning("No plot images were found in `artifacts/plots/`.")
    st.stop()

# ===============================
# 1 IMAGE PER ROW
# ===============================
for fname in plot_files:
    meta = PLOT_META.get(fname, None)
    title = meta["title"] if meta else filename_to_title(fname)
    desc  = meta["desc"]  if meta else filename_to_desc(fname)

    img_path = os.path.join(PLOT_DIR, fname)

    # Card-like section per image
    with st.container(border=True):
        st.subheader(title)
        st.write(desc)
        st.image(img_path, use_container_width=True)

st.divider()
st.caption("Tip: Add a new PNG file into `artifacts/plots/` and it will appear here automatically.")
