import os

import streamlit as st


PLOT_CARDS = [
    {
        "title": "Top Anime Community",
        "description": "Top anime by member count.",
        "path": "artifacts/plots/top_anime_community.png",
    },
    {
        "title": "Anime Categories Distribution",
        "description": "Share of anime types (TV, OVA, Movie, etc.).",
        "path": "artifacts/plots/anime_categories_distribution.png",
    },
    {
        "title": "Anime Categories Hub",
        "description": "Count of anime by category.",
        "path": "artifacts/plots/anime_categories_hub.png",
    },
    {
        "title": "Ratings Distribution (Overall)",
        "description": "Average anime ratings and user ratings distributions.",
        "path": "artifacts/plots/ratings_distribution_overall.png",
    },
    {
        "title": "Top Animes Based On Ratings",
        "description": "Top anime titles by average rating.",
        "path": "artifacts/plots/top_anime_ratings.png",
    },
    {
        "title": "Ratings Distribution (TV)",
        "description": "TV category: average and user rating distributions.",
        "path": "artifacts/plots/ratings_distribution_tv.png",
    },
    {
        "title": "Ratings Distribution (OVA)",
        "description": "OVA category: average and user rating distributions.",
        "path": "artifacts/plots/ratings_distribution_ova.png",
    },
    {
        "title": "Genre WordCloud",
        "description": "Word cloud of genres.",
        "path": "artifacts/plots/genre_wordcloud.png",
    },
]


def _init_plot_cards():
    if "plot_cards" not in st.session_state:
        st.session_state["plot_cards"] = [dict(card) for card in PLOT_CARDS]


def render():
    st.title("Data Visualization")
    _init_plot_cards()

    for card in st.session_state["plot_cards"]:
        st.subheader(card["title"])
        st.write(card["description"])
        if os.path.exists(card["path"]):
            st.image(card["path"], use_container_width=True)
        else:
            st.warning(f"Missing plot file: {card['path']}")
