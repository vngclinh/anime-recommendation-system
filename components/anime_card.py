import streamlit as st


def anime_card(img_url: str, title: str, genres: list, like_pct: float, anime_id: int):
    action = None

    with st.container():
        c1, c2, c3 = st.columns([1.1, 4.2, 1.6], vertical_alignment="center")

        with c1:
            st.image(img_url, width=150)

        with c2:
            st.markdown(f"## {title}")
            if genres:
                st.markdown(" ".join([f"`{g}`" for g in genres]))
            st.progress(min(int(like_pct), 100))
            st.caption(f"Estimated chance you'll like this: **{like_pct:.1f}%**")

        with c3:
            if st.button(
                "Already Watched & Liked",
                use_container_width=True,
                key=f"watch_{anime_id}",
            ):
                action = "watched"
            if st.button(
                "Not Interested",
                use_container_width=True,
                key=f"no_{anime_id}",
            ):
                action = "disliked"

    return action
