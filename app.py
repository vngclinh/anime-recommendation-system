import streamlit as st

st.set_page_config(
    page_title="Anime Recommender",
    layout="wide"
)

st.sidebar.title("Anime Recommender")
st.sidebar.text_input("Username", key="username", placeholder="Enter a username")

if not st.session_state.get("username", "").strip():
    st.title("Anime Recommender")
    st.info("Enter a username in the sidebar to continue.")
    st.stop()

page = st.sidebar.radio(
    "Navigate",
    ["Demo", "Debug", "Data Visualization"]
)

if page == "Demo":
    from pages.demo_page import render
    render()

elif page == "Debug":
    from pages.debug_page import render
    render()

elif page == "Data Visualization":
    from pages.visualize_page import render
    render()
