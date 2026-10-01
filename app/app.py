import streamlit as st

from src.model_loader import load_models
from src.recommendation_service import (
    get_content_recommendations,
    get_collaborative_recommendations,
    get_hybrid_recommendations
)


st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide"
)


st.title("🎬 CineMatch")

st.write(
    "Movie recommendations powered by Machine Learning."
)


# Load models
models = load_models()


st.success("Recommendation engine loaded successfully.")


movie_title = st.text_input(
    "Enter a movie",
    value="Toy Story (1995)"
)


user_id = st.number_input(
    "User ID",
    min_value=1,
    value=1
)


if st.button("Get Recommendations"):

    recommendations = get_hybrid_recommendations(
        movie_title,
        user_id,
        models,
        n=10,
        alpha=0.5
    )

    st.subheader("Recommended Movies")

    st.dataframe(
        recommendations,
        use_container_width=True
    )