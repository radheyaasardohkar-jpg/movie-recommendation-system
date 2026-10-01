import streamlit as st

from src.model_pipeline import build_models


@st.cache_resource
def load_models():

    models = build_models()

    return models