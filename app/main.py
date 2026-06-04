import streamlit as st

st.set_page_config(
    page_title="FTIR Physiology Prediction",
    page_icon="🧬",
    layout="wide",
)

st.title("FTIR Physiology Prediction")
st.markdown(
    """
    Dashboard for exploring FTIR-ATR spectroscopy + ML results predicting
    physiological variables from biological fluid spectra.

    Use the sidebar to navigate between pages.
    """
)
