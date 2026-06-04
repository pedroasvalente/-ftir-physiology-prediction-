"""
Shared appearance settings and data loading for all Streamlit pages.
"""
import os
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DEFAULT_GROUP_COLORS = {
    "cardiorespiratory": "#e41a1c",
    "body_composition": "#377eb8",
    "cbc": "#4daf4a",
    "hormonal": "#984ea3",
    "other": "#999999",
}

DEFAULT_MATRIX_COLORS = {
    "CAPILAR": "#1b9e77",
    "PLASMA": "#d95f02",
    "SALIVA": "#7570b3",
    "SERUM": "#e7298a",
    "URINE": "#66a61e",
}

DEFAULT_MODEL_COLORS = {
    "Random Forest": "#2166ac",
    "MLP Regressor": "#d6604d",
    "Decision Tree": "#4dac26",
    "XGBoost": "#8073ac",
    "Dummy (mean)": "#aaaaaa",
    "Ridge": "#f4a442",
    "PLS (3 comp)": "#a6cee3",
}


def _init_defaults():
    if "matrix_colors" not in st.session_state:
        st.session_state.matrix_colors = dict(DEFAULT_MATRIX_COLORS)
    if "model_colors" not in st.session_state:
        st.session_state.model_colors = dict(DEFAULT_MODEL_COLORS)
    if "group_colors" not in st.session_state:
        st.session_state.group_colors = dict(DEFAULT_GROUP_COLORS)


def render_appearance_sidebar(show_matrices=True, show_models=False):
    _init_defaults()
    with st.sidebar.expander("🎨 Appearance", expanded=False):
        if show_matrices:
            st.markdown("**Matrix colours**")
            for key in list(st.session_state.matrix_colors.keys()):
                st.session_state.matrix_colors[key] = st.color_picker(
                    key, value=st.session_state.matrix_colors[key], key=f"matc_{key}"
                )
        if show_models:
            st.markdown("**Model colours**")
            for key in list(st.session_state.model_colors.keys()):
                st.session_state.model_colors[key] = st.color_picker(
                    key, value=st.session_state.model_colors[key], key=f"mc_{key}"
                )
    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <small>
        © 2025 Pedro Afonso Valente<br>
        University of Coimbra<br>
        Licensed under CC BY-NC-ND 4.0
        </small>
        """,
        unsafe_allow_html=True,
    )
    return st.session_state.matrix_colors, st.session_state.model_colors


@st.cache_data(ttl=300)
def load_results(results_dir: str = str(RESULTS_DIR)) -> pd.DataFrame:
    base = Path(results_dir)
    csv_files = sorted(base.rglob("results_summary.csv"))
    if not csv_files:
        return pd.DataFrame()
    frames = []
    for f in csv_files:
        try:
            sub = pd.read_csv(f)
            sub["run"] = f.parent.name
            frames.append(sub)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    for col in ["r2", "rmse", "mae", "mape", "pearson_r"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def render_data_sidebar() -> pd.DataFrame | None:
    with st.sidebar:
        st.header("Data")
        df = load_results()
        if df.empty:
            st.error("No results found. Run a training experiment first.")
            return None

        runs = sorted(df["run"].unique()) if "run" in df.columns else []
        if runs:
            selected = st.selectbox("Run", ["All"] + runs)
            if selected != "All":
                df = df[df["run"] == selected]

        st.caption(f"{len(df)} rows · {df['target'].nunique() if 'target' in df.columns else '?'} targets")
    return df
