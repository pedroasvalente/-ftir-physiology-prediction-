import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import render_appearance_sidebar, render_data_sidebar

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")

df = render_data_sidebar()
matrix_colors, model_colors = render_appearance_sidebar(show_models=True)

if df is None or df.empty:
    st.stop()

tab_ml_vs_baseline, tab_grid_vs_bayes = st.tabs(["ML vs Baseline", "Grid vs Bayes"])

with tab_ml_vs_baseline:
    st.subheader("ML Models vs Baseline — R²")
    if "is_baseline" not in df.columns:
        st.info("No baseline column in results.")
    else:
        r2_min = st.slider("Clip R² below", min_value=-10.0, max_value=0.0, value=-2.0, step=0.5,
                           help="Hide extreme outliers to keep the chart readable")
        n_clipped = (df["r2"] < r2_min).sum()
        if n_clipped:
            st.caption(f"{n_clipped} rows with R² < {r2_min} hidden from plot (catastrophic failures, usually n_test < 10).")
        plot_df = df[df["r2"] >= r2_min]
        fig = px.box(
            plot_df,
            x="model",
            y="r2",
            color="is_baseline",
            color_discrete_map={True: "#aaaaaa", False: "#2166ac"},
            labels={"r2": "R²", "model": "Model", "is_baseline": "Baseline"},
            points="all",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5,
                      annotation_text="R²=0 (baseline floor)", annotation_position="bottom right")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

with tab_grid_vs_bayes:
    st.subheader("Grid Search vs Bayesian Optimisation — R²")
    ml_only = df[~df["is_baseline"].astype(bool)] if "is_baseline" in df.columns else df
    if "search" not in ml_only.columns:
        st.info("No search column in results.")
    else:
        fig = px.box(
            ml_only,
            x="search",
            y="r2",
            color="model",
            color_discrete_map=model_colors,
            labels={"r2": "R²", "search": "Search type", "model": "Model"},
            points="all",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
