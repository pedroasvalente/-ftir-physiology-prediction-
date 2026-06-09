import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import RESULTS_DIR, render_appearance_sidebar, render_data_sidebar
from ftir_pred.analysis.spectral_interpretation import top_wavenumbers
from ftir_pred.analysis.wavenumber_reference import BAND_ASSIGNMENTS

st.set_page_config(page_title="PLS-R Direct Regression", layout="wide")
st.title("PLS-R — Direct Chemometric Regression")

matrix_colors, _ = render_appearance_sidebar(show_matrices=True)
render_data_sidebar()

PLSR_DIR = RESULTS_DIR / "plsr_results"
json_path = PLSR_DIR / "plsr_results.json"
csv_path  = PLSR_DIR / "plsr_summary.csv"

if not json_path.exists():
    st.info("No PLS-R results found. Run: `python scripts/run_plsr.py`")
    st.stop()

with open(json_path) as f:
    plsr_data = json.load(f)

summary_df = pd.read_csv(csv_path) if csv_path.exists() else None

tab_heatmap, tab_detail = st.tabs(["R² Heatmap", "Per-target detail"])

# ── Heatmap ───────────────────────────────────────────────────────────────────
with tab_heatmap:
    if summary_df is not None and not summary_df.empty:
        st.subheader("PLS-R R² — target × matrix")

        r2_clip = st.slider("Clip R² below", min_value=-5.0, max_value=0.0,
                            value=-1.0, step=0.5)
        disp = summary_df[summary_df["r2"] >= r2_clip].copy()
        if len(disp) < len(summary_df):
            st.caption(f"{len(summary_df) - len(disp)} rows hidden (R² < {r2_clip})")

        pivot = disp.pivot_table(index="target", columns="sample_type",
                                 values="r2", aggfunc="max")
        if "target_group" in disp.columns:
            tg = disp.drop_duplicates("target").set_index("target")["target_group"]
            pivot = pivot.loc[sorted(pivot.index, key=lambda t: (tg.get(t, "z"), t))]

        fig = px.imshow(
            pivot, color_continuous_scale="RdYlGn",
            zmin=0, zmax=0.5, text_auto=".3f", aspect="auto",
            labels={"color": "R²"},
        )
        fig.update_layout(height=max(500, 18 * len(pivot)),
                          margin=dict(l=10, r=10, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Cross-validated n_components selection (GroupKFold, person-aware).")

        st.subheader("Top results")
        top = disp.sort_values("r2", ascending=False).head(20)
        fmt = {c: "{:.3f}" for c in ["r2", "rmse"] if c in top.columns}
        st.dataframe(
            top.style.format(fmt)
            .background_gradient(subset=["r2"], cmap="RdYlGn", vmin=0, vmax=0.5),
            use_container_width=True, hide_index=True,
        )

# ── Per-target detail ─────────────────────────────────────────────────────────
with tab_detail:
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Target", sorted(plsr_data.keys()))
    with col2:
        matrices_avail = sorted(plsr_data.get(target, {}).keys())
        if not matrices_avail:
            st.info("No PLS-R results for this target.")
            st.stop()
        matrix = st.selectbox("Matrix", matrices_avail)

    entry   = plsr_data[target][matrix]
    color   = matrix_colors.get(matrix, "#2166ac")
    r2      = entry["r2"]
    rmse    = entry["rmse"]
    n_comp  = entry["n_components"]
    n_train = entry.get("n_train", "?")
    n_test  = entry.get("n_test", "?")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (test)", f"{r2:.3f}")
    c2.metric("RMSE (test)", f"{rmse:.3f}")
    c3.metric("Optimal components", str(n_comp))
    c4.metric("n_train / n_test", f"{n_train} / {n_test}")

    col_cv, col_pv = st.columns(2)

    with col_cv:
        cv_curve = entry.get("cv_curve")
        if cv_curve:
            cv_df = pd.DataFrame(cv_curve)
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Scatter(
                x=cv_df["n_components"], y=cv_df["r2_cv"],
                mode="lines+markers",
                error_y=dict(type="data", array=cv_df["r2_std"].tolist(), visible=True),
                line=dict(color=color, width=2),
                name="CV R²",
            ))
            fig_cv.add_vline(x=n_comp, line_dash="dash", line_color="red", opacity=0.7,
                             annotation_text=f"n={n_comp}", annotation_position="top right")
            fig_cv.update_layout(
                xaxis_title="PLS components",
                yaxis_title="CV R²",
                plot_bgcolor="white", paper_bgcolor="white",
                height=300, margin=dict(l=60, r=20, t=20, b=50),
                xaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
                yaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
            )
            st.plotly_chart(fig_cv, use_container_width=True)

    with col_pv:
        if "y_test" in entry and "y_pred" in entry:
            yt = np.array(entry["y_test"])
            yp = np.array(entry["y_pred"])
            lo = float(min(yt.min(), yp.min()))
            hi = float(max(yt.max(), yp.max()))
            fig_pv = go.Figure()
            fig_pv.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi],
                mode="lines", line=dict(color="gray", dash="dash", width=1.5),
                name="Identity", hoverinfo="skip",
            ))
            fig_pv.add_trace(go.Scatter(
                x=yt.tolist(), y=yp.tolist(),
                mode="markers",
                marker=dict(color=color, size=7, opacity=0.75,
                            line=dict(color="white", width=0.5)),
                hovertemplate="Observed: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>",
            ))
            fig_pv.update_layout(
                xaxis_title=f"Observed {target}",
                yaxis_title="Predicted",
                plot_bgcolor="white", paper_bgcolor="white",
                height=300, margin=dict(l=60, r=20, t=20, b=50),
                xaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
                yaxis=dict(showgrid=True, gridcolor="#e5e5e5"),
            )
            st.plotly_chart(fig_pv, use_container_width=True)

    # ── VIP spectrum ──────────────────────────────────────────────────────────
    if "vip_scores" in entry and "wavenumbers" in entry:
        wn  = np.array(entry["wavenumbers"])
        vip = np.array(entry["vip_scores"])

        st.subheader("VIP scores")

        BAND_COLORS = [
            "rgba(255,165,0,0.10)", "rgba(65,105,225,0.10)", "rgba(60,179,113,0.10)",
            "rgba(220,20,60,0.08)", "rgba(186,85,211,0.08)", "rgba(128,128,0,0.10)",
            "rgba(0,128,128,0.10)", "rgba(255,140,0,0.08)", "rgba(180,180,180,0.08)",
            "rgba(65,105,225,0.08)", "rgba(60,179,113,0.08)",
        ]
        fig_vip = go.Figure()
        for i, (lo, hi, band, _) in enumerate(BAND_ASSIGNMENTS):
            fig_vip.add_vrect(
                x0=lo, x1=hi,
                fillcolor=BAND_COLORS[i % len(BAND_COLORS)],
                line_width=0,
                annotation_text=band.split("(")[0].strip(),
                annotation_position="top left",
                annotation_font_size=7,
                annotation_font_color="#888",
            )
        fig_vip.add_hline(y=1.0, line_dash="dot", line_color="red", opacity=0.6,
                          annotation_text="VIP=1", annotation_position="right",
                          annotation_font_size=10)
        fig_vip.add_trace(go.Scatter(
            x=wn.tolist(), y=vip.tolist(),
            fill="tozeroy", mode="lines",
            line=dict(width=1.5, color=color),
            hovertemplate="%{x:.1f} cm⁻¹ — VIP: %{y:.3f}<extra></extra>",
        ))
        fig_vip.update_layout(
            xaxis=dict(title="Wavenumber (cm⁻¹)", autorange="reversed",
                       showgrid=True, gridcolor="#e5e5e5"),
            yaxis=dict(title="VIP score", showgrid=True, gridcolor="#e5e5e5"),
            plot_bgcolor="white", paper_bgcolor="white",
            height=320, margin=dict(l=60, r=30, t=30, b=50),
        )
        st.plotly_chart(fig_vip, use_container_width=True)

        col_top, col_reg = st.columns(2)
        with col_top:
            st.subheader("Top 20 wavenumbers")
            top_df = top_wavenumbers(wn, vip, top_n=20)
            st.dataframe(
                top_df.style.format({"wavenumber": "{:.1f}", "importance": "{:.4f}"}, na_rep="—"),
                use_container_width=True, hide_index=True,
            )
        with col_reg:
            from ftir_pred.analysis.spectral_interpretation import region_summary
            st.subheader("Importance by region")
            reg_df = region_summary(wn, vip)
            st.dataframe(
                reg_df.style.format({"importance": "{:.4f}", "importance_pct": "{:.1f}"}),
                use_container_width=True, hide_index=True,
            )

        st.caption("Movasaghi et al. (2008) Applied Spectroscopy Reviews, 43(2), 134–179.")
