import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import TARGET_GROUP_COLORS, render_appearance_sidebar, render_data_sidebar
from ftir_pred.data.config import SAMPLE_TYPES

st.set_page_config(page_title="ML Results", layout="wide")
st.title("ML Results")

matrix_colors, model_colors = render_appearance_sidebar(show_matrices=True, show_models=True)
df = render_data_sidebar()
if df is None or df.empty:
    st.stop()

METRIC_COLS = [c for c in ["r2", "rmse", "mae", "pearson_r"] if c in df.columns]

with st.sidebar:
    st.header("Filters")
    groups = sorted(df["target_group"].dropna().unique()) if "target_group" in df.columns else []
    sel_groups  = st.multiselect("Target groups", groups, default=groups)
    sel_matrices = st.multiselect("Matrices", SAMPLE_TYPES, default=SAMPLE_TYPES)
    if "timepoints" in df.columns:
        tp_opts  = sorted(df["timepoints"].fillna("all").unique())
        sel_tp   = st.multiselect("Timepoints", tp_opts, default=tp_opts)
    else:
        sel_tp = []
    show_baseline = st.checkbox("Include baselines", value=False)
    metric = st.selectbox("Metric", METRIC_COLS, index=0)

mask = df["sample_type"].isin(sel_matrices)
if sel_groups and "target_group" in df.columns:
    mask &= df["target_group"].isin(sel_groups)
if sel_tp and "timepoints" in df.columns:
    mask &= df["timepoints"].fillna("all").isin(sel_tp)
if not show_baseline and "is_baseline" in df.columns:
    mask &= ~df["is_baseline"].astype(bool)

filtered = df[mask].copy()
if filtered.empty:
    st.info("No results for the current selection.")
    st.stop()

best = (
    filtered.loc[filtered.groupby(["target", "sample_type"])[metric].idxmax()]
    .reset_index(drop=True)
)

tab_heatmap, tab_top, tab_dist, tab_detail = st.tabs(
    ["Heatmap", "Top results", "Distributions", "All results"]
)

# ── Heatmap ───────────────────────────────────────────────────────────────────
with tab_heatmap:
    st.subheader(f"Best {metric.upper()} per target × matrix")
    pivot = best.pivot(index="target", columns="sample_type", values=metric)
    pivot = pivot.reindex(columns=[c for c in SAMPLE_TYPES if c in pivot.columns])

    if "target_group" in best.columns:
        grp_order = best.drop_duplicates("target").set_index("target")["target_group"]
        pivot = pivot.loc[
            sorted(pivot.index, key=lambda t: (grp_order.get(t, "z"), t))
        ]

    cscale = "RdYlGn" if metric == "r2" else "RdYlGn_r"
    vmin   = 0 if metric == "r2" else None
    # Dynamic upper bound: show real spread, not washed out by 0–1 scale
    raw_max = float(best[metric].max()) if not best.empty else 0.5
    import math
    vmax = math.ceil(raw_max * 10) / 10 if metric == "r2" else None

    fig = px.imshow(
        pivot, color_continuous_scale=cscale,
        zmin=vmin, zmax=vmax,
        text_auto=".3f", aspect="auto",
        labels={"color": metric.upper()},
    )
    fig.update_layout(height=max(500, 18 * len(pivot)), margin=dict(l=10, r=10, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)

    note = f"Scale: 0 – {vmax:.1f} (observed max = {raw_max:.3f}). " if metric == "r2" else ""
    st.caption(f"{note}Targets ordered by physiological group. Best run per (target, matrix) combination.")

# ── Top results ───────────────────────────────────────────────────────────────
with tab_top:
    top_n = st.slider("Top N per matrix", 1, 10, 5)
    top_cols = [c for c in ["target", "target_group", "sample_type", "model", "timepoints",
                             "r2", "r2_ci95_low", "r2_ci95_high", "rmse", "pearson_r", "n_test"]
                if c in filtered.columns]
    top_df = (
        filtered.sort_values(metric, ascending=False)
        .groupby("sample_type").head(top_n)
        .reset_index(drop=True)[top_cols]
    )
    top_df.index += 1
    fmt = {c: "{:.3f}" for c in ["r2", "rmse", "mae", "pearson_r",
                                   "r2_ci95_low", "r2_ci95_high"] if c in top_df.columns}
    st.dataframe(
        top_df.style.format(fmt)
        .background_gradient(subset=[metric], cmap="RdYlGn",
                             vmin=0 if metric == "r2" else None,
                             vmax=1 if metric == "r2" else None),
        use_container_width=True,
    )

    csv_bytes = top_df.to_csv(index=True).encode()
    st.download_button("Download top results (CSV)", csv_bytes,
                       file_name="top_results.csv", mime="text/csv")

# ── Distributions ─────────────────────────────────────────────────────────────
with tab_dist:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader(f"{metric.upper()} by matrix — all models")
        fig_box = px.box(
            filtered, x="sample_type", y=metric,
            color="model", color_discrete_map=model_colors,
            points="all",
            hover_data=[c for c in ["target", "model", "timepoints"] if c in filtered.columns],
            labels={"sample_type": "Matrix", metric: metric.upper()},
            category_orders={"sample_type": SAMPLE_TYPES},
        )
        if metric == "r2":
            fig_box.add_hline(y=0.3, line_dash="dot", line_color="gray", opacity=0.6,
                              annotation_text="R²=0.30")
        fig_box.update_layout(height=420, plot_bgcolor="white", paper_bgcolor="white",
                              yaxis=dict(showgrid=True, gridcolor="#e5e5e5"))
        st.plotly_chart(fig_box, use_container_width=True)

    with col_b:
        st.subheader(f"{metric.upper()} by target group")
        if "target_group" in filtered.columns:
            fig_grp = px.box(
                filtered, x="target_group", y=metric,
                color="target_group", color_discrete_map=TARGET_GROUP_COLORS,
                points="all",
                hover_data=[c for c in ["target", "sample_type", "model"] if c in filtered.columns],
                labels={"target_group": "Physiological group", metric: metric.upper()},
            )
            if metric == "r2":
                fig_grp.add_hline(y=0.3, line_dash="dot", line_color="gray", opacity=0.6)
            fig_grp.update_layout(height=420, showlegend=False,
                                  plot_bgcolor="white", paper_bgcolor="white",
                                  yaxis=dict(showgrid=True, gridcolor="#e5e5e5"))
            st.plotly_chart(fig_grp, use_container_width=True)

    st.subheader("Cross-matrix summary — best model per matrix")
    rows = []
    for mat in SAMPLE_TYPES:
        sub = filtered[filtered["sample_type"] == mat]
        if sub.empty:
            continue
        row = sub.loc[sub[metric].idxmax()]
        rows.append(row)
    if rows:
        summary = pd.DataFrame(rows)
        show_cols = [c for c in ["sample_type", "target", "target_group", "model",
                                  "r2", "rmse", "pearson_r", "n_test"] if c in summary.columns]
        fmt2 = {c: "{:.3f}" for c in ["r2", "rmse", "pearson_r"] if c in summary.columns}
        st.dataframe(
            summary[show_cols].reset_index(drop=True).style
            .format(fmt2)
            .background_gradient(subset=["r2"] if "r2" in summary.columns else [],
                                 cmap="RdYlGn", vmin=0, vmax=0.5),
            use_container_width=True, hide_index=True,
        )

# ── All results ───────────────────────────────────────────────────────────────
with tab_detail:
    st.dataframe(
        filtered.sort_values(metric, ascending=False).reset_index(drop=True)
        .style.format({c: "{:.3f}" for c in METRIC_COLS if c in filtered.columns}),
        use_container_width=True,
    )
    csv_all = filtered.to_csv(index=False).encode()
    st.download_button("Download filtered results (CSV)", csv_all,
                       file_name="ml_results.csv", mime="text/csv")
