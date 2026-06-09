import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import (
    DEFAULT_GROUP_COLORS, TARGET_GROUP_COLORS,
    render_appearance_sidebar, render_data_sidebar,
)
from ftir_pred.config import TRAINING_DATA_PATH
from ftir_pred.data.config import REGRESSION_TARGETS, SAMPLE_TYPES, get_target_group
from ftir_pred.data.loader import get_ftir_columns, load_csv

st.set_page_config(page_title="Overview", layout="wide")
st.title("Study Overview")

matrix_colors, _ = render_appearance_sidebar(show_matrices=True)
df_res = render_data_sidebar()


@st.cache_data
def _load_data():
    df = load_csv(str(TRAINING_DATA_PATH))
    ftir_cols = get_ftir_columns(df)
    wn = np.array([float(c) for c in ftir_cols])
    return df, ftir_cols, wn


df, ftir_cols, wn = _load_data()

# ── Key metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
if "group" in df.columns and "person_code" in df.columns:
    n_persons = df.groupby("group")["person_code"].nunique().sum()
else:
    n_persons = df["person_code"].nunique() if "person_code" in df.columns else "—"
c1.metric("Participants", n_persons)
c2.metric("Biological matrices", len(SAMPLE_TYPES))
c3.metric("Target variables", len(REGRESSION_TARGETS))
c4.metric("Spectral range", "950–3050 cm⁻¹")
if df_res is not None and not df_res.empty and "r2" in df_res.columns:
    ml = df_res[~df_res["is_baseline"].astype(bool)] if "is_baseline" in df_res.columns else df_res
    best_r2  = ml["r2"].max()
    best_row = ml.loc[ml["r2"].idxmax()]
    c5.metric("Best ML R²", f"{best_r2:.3f}",
              f"{best_row.get('target','?')} / {best_row.get('sample_type','?')}")
else:
    c5.metric("Best ML R²", "—")

st.divider()

# ── Sample counts ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Samples per matrix")
    counts = df["sample_type"].value_counts().reset_index()
    counts.columns = ["matrix", "n"]
    counts["matrix"] = pd.Categorical(counts["matrix"], categories=SAMPLE_TYPES, ordered=True)
    counts = counts.sort_values("matrix")
    fig = px.bar(
        counts, x="matrix", y="n",
        color="matrix", color_discrete_map=matrix_colors,
        text_auto=True,
        labels={"matrix": "Matrix", "n": "Samples"},
    )
    fig.update_layout(showlegend=False, height=320,
                      plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(t=20, b=40, l=50, r=10))
    fig.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Samples × subgroup × timepoint")
    if {"sample_type", "group", "timepoint"}.issubset(df.columns):
        pivot = (
            df.groupby(["sample_type", "group", "timepoint"])
            .size().reset_index(name="n")
            .pivot_table(index=["sample_type", "group"],
                         columns="timepoint", values="n", fill_value=0)
        )
        pivot.columns = [f"T{c}" for c in pivot.columns]
        pivot["Total"] = pivot.sum(axis=1)
        st.dataframe(
            pivot.style.background_gradient(cmap="Blues", subset=pivot.columns.tolist())
            .format("{:.0f}"),
            use_container_width=True,
        )

st.divider()

# ── Demographics ──────────────────────────────────────────────────────────────
st.subheader("Participant characteristics (mean ± SD)")
demo_cols = {
    "age_years": "Age (yr)", "height_cm": "Height (cm)",
    "bodyweight_kg": "Weight (kg)", "bodyfat_perc": "Body fat (%)",
    "ffm_kg": "FFM (kg)",
}
avail_demo = {k: v for k, v in demo_cols.items() if k in df.columns}

if avail_demo and "group" in df.columns:
    rows = []
    for grp, sub in df.groupby("group"):
        unique_in_grp = sub.drop_duplicates("person_code")
        grp_fam = sub["group_fam"].iloc[0].title() if "group_fam" in sub.columns else grp
        entry = {"Subgroup": grp, "Family": grp_fam, "n": len(unique_in_grp)}
        for col, label in avail_demo.items():
            vals = unique_in_grp[col].dropna()
            if len(vals) == 0:
                entry[label] = "—"
            elif len(vals) == 1:
                entry[label] = f"{vals.mean():.1f}"
            else:
                entry[label] = f"{vals.mean():.1f} ± {vals.std():.1f}"
        rows.append(entry)
    st.dataframe(
        pd.DataFrame(rows).set_index("Subgroup"),
        use_container_width=True,
    )
    st.caption("F1/F2 = football cohorts · G1/G2/G3 = sedentary cohorts (independent, person_code reused) · U1 = ultrarunning")

st.divider()

# ── Target variables by group ─────────────────────────────────────────────────
st.subheader("Target variables by physiological group")

tg_map = {t: get_target_group(t) for t in REGRESSION_TARGETS}
tg_counts = pd.Series(tg_map).value_counts().reset_index()
tg_counts.columns = ["group", "n_targets"]

col_a, col_b = st.columns([1, 2])
with col_a:
    fig_tg = px.pie(
        tg_counts, names="group", values="n_targets",
        color="group", color_discrete_map=TARGET_GROUP_COLORS,
        hole=0.4,
    )
    fig_tg.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10),
                         showlegend=True, legend=dict(orientation="v"))
    st.plotly_chart(fig_tg, use_container_width=True)

with col_b:
    rows = []
    for tg in sorted(tg_counts["group"]):
        targets_in_group = [t for t, g in tg_map.items() if g == tg]
        rows.append({"Physiological group": tg.replace("_", " ").title(),
                     "n": len(targets_in_group),
                     "Variables": ", ".join(targets_in_group[:6]) + ("…" if len(targets_in_group) > 6 else "")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.divider()

# ── Spectral quality ──────────────────────────────────────────────────────────
st.subheader("Spectral data completeness per matrix")
quality_rows = []
for mat in SAMPLE_TYPES:
    sub = df[df["sample_type"] == mat][ftir_cols]
    if sub.empty:
        continue
    pct = float((sub != 0).all(axis=0).mean() * 100)
    quality_rows.append({"matrix": mat, "non_zero_pct": pct, "n_samples": len(sub)})

if quality_rows:
    q_df = pd.DataFrame(quality_rows)
    fig_q = px.bar(
        q_df, x="matrix", y="non_zero_pct",
        color="matrix", color_discrete_map=matrix_colors,
        text_auto=".1f",
        labels={"matrix": "Matrix", "non_zero_pct": "Wavenumbers with no zeros (%)"},
        hover_data=["n_samples"],
    )
    fig_q.update_layout(showlegend=False, height=300, yaxis_range=[0, 105],
                        plot_bgcolor="white", paper_bgcolor="white",
                        margin=dict(t=20, b=40, l=60, r=10))
    fig_q.update_yaxes(showgrid=True, gridcolor="#e5e5e5")
    st.plotly_chart(fig_q, use_container_width=True)
