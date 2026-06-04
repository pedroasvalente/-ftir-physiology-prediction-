import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import RESULTS_DIR, render_data_sidebar
from ftir_pred.analysis.statistical_validation import bland_altman

st.set_page_config(page_title="Statistical Validation", layout="wide")
st.title("Statistical Validation")

df = render_data_sidebar()
if df is None or df.empty:
    st.stop()

# Load predictions from JSON
pred_files = sorted(RESULTS_DIR.rglob("predictions_data.json"))
if not pred_files:
    st.info("No predictions_data.json found. Run a training experiment first.")
    st.stop()

pred_data: dict = {}
for f in pred_files:
    try:
        pred_data.update(json.loads(f.read_text()))
    except Exception:
        pass

if not pred_data:
    st.info("No prediction data available.")
    st.stop()

keys = sorted(pred_data.keys())
selected = st.selectbox("Select model run", keys)

entry = pred_data[selected]
y_test = np.array(entry["y_test"])
y_pred = np.array(entry["y_pred"])
target = entry.get("target", "")
sample_type = entry.get("sample_type", "")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Predicted vs Measured")
    fig = go.Figure()
    lim = [float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))]
    fig.add_trace(go.Scatter(x=y_test.tolist(), y=y_pred.tolist(), mode="markers",
                              marker=dict(size=7, opacity=0.7)))
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="red", dash="dash"),
                              name="Identity"))
    fig.update_layout(xaxis_title="Measured", yaxis_title="Predicted",
                      title=f"{target} — {sample_type}", height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Bland-Altman")
    ba = bland_altman(y_test, y_pred)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ba["mean"], y=ba["diff"], mode="markers",
                               marker=dict(size=7, opacity=0.7)))
    for y_val, color, label in [
        (ba["bias"], "blue", f"Bias = {ba['bias']:.3f}"),
        (ba["loa_upper"], "red", f"+1.96 SD = {ba['loa_upper']:.3f}"),
        (ba["loa_lower"], "red", f"−1.96 SD = {ba['loa_lower']:.3f}"),
    ]:
        fig2.add_hline(y=y_val, line_color=color, line_dash="dash" if y_val != ba["bias"] else "solid",
                       annotation_text=label)
    fig2.update_layout(xaxis_title="Mean of measured and predicted",
                       yaxis_title="Difference (measured − predicted)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()
col_r2, col_rmse, col_r = st.columns(3)
row = df[df["model"] == entry.get("model", "")]
if not row.empty:
    row = row.iloc[0]
    col_r2.metric("R²", f"{row.get('r2', '?'):.3f}",
                  f"[{row.get('r2_ci95_low', '?'):.3f}, {row.get('r2_ci95_high', '?'):.3f}] 95% CI")
    col_rmse.metric("RMSE", f"{row.get('rmse', '?'):.3f}")
    col_r.metric("Pearson r", f"{row.get('pearson_r', '?'):.3f}",
                 f"p = {row.get('pearson_p', '?'):.4f}")
