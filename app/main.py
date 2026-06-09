import streamlit as st

st.set_page_config(
    page_title="FTIR Physiology Prediction",
    page_icon="🧬",
    layout="wide",
)

st.title("FTIR-ATR Spectroscopy for Physiological Prediction")
st.markdown(
    "Regression ML pipeline predicting **33 physiological variables** from "
    "FTIR-ATR spectra of **5 biological fluids** in **138 individuals** "
    "(football, sedentary, ultrarunning cohorts)."
)

st.divider()

# ── Key study metrics ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Participants", "138", "6 cohorts")
c2.metric("Biological matrices", "5", "Capillary · Plasma · Saliva · Serum · Urine")
c3.metric("Target variables", "33", "4 physiological groups")
c4.metric("Spectral range", "950–3050 cm⁻¹", "1850–2500 excluded")
c5.metric("Best ML R²", "0.396", "Resp. compensation / Urine / DT")

st.divider()

# ── Dashboard pages ────────────────────────────────────────────────────────────
st.subheader("Dashboard pages")

pages = [
    ("📊 Overview",                "Sample counts, demographics, target groups, spectral completeness"),
    ("🌈 Spectra",                 "Mean ± SD spectra by matrix and cohort, with biochemical band shading"),
    ("🏆 ML Results",              "R² heatmap (target × matrix), top results, model distributions"),
    ("⚖️ Model Comparison",        "Radar chart, R² vs RMSE scatter, ML vs baseline boxplot"),
    ("🔬 Statistical Validation",  "Observed vs predicted, Bland-Altman, all-targets summary per matrix"),
    ("🧪 Spectral Interpretation", "VIP scores with biochemical band shading, cascading target/matrix/model filters"),
    ("📉 Spectral Comparison",     "High vs low VO₂max AUC comparison (Mann-Whitney) across VIP regions"),
    ("📐 PLS-R",                   "Chemometric baseline: CV curve, observed vs predicted, VIP by region"),
]

col_a, col_b = st.columns(2)
for i, (name, desc) in enumerate(pages):
    (col_a if i % 2 == 0 else col_b).info(f"**{name}**  \n{desc}")

st.divider()

# ── Methods summary ────────────────────────────────────────────────────────────
st.subheader("Methods")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**Preprocessing**
SNV normalisation + Savitzky-Golay smoothing (optional)

**Models**
Random Forest · XGBoost · MLP Regressor · Decision Tree
*Baselines:* Dummy (mean) · Ridge · PLS-R (3 comp)

**Cross-validation**
GroupKFold (k=5) — person-aware, no participant leakage across folds
""")

with col2:
    st.markdown("""
**Spectral interpretation**
VIP scores (Wold et al. 2001) + Mann-Whitney AUC comparison
(high vs low VO₂max) in VIP-guided spectral regions

**Band assignments**
Movasaghi et al. (2008) *Applied Spectroscopy Reviews* 43(2), 134–179
11 validated bands, 950–3050 cm⁻¹

**Metrics**
R², RMSE, MAE, Pearson r — all with 95% bootstrap CIs
""")

st.divider()
st.caption(
    "Pedro Afonso Valente · University of Coimbra · 2025 · "
    "[GitHub](https://github.com/pedroasvalente/ftir-physiology-prediction) · "
    "CC BY-NC-ND 4.0"
)
