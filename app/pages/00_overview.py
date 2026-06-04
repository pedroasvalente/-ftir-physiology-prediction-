import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.shared_settings import render_data_sidebar

st.set_page_config(page_title="Overview", layout="wide")
st.title("Study Overview")

df = render_data_sidebar()

st.markdown("""
**Study:** Prediction of physiological variables from FTIR-ATR spectra of biological fluids.

**Participants:** 132 athletes and sedentary individuals (Sedentary · Football · Ultrarunning)

**Biological matrices:** Capillary blood · Plasma · Saliva · Serum · Urine

**Spectral range:** 950–3050 cm⁻¹ (1850–2500 cm⁻¹ excluded — water vapour region)

**Target variables:** 33 continuous physiological variables (cardiorespiratory, body composition, CBC, hormonal)

**Pipeline:**
- Person-aware 80/20 train/test split (no leakage across timepoints)
- GroupKFold cross-validation (5 folds, person-aware)
- PLS dimensionality reduction inside CV pipeline
- GridSearchCV + BayesSearchCV hyperparameter optimisation
- Bootstrap 95% confidence intervals on R², RMSE, MAE
""")

if df is not None and not df.empty:
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Targets", df["target"].nunique() if "target" in df.columns else "—")
    col2.metric("Matrices", df["sample_type"].nunique() if "sample_type" in df.columns else "—")
    col3.metric("Models evaluated", df["model"].nunique() if "model" in df.columns else "—")
    if "r2" in df.columns:
        ml = df[~df.get("is_baseline", pd.Series(False, index=df.index)).astype(bool)]
        best = ml["r2"].max()
        best_row = ml.loc[ml["r2"].idxmax()]
        col4.metric("Best R²", f"{best:.3f}",
                    f"{best_row.get('target','?')} / {best_row.get('sample_type','?')}")
