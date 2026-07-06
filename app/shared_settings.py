import os
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DEFAULT_MATRIX_COLORS = {
    "CAPILAR": "#1b9e77",
    "PLASMA":  "#d95f02",
    "SALIVA":  "#7570b3",
    "SERUM":   "#e7298a",
    "URINE":   "#66a61e",
}

DEFAULT_MODEL_COLORS = {
    "Random Forest": "#2166ac",
    "MLP Regressor": "#d6604d",
    "Decision Tree": "#4dac26",
    "XGBoost":       "#8073ac",
    "Dummy (mean)":  "#aaaaaa",
    "Ridge":         "#f4a442",
    "PLS (3 comp)":  "#a6cee3",
}

DEFAULT_GROUP_COLORS = {
    "football":    "#e41a1c",
    "sedentary":   "#377eb8",
    "ultrarunning":"#4daf4a",
}

TARGET_GROUP_COLORS = {
    "cardiorespiratory": "#d73027",
    "body_composition":  "#4575b4",
    "cbc":               "#1a9850",
    "hormonal":          "#8073ac",
}

TARGET_DISPLAY_NAMES: dict[str, str] = {
    # Cardiorespiratory
    "vo2max_absolute":                   "VO₂max (L/min)",
    "vo2max_relative":                   "VO₂max (mL/kg/min)",
    "vo2max_bpm":                        "VO₂max HR (bpm)",
    "anaerobicthreshold_absolute":       "Anaerobic threshold (L/min)",
    "anaerobicthreshold_relative":       "Anaerobic threshold (mL/kg/min)",
    "anaerobicthreshold_bpm":            "Anaerobic threshold (bpm)",
    "respiratorycompensation_absolute":  "Resp. compensation (L/min)",
    "respiratorycompensation_relative":  "Resp. compensation (mL/kg/min)",
    "respiratorycompensation_bpm":       "Resp. compensation (bpm)",
    # Body composition
    "bodyweight_kg":  "Body weight (kg)",
    "bodyfat_kg":     "Body fat (kg)",
    "bodyfat_perc":   "Body fat (%)",
    "ffm_kg":         "Fat-free mass (kg)",
    "ffm_%":          "Fat-free mass (%)",
    "h2o_L":          "Total body water (L)",
    "h20_perc":       "Total body water (%)",
    # CBC
    "erythrocytes":   "Erythrocytes",
    "hemoglobine":    "Haemoglobin",
    "hematocrit":     "Haematocrit",
    "mgv":            "MCV",
    "mch":            "MCH",
    "mchc":           "MCHC",
    "rdw":            "RDW",
    "leukocytes":     "Leukocytes (WBC)",
    "neutrophiles":   "Neutrophils",
    "eosinophils":    "Eosinophils",
    "basophils":      "Basophils",
    "lymphocytes":    "Lymphocytes",
    "monocytes":      "Monocytes",
    "platetes":       "Platelets",
    # Hormonal
    "salivarycortisol":     "Salivary cortisol",
    "salivarytestosterone": "Salivary testosterone",
    "il_10":                "IL-10",
}


def fmt_target(name: str) -> str:
    return TARGET_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


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
    _sidebar_footer()
    return st.session_state.matrix_colors, st.session_state.model_colors


def _sidebar_footer():
    st.sidebar.divider()
    st.sidebar.markdown(
        """
        <small>
        © 2025 Pedro Afonso Valente<br>
        University of Coimbra<br>
        <a href="https://github.com/pedroasvalente/ftir-physiology-prediction" target="_blank">
        GitHub repository</a><br>
        Licensed under CC BY-NC-ND 4.0
        </small>
        """,
        unsafe_allow_html=True,
    )


def _render_run_info(df: pd.DataFrame, run_label: str):
    matrices = sorted(df["sample_type"].dropna().unique()) if "sample_type" in df.columns else []
    models   = sorted(df["model"].dropna().unique())       if "model"       in df.columns else []
    tps      = sorted(df["timepoints"].fillna("all").unique()) if "timepoints" in df.columns else []

    with st.sidebar.expander("ℹ️ Loaded run", expanded=True):
        st.markdown(f"**Run:** `{run_label}`")
        st.markdown(f"**Total rows:** {len(df)}")
        if matrices:
            st.markdown(f"**Matrices:** {', '.join(matrices)}")
        if models:
            st.markdown(f"**Models ({len(models)}):** {', '.join(models)}")
        if "r2" in df.columns:
            ml = df[~df["is_baseline"].astype(bool)] if "is_baseline" in df.columns else df
            if not ml.empty:
                best_r2  = ml["r2"].max()
                best_row = ml.loc[ml["r2"].idxmax()]
                st.markdown(
                    f"**Best R²:** `{best_r2:.3f}` "
                    f"<small>({best_row.get('target','?')} / {best_row.get('sample_type','?')})</small>",
                    unsafe_allow_html=True,
                )


V2_CSV = RESULTS_DIR / "study_regression_v2" / "results_summary.csv"


@st.cache_data(ttl=300)
def _load_v2_results() -> pd.DataFrame:
    if not V2_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(V2_CSV)
    for col in ["r2", "rmse", "mae", "mape", "pearson_r", "pearson_p"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_baseline" in df.columns:
        df["is_baseline"] = df["is_baseline"].astype(bool)
    return df


def render_data_sidebar() -> pd.DataFrame | None:
    _init_defaults()
    with st.sidebar:
        st.header("Data")
        df = _load_v2_results()
        if df.empty:
            st.error("No results found. Run a training experiment first.")
            _sidebar_footer()
            return None

    _render_run_info(df, "study_regression_v2")
    _sidebar_footer()
    return df
