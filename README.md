# FTIR-ATR Spectroscopy for Physiological Prediction

Regression machine learning pipeline predicting 33 physiological variables from FTIR-ATR spectra of 5 biological fluids (capillary blood, plasma, saliva, serum, urine) across 138 individuals (6 cohorts: football, sedentary, ultrarunning).

## Study design

**Participants:** 138 individuals across 6 cohorts (F1: 37, F2: 13 football · G1/G2/G3: 15 each sedentary · U1: 43 ultrarunning), 3 measurement timepoints (T1–T3).

**Biological matrices:** Capillary blood · Plasma · Saliva · Serum · Urine (~200 samples per matrix).

**Spectral range:** 950–3050 cm⁻¹ (1850–2500 cm⁻¹ excluded — atmospheric CO₂ / H₂O).

**Target variables (33):**

| Physiological group | Variables |
|---|---|
| Cardiorespiratory (9) | VO₂max (absolute, relative, HR), Anaerobic threshold (3), Respiratory compensation (3) |
| Body composition (7) | Body weight, body fat, fat-free mass, total body water |
| CBC — haematology (14) | Erythrocytes, haemoglobin, haematocrit, MCH, MCV, MCHC, RDW, WBC differential |
| Hormonal (3) | Salivary cortisol, testosterone, IL-10 |

## Methods

| Step | Implementation |
|---|---|
| Preprocessing | SNV + Savitzky-Golay (optional) |
| Dimensionality reduction | PLS transformer (VIP-guided, inside CV pipeline) |
| Models | Random Forest, XGBoost, MLP Regressor, Decision Tree + 3 baselines (Dummy mean, Ridge, PLS-R 3 comp) |
| Hyperparameter search | GridSearchCV only |
| Cross-validation | GroupKFold (k=5, person-aware — no participant across train/test) |
| Train/test split | 80/20, person-aware via `split_by_person` |
| Metrics | R², RMSE, MAE, Pearson r — all with 95% bootstrap CIs |
| Spectral interpretation | VIP scores (Wold et al. 2001) + Mann-Whitney spectral AUC comparison (high vs low VO₂max) |
| Band assignments | Movasaghi et al. (2008) Applied Spectroscopy Reviews 43(2), 134–179 |

## Key results

Best ML performance (test set R²):

| Target | Matrix | Model | R² |
|---|---|---|---|
| Respiratory compensation (bpm) | Urine | Decision Tree | 0.396 |
| VO₂max absolute | Capillary blood | MLP Regressor | 0.383 |
| Neutrophils | Serum | Decision Tree | 0.356 |
| Anaerobic threshold (relative) | Plasma | XGBoost | 0.329 |
| Respiratory compensation (relative) | Urine | Random Forest | 0.314 |

PLS-R chemometric baseline: best R²=0.366 (respiratory compensation / Plasma).

## Repository structure

```
src/ftir_pred/
  data/         — loader, splits, config (targets, sample types)
  preprocessing/— SNV, Savitzky-Golay, PLS transformer
  models/       — model configs, training, evaluation (bootstrap CIs)
  analysis/     — spectral comparison, PLS-R direct, VIP interpretation
app/pages/      — Streamlit interactive dashboard (9 pages)
scripts/        — run_spectral_comparison.py, run_plsr.py
experiments/    — JSON configs for full training runs
results/        — training results, spectral comparison, PLS-R results
```

## Quick start

```bash
git clone https://github.com/pedroasvalente/ftir-physiology-prediction
cd ftir-physiology-prediction
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Launch Streamlit dashboard
streamlit run app/main.py

# Run spectral group comparison (high vs low VO2max)
python scripts/run_spectral_comparison.py

# Run PLS-R chemometric baseline
python scripts/run_plsr.py

# Full ML training
ftir-pred train experiments/configs/all_targets.json
```

## Streamlit dashboard

| Page | Content |
|---|---|
| Overview | Sample counts, demographics, target groups, spectral quality |
| Spectra | Mean ± SD spectra by matrix and sport group |
| ML Results | R² heatmap (target × matrix), top results, distributions |
| Model Comparison | Radar chart, model vs baseline, cross-matrix summary |
| Statistical Validation | Observed vs predicted, Bland-Altman (interactive) |
| Spectral Interpretation | VIP scores with biochemical band shading |
| Spectral Comparison | High vs low VO₂max AUC + Mann-Whitney (VIP-guided regions) |
| PLS-R | Chemometric baseline: CV curve, observed vs predicted, VIP |

## References

- Movasaghi, Z., Rehman, S., & Rehman, I.U. (2008). Fourier transform infrared (FTIR) spectroscopy of biological tissues. *Applied Spectroscopy Reviews*, 43(2), 134–179. https://doi.org/10.1080/05704920701829043
- Wold, S., Sjöström, M., & Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109–130.
- Valente, P.A. et al. (Study 1) FTIR-ATR spectroscopy for sport classification. University of Coimbra.

## License

CC BY-NC-ND 4.0 — Pedro Afonso Valente, University of Coimbra, 2025.
