<div align="center">

# Multitask Singularity Regression Kriging for Spatio-Temporal Mapping of Soil NPK Dynamics, Crop Stress, and Yield Prediction in Precision Agriculture

### A Deep-Analysis GeoAI Study with Geostatistical, Machine-Learning, and Decision-Support Extensions

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-active-brightgreen)
![Domain](https://img.shields.io/badge/domain-GeoAI%20%7C%20Geostatistics%20%7C%20Precision%20Agriculture-informational)

**Author:** Naziru Halilu
Universidade de Trás-os-Montes e Alto Douro (UTAD) — GIS, Remote Sensing and Precision Agriculture Research

</div>

---

## Table of Contents

- [Overview](#overview)
- [The Five Research Questions](#the-five-research-questions)
- [Key Results](#key-results)
- [Figures](#figures)
- [Repository Structure](#repository-structure)
- [How to Run the Code](#how-to-run-the-code)
- [Documents](#documents)
- [License](#license)

---

## Overview

Regression kriging couples a data-driven trend model with geostatistical interpolation of
residuals, but standard implementations treat each agronomic response variable
independently, use a fixed single-scale notion of spatial context, and rarely report the
deeper geostatistical diagnostics needed to trust a management map.

This study develops and rigorously interrogates **Multitask Singularity Regression Kriging
(MSRK)**, a GeoAI framework that:

1. Engineers a **multiscale local singularity index**, α<sub>k</sub>(s), for every task from
   local-singularity/multifractal theory
2. Trains a **shared multi-output Random Forest trend model** across correlated agronomic
   tasks (yield, NPK, pH, CaCO₃)
3. **Krigs the resulting residuals** per task and, for repeated sampling, per date

MSRK is instantiated and stress-tested on real precision-agriculture data from three linked
University field practicals at UTAD:

| Dataset | Site | Scale | Tasks |
|---|---|---|---|
| Maize grid | Coimbra-region field | 16.85 ha, *n* = 17,864 | Yield, NPK, soil pH, CaCO₃ |
| Vineyard ripening monitoring | Quinta de Nossa Senhora de Lurdes | 6 ha, 68 stations × 5 dates | Brix, NDVI, crop-stress index |
| Soil-chemistry workbook | Auxiliary dataset | *n* = 35,448 | Soil chemistry → yield |
| Grape-moth IPM planning | Quinta da Senhora da Graça | 42.97 ha | Diffuser-density prescription |

---

## The Five Research Questions

| # | Question |
|---|---|
| **RQ1** | Does engineering a multiscale local singularity index as a trend-model feature improve spatial prediction accuracy for agronomic variables beyond standard baselines? |
| **RQ2** | Where does a shared multitask trend model (jointly predicting yield, NPK, pH, and CaCO₃) add value relative to single-task models, and where is it neutral? |
| **RQ3** | Does the singularity/multitask machinery reduce prediction *uncertainty* (kriging variance), even where it doesn't improve point-prediction accuracy? |
| **RQ4** | Does extending the singularity index and residual kriging into the space-time domain improve spatio-temporal crop-stress (Brix) prediction? |
| **RQ5** | What do rigorous geostatistical diagnostics (Moran's I, variogram-model comparison, sensitivity analysis, residual normality) reveal about when this machinery is worth its complexity? |

---

## Key Results

Leakage-controlled 5-fold spatial cross-validation, six competing estimators (maize field, *n* = 17,864):

| Model | Yield R² | NPK R² | pH R² | CaCO₃ R² | Yield RMSE | NPK RMSE |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| LM | 0.187 | 0.189 | 0.307 | 0.306 | 0.188 | 22.816 |
| IDW | 0.734 | 0.728 | 0.994 | 0.994 | 0.108 | 13.208 |
| **OK** | **0.736** | 0.710 | **1.000** | **1.000** | **0.107** | 13.655 |
| RF | 0.715 | 0.703 | 0.961 | 0.961 | 0.111 | 13.800 |
| RFK | 0.725 | **0.706** | 0.993 | 0.993 | 0.109 | 13.742 |
| SRK (proposed) | 0.709 | 0.686 | 0.987 | 0.980 | 0.112 | 14.209 |

**Honest headline finding:** ordinary kriging (OK) alone is close to optimal for the
smoothest, most autocorrelated fields (soil pH, CaCO₃; Moran's I ≈ 0.98) — SRK does **not**
win on raw R² here. Its value shows up elsewhere: a **20% reduction in residual-kriging
uncertainty (variance)** relative to RFK, and mechanistic robustness in the spatio-temporal
setting, contingent on adequate station density. This distinction — where added model
complexity helps and where it doesn't — is the actual subject of the paper, not a caveat
buried in it.

Global spatial autocorrelation (Moran's I) confirmed strong, highly significant clustering
across all tasks (I = 0.72–0.98, all *p* < 0.001), justifying the geostatistical approach in
the first place.

Also uncovered: the deterministic quadratic prescription equations underlying the NPK and
CaCO₃ covariates, verified directly against the source field report — this is what produces
the near-perfect (|ρ| ≈ 0.99–1.00) cross-task correlations controlled for throughout the
analysis.

---

## Figures

All 22 data-derived figures, extracted directly from the manuscript:

<table>
<tr><td width="50%">

**Figure 1 — Spearman Correlation Matrix**
![Spearman correlation matrix](figures/Figure_01_Spearman_correlation_singularity_index.png)

</td><td width="50%">

**Figure 2 — Multiscale Local Singularity Index**
![Multiscale singularity index](figures/Figure_02_Multiscale_local_singularity_index.png)

</td></tr>
<tr><td width="50%">

**Figure 3 — Cross-Validated R² Heatmap**
![CV R2 heatmap](figures/Figure_03_CV_R2_heatmap_model_task.png)

</td><td width="50%">

**Figure 4 — RMSE Relative to IDW Baseline**
![RMSE relative to IDW](figures/Figure_04_RMSE_relative_to_IDW_baseline.png)

</td></tr>
<tr><td width="50%">

**Figure 5 — Observed vs. Predicted (Yield, NPK)**
![Observed vs predicted](figures/Figure_05_Observed_vs_predicted_corn_yield_NPK.png)

</td><td width="50%">

**Figure 6 — Brix Maturation Trajectories**
![Brix maturation trajectories](figures/Figure_06_Brix_maturation_trajectories.png)

</td></tr>
<tr><td width="50%">

**Figure 7 — Crop-Ripening Stress, NDVI, Space-Time Index**
![Crop ripening stress NDVI](figures/Figure_07_Crop_ripening_stress_NDVI_spacetime.png)

</td><td width="50%">

**Figure 8 — Spatio-Temporal Feature Importance & CV R²**
![Spacetime feature importance](figures/Figure_08_Spacetime_feature_importance_CV_R2.png)

</td></tr>
<tr><td width="50%">

**Figure 9 — Feature Importance & Yield-Response Curves**
![Feature importance response curves](figures/Figure_09_Feature_importance_yield_response_curves.png)

</td><td width="50%">

**Figure 10 — Vineyard Block Diffuser Density**
![Vineyard diffuser density](figures/Figure_10_Vineyard_block_diffuser_density.png)

</td></tr>
<tr><td width="50%">

**Figure 11 — Moran's I Scatterplot**
![Morans I scatterplot](figures/Figure_11_Morans_I_scatterplot.png)

</td><td width="50%">

**Figure 12 — Experimental Semivariograms**
![Semivariograms](figures/Figure_12_Experimental_semivariograms_fitted_models.png)

</td></tr>
<tr><td width="50%">

**Figure 13 — Sensitivity Heatmap (Radius Bank)**
![Sensitivity heatmap](figures/Figure_13_Sensitivity_heatmap_radius_scales.png)

</td><td width="50%">

**Figure 14 — Out-of-Fold Permutation Importance**
![Permutation importance](figures/Figure_14_Out_of_fold_permutation_importance.png)

</td></tr>
<tr><td width="50%">

**Figure 15 — PCA Biplot (Soil Chemistry)**
![PCA biplot](figures/Figure_15_PCA_biplot_soil_chemistry.png)

</td><td width="50%">

**Figure 16 — West–East Transect (OK vs. RF vs. SRK)**
![Transect](figures/Figure_16_West_east_transect_OK_RF_SRK.png)

</td></tr>
<tr><td width="50%">

**Figure 17 — Residual-Kriging Uncertainty Comparison**
![Uncertainty comparison](figures/Figure_17_Residual_kriging_uncertainty_comparison.png)

</td><td width="50%">

**Figure 18 — Residual Diagnostics (Q-Q Plot)**
![Residual diagnostics](figures/Figure_18_Residual_diagnostics_QQ_plot.png)

</td></tr>
<tr><td width="50%">

**Figure 19 — Directional (Anisotropic) Semivariograms**
![Anisotropic semivariograms](figures/Figure_19_Directional_anisotropic_semivariograms.png)

</td><td width="50%">

**Figure 20 — Multi-Metric Radar Charts**
![Radar charts](figures/Figure_20_Multi_metric_radar_charts.png)

</td></tr>
<tr><td width="50%">

**Figure 21 — SRK 3-D Prediction Surface**
![3D prediction surface](figures/Figure_21_SRK_3D_prediction_surface.png)

</td><td width="50%">

**Figure 22 — Fold-Level R² Variability**
![Fold level R2 variability](figures/Figure_22_Fold_level_R2_variability.png)

</td></tr>
</table>

---

## Repository Structure

```
.
├── README.md
├── LICENSE
├── MSRK_Deep_Analysis_Naziru.docx      # Full manuscript
├── code/                                # Full reproducible Python pipeline
│   ├── 01_data_prep.py
│   ├── 02_msrk_engine.py / msrk_engine.py   # Core MSRK algorithm (singularity index, multitask RF, residual kriging)
│   ├── 03_run_spatial_multitask.py
│   ├── 04_run_spatiotemporal.py
│   ├── 05_make_figures.py               # Figures 1–10
│   ├── 06_soil_chem_yield.py            # Auxiliary soil-chemistry → yield experiment (Table 6, Fig. 10)
│   ├── 07_deep_analysis.py
│   ├── 08_deep_analysis2.py             # Permutation importance, PCA, transect, RFK-vs-SRK uncertainty, Q-Q diagnostics
│   ├── 09_advanced_figures.py           # Figures 11–18
│   └── 10_extra_complex_figures.py      # Figures 19–22
├── figures/                             # All 22 publication-quality figures (PNG)
└── outputs_data/                        # Numeric results (22 CSVs): CV results, feature importance,
                                          # variogram/anisotropy summaries, PCA loadings, partial-dependence curves
```

---

## How to Run the Code

### 1. Clone the repository

```bash
git clone https://github.com/halilunaziru73-creator/Multitask-Singularity-Regression-Kriging-for-Spatio-Temporal-Soil-NPK-Crop-Stress-Yield-Prediction.git
cd Multitask-Singularity-Regression-Kriging-for-Spatio-Temporal-Soil-NPK-Crop-Stress-Yield-Prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

### 3. Run the pipeline in order

```bash
cd code
python 01_data_prep.py                  # data preparation
python 02_msrk_engine.py                # core MSRK engine
python 03_run_spatial_multitask.py      # RQ1–RQ3: spatial multitask experiment
python 04_run_spatiotemporal.py         # RQ4: spatio-temporal Brix experiment
python 05_make_figures.py               # → Figures 1–10
python 06_soil_chem_yield.py            # auxiliary soil-chemistry experiment
python 07_deep_analysis.py
python 08_deep_analysis2.py             # permutation importance, PCA, transect, uncertainty grid, Q-Q
python 09_advanced_figures.py           # → Figures 11–18
python 10_extra_complex_figures.py      # → Figures 19–22
```

---

## Documents

| Document | Description |
|---|---|
| [`MSRK_Deep_Analysis_Naziru.docx`](./MSRK_Deep_Analysis_Naziru.docx) | Full manuscript: methodology, five research questions, results, and discussion |

---

## License

Released under the [MIT License](./LICENSE).
