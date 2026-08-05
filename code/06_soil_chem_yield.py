"""
06_soil_chem_yield.py
Auxiliary experiment: soil-chemistry -> yield trend relationships using the larger,
non-georeferenced soil workbook (n=35,448). No XX_UTM/YY_UTM key is present in this
sheet, so it cannot be spatially joined to the SRK grid or kriged; instead we use it
as a large, independent sample to learn and report the soil-chemistry -> yield
relationships that motivate/validate the covariates used in the spatial model,
consistent with a multitask GeoAI framing (shared trend structure, no shared
coordinates here).

We also verify and document a useful internal structure of this workbook: several
variables are supplied twice - a coarse, discretely-binned raw soil-test value
(e.g. PH_H2O, P2O5, K2O, CaO, MgO, MO_PERC, Top_Soil_E, Deep_Soil_) and a smoothed
continuous companion (Soil_pH1, PO11, KO11, CA011, MgO1, PERC_MO1, WT1, WD1) whose
correlation with the raw version (r=0.86-0.90) is consistent with the companion
being a trend/kriged estimate of the coarse soil-test class. We use only the RAW
(coarse) soil-test covariates as predictors below, to avoid the same target-leakage
risk documented for the main spatial dataset.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

soil = pd.read_excel('../raw_data/CORN_SOIL_WATER_pH_OM_PO_KO_CA_Mg_WT_WD.xlsx', sheet_name=0)

RAW_COVARIATES = ['PH_H2O', 'MO_PERC', 'P2O5', 'K2O', 'CaO', 'MgO', 'Top_Soil_E', 'Deep_Soil_']
TARGET = 'CORN_KG_M2'

soil = soil.dropna(subset=RAW_COVARIATES + [TARGET]).reset_index(drop=True)
X = soil[RAW_COVARIATES].to_numpy()
y = soil[TARGET].to_numpy()

print(f'Auxiliary soil-chemistry dataset: n={len(soil)}, covariates={RAW_COVARIATES}')

kf = KFold(n_splits=5, shuffle=True, random_state=5)
pred_rf = np.zeros(len(y))
pred_lm = np.zeros(len(y))
from sklearn.linear_model import LinearRegression

for tr, te in kf.split(X):
    rf = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                random_state=5, n_jobs=-1).fit(X[tr], y[tr])
    pred_rf[te] = rf.predict(X[te])
    lm = LinearRegression().fit(X[tr], y[tr])
    pred_lm[te] = lm.predict(X[te])

rows = []
for name, p in [('LM', pred_lm), ('RF', pred_rf)]:
    rows.append({'model': name, 'R2': r2_score(y, p), 'RMSE': np.sqrt(mean_squared_error(y, p)),
                 'MAE': mean_absolute_error(y, p)})
cv_soilchem = pd.DataFrame(rows)
cv_soilchem.to_csv(os.path.join(THIS_DIR, '..', 'outputs_data', 'cv_results_soilchem_yield.csv'), index=False)
print(cv_soilchem.round(4))

# full-data RF fit for feature importance + partial-dependence-style binned means
rf_full = RandomForestRegressor(n_estimators=600, max_depth=10, min_samples_leaf=5,
                                 random_state=5, n_jobs=-1).fit(X, y)
imp = pd.Series(rf_full.feature_importances_, index=RAW_COVARIATES).sort_values(ascending=False)
imp.to_csv(os.path.join(THIS_DIR, '..', 'outputs_data', 'feature_importance_soilchem.csv'))
print('\nFeature importances (soil chemistry -> yield):')
print(imp)

# simple binned-mean response curves (cheap partial-dependence proxy) for top covariates
pd_curves = {}
for c in RAW_COVARIATES:
    bins = pd.qcut(soil[c], q=8, duplicates='drop')
    grp = soil.groupby(bins, observed=True)[TARGET].agg(['mean', 'std', 'count'])
    grp.index = [f'{iv.left:.2f}-{iv.right:.2f}' for iv in grp.index]
    pd_curves[c] = grp
    grp.to_csv(fos.path.join(THIS_DIR, '..', 'outputs_data', 'pdcurve_{c}.csv'))

print('\nSaved binned response curves for:', list(pd_curves.keys()))
