import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from pykrige.ok import OrdinaryKriging
from msrk_engine import singularity_index_from_train, ordinary_kriging_predict, idw_predict
import warnings
warnings.filterwarnings('ignore')

OUT = os.path.join(THIS_DIR, '..', 'outputs_data')
df = pd.read_csv(os.path.join(THIS_DIR, '..', 'data', 'corn_multitask_with_singularity.csv'))
TASKS = ['yield_kgm2', 'npk_ppm', 'soil_ph', 'caco3_ppm']
coords = df[['x', 'y']].to_numpy()
x_n = (df['x'] - df['x'].mean()) / df['x'].std()
y_n = (df['y'] - df['y'].mean()) / df['y'].std()
base_xy = np.column_stack([x_n, y_n])
radii = (50, 100, 200, 400)

# =====================================================================
# 4. Permutation importance (5-fold, out-of-fold) - more robust than impurity
# =====================================================================
perm_rows = []
for task in TASKS:
    other = [c for c in TASKS if c != task]
    y = df[task].to_numpy()
    kf = KFold(n_splits=5, shuffle=True, random_state=9)
    for tr, te in kf.split(df):
        sing_tr, sing_all = {}, {}
        for tcol in [task] + other:
            v_tr = df[tcol].to_numpy()[tr]
            sing_tr[tcol] = singularity_index_from_train(coords[tr], v_tr, coords[tr], radii)
            sing_all[tcol] = singularity_index_from_train(coords[tr], v_tr, coords, radii)
        feat_names = ['x', 'y'] + [f'alpha_{c}' for c in [task] + other]
        Xtr = np.column_stack([base_xy[tr]] + [sing_tr[c] for c in [task] + other])
        Xte = np.column_stack([base_xy[te]] + [sing_all[c][te] for c in [task] + other])
        rf = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=4,
                                    random_state=9, n_jobs=-1).fit(Xtr, y[tr])
        pi = permutation_importance(rf, Xte, y[te], n_repeats=8, random_state=9, n_jobs=-1)
        for fname, m, s in zip(feat_names, pi.importances_mean, pi.importances_std):
            perm_rows.append({'task': task, 'feature': fname, 'importance_mean': m, 'importance_std': s})

perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(f'{OUT}/permutation_importance.csv', index=False)
print('=== Permutation importance (mean over 5 folds x 8 repeats) ===')
print(perm_df.groupby(['task', 'feature'])['importance_mean'].mean().round(4))

# =====================================================================
# 5. PCA biplot on auxiliary soil-chemistry covariates
# =====================================================================
soil = pd.read_excel('../raw_data/CORN_SOIL_WATER_pH_OM_PO_KO_CA_Mg_WT_WD.xlsx', sheet_name=0)
RAW_COVARIATES = ['PH_H2O', 'MO_PERC', 'P2O5', 'K2O', 'CaO', 'MgO', 'Top_Soil_E', 'Deep_Soil_']
soil = soil.dropna(subset=RAW_COVARIATES + ['CORN_KG_M2']).reset_index(drop=True)
Xs = StandardScaler().fit_transform(soil[RAW_COVARIATES])
pca = PCA(n_components=4).fit(Xs)
scores = pca.transform(Xs)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
np.save(f'{OUT}/pca_scores.npy', scores[:3000])  # subsample for plotting
np.save(f'{OUT}/pca_loadings.npy', loadings)
pd.Series(pca.explained_variance_ratio_, name='explained_var_ratio').to_csv(f'{OUT}/pca_explained_variance.csv')
pd.DataFrame(loadings[:, :2], index=RAW_COVARIATES, columns=['PC1', 'PC2']).to_csv(f'{OUT}/pca_loadings_table.csv')
print('\n=== PCA explained variance ratio (soil-chemistry covariates) ===')
print(pca.explained_variance_ratio_.round(3))
print(pd.DataFrame(loadings[:, :2], index=RAW_COVARIATES, columns=['PC1', 'PC2']).round(3))

# =====================================================================
# 6. Cross-section transect: OK vs RF vs SRK along a west-east line
# =====================================================================
df_sub = df.copy()
y_mid = df_sub['y'].median()
band = df_sub[(df_sub['y'] > y_mid - 60) & (df_sub['y'] < y_mid + 60)]
x_line = np.linspace(df_sub['x'].min(), df_sub['x'].max(), 200)
y_line = np.full_like(x_line, y_mid)
query_xy = np.column_stack([x_line, y_line])

transect_results = {}
for task in TASKS:
    other = [c for c in TASKS if c != task]
    y = df_sub[task].to_numpy()
    # OK
    z_ok, ss_ok = ordinary_kriging_predict(coords, y, query_xy)
    # RF (coords only)
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4,
                                random_state=1, n_jobs=-1).fit(base_xy, y)
    x_line_n = (x_line - df_sub['x'].mean()) / df_sub['x'].std()
    y_line_n = (y_line - df_sub['y'].mean()) / df_sub['y'].std()
    z_rf = rf.predict(np.column_stack([x_line_n, y_line_n]))
    # SRK (coords + singularity, fit on full data, singularity computed from full data as "train")
    sing_full, sing_query = {}, {}
    for tcol in [task] + other:
        v = df_sub[tcol].to_numpy()
        sing_full[tcol] = singularity_index_from_train(coords, v, coords, radii)
        sing_query[tcol] = singularity_index_from_train(coords, v, query_xy, radii)
    Xfull = np.column_stack([base_xy] + [sing_full[c] for c in [task] + other])
    Xquery = np.column_stack([np.column_stack([x_line_n, y_line_n])] + [sing_query[c] for c in [task] + other])
    rfs = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4,
                                 random_state=1, n_jobs=-1).fit(Xfull, y)
    mu_q = rfs.predict(Xquery)
    res = y - rfs.predict(Xfull)
    z_res, ss_srk = ordinary_kriging_predict(coords, res, query_xy)
    z_srk = mu_q + z_res
    transect_results[task] = {'x': x_line, 'OK': z_ok, 'RF': z_rf, 'SRK': z_srk,
                               'OK_var': ss_ok, 'SRK_var': ss_srk}

np.save(f'{OUT}/transect_results.npy', transect_results, allow_pickle=True)
print('\nCross-section transect computed for all 4 tasks.')

# =====================================================================
# 7. Kriging uncertainty comparison: RFK vs SRK residual-kriging variance,
#    evaluated on a regular prediction grid (yield task)
# =====================================================================
task = 'yield_kgm2'
other = [c for c in TASKS if c != task]
gx = np.linspace(df['x'].min(), df['x'].max(), 40)
gy = np.linspace(df['y'].min(), df['y'].max(), 40)
GX, GY = np.meshgrid(gx, gy)
grid_xy = np.column_stack([GX.ravel(), GY.ravel()])

y = df[task].to_numpy()
rf_base = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4,
                                 random_state=1, n_jobs=-1).fit(base_xy, y)
res_base = y - rf_base.predict(base_xy)
_, var_rfk = ordinary_kriging_predict(coords, res_base, grid_xy)

sing_full, sing_grid = {}, {}
for tcol in [task] + other:
    v = df[tcol].to_numpy()
    sing_full[tcol] = singularity_index_from_train(coords, v, coords, radii)
    sing_grid[tcol] = singularity_index_from_train(coords, v, grid_xy, radii)
Xfull = np.column_stack([base_xy] + [sing_full[c] for c in [task] + other])
grid_xn = (gx - df['x'].mean()) / df['x'].std()
grid_yn = (gy - df['y'].mean()) / df['y'].std()
GXN, GYN = np.meshgrid(grid_xn, grid_yn)
Xgrid = np.column_stack([np.column_stack([GXN.ravel(), GYN.ravel()])] + [sing_grid[c] for c in [task] + other])
rfs = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4,
                             random_state=1, n_jobs=-1).fit(Xfull, y)
res_s = y - rfs.predict(Xfull)
_, var_srk = ordinary_kriging_predict(coords, res_s, grid_xy)

np.savez(f'{OUT}/uncertainty_grid.npz', GX=GX, GY=GY, var_rfk=var_rfk.reshape(GX.shape),
         var_srk=var_srk.reshape(GX.shape))
print('\nUncertainty grids (RFK vs SRK residual-kriging variance) computed for yield task.')
print(f'RFK var: mean={var_rfk.mean():.5f}, SRK var: mean={var_srk.mean():.5f}')

# =====================================================================
# 8. Residual diagnostics (QQ-plot data) for SRK, yield task, in-sample
# =====================================================================
resid = res_s  # SRK residuals (post multitask RF trend) for yield
(osm, osr), (slope, intercept, r) = stats.probplot(resid, dist='norm')
np.savez(f'{OUT}/qq_yield_srk.npz', osm=osm, osr=osr, slope=slope, intercept=intercept, r=r)
print(f'\nSRK yield-trend residual QQ fit: r={r:.4f} (near 1 = approx. normal residuals)')
print('Deep analysis part 2 complete.')
