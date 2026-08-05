import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
from pykrige.ok import OrdinaryKriging
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(os.path.join(THIS_DIR, '..', 'data', 'corn_multitask_with_singularity.csv'))
TASKS = ['yield_kgm2', 'npk_ppm', 'soil_ph', 'caco3_ppm']
OUT = os.path.join(THIS_DIR, '..', 'outputs_data')

# =====================================================================
# 1. Global Moran's I (spatial autocorrelation) - k-NN row-standardised W
# =====================================================================
def morans_i(coords, values, k=8):
    n = len(coords)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # includes self at col 0
    idx = idx[:, 1:]
    z = values - values.mean()
    W_sum = n * k  # row-standardised weights sum to 1 per row -> total = n
    num = 0.0
    for i in range(n):
        num += z[i] * z[idx[i]].sum()
    num *= n
    den = (z ** 2).sum() * W_sum
    I = num / den
    EI = -1.0 / (n - 1)
    # analytic variance (randomisation assumption), simplified normal approx
    b2 = (n * (z ** 4).sum()) / ((z ** 2).sum() ** 2)
    S0 = W_sum
    S1 = 2 * n * k  # for row-standardised symmetric-ish binary weights approx
    S2 = 4 * n * (k ** 2)
    VarI = (n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
            - b2 * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)) / \
           ((n - 1) * (n - 2) * (n - 3) * S0 ** 2)
    z_score = (I - EI) / np.sqrt(max(VarI, 1e-12))
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return I, EI, z_score, p_value

coords = df[['x', 'y']].to_numpy()
moran_rows = []
for t in TASKS:
    I, EI, z, p = morans_i(coords, df[t].to_numpy(), k=8)
    moran_rows.append({'task': t, 'MoransI': I, 'Expected': EI, 'z': z, 'p_value': p})
moran_df = pd.DataFrame(moran_rows)
moran_df.to_csv(f'{OUT}/morans_i.csv', index=False)
print("=== Global Moran's I (k=8 nearest neighbours) ===")
print(moran_df.round(4))

# =====================================================================
# 2. Variogram model comparison (spherical / exponential / gaussian)
# =====================================================================
def empirical_semivariance(coords, values, n_lags=15, max_dist_frac=0.5):
    tree = cKDTree(coords)
    n = len(coords)
    rng = np.random.default_rng(0)
    # subsample pairs for tractability
    idx_pairs = rng.choice(n, size=(min(200000, n * 50), 2))
    d = np.sqrt(((coords[idx_pairs[:, 0]] - coords[idx_pairs[:, 1]]) ** 2).sum(axis=1))
    max_dist = np.quantile(d, max_dist_frac)
    mask = (d > 0) & (d <= max_dist)
    d, pv = d[mask], idx_pairs[mask]
    gamma = 0.5 * (values[pv[:, 0]] - values[pv[:, 1]]) ** 2
    bins = np.linspace(0, max_dist, n_lags + 1)
    lag_centers, sv = [], []
    for i in range(n_lags):
        m = (d >= bins[i]) & (d < bins[i + 1])
        if m.sum() > 10:
            lag_centers.append(0.5 * (bins[i] + bins[i + 1]))
            sv.append(gamma[m].mean())
    return np.array(lag_centers), np.array(sv)

def spherical(h, nugget, sill, rang):
    h = np.asarray(h)
    out = np.where(h <= rang, nugget + (sill - nugget) * (1.5 * h / rang - 0.5 * (h / rang) ** 3),
                   sill)
    return out

def exponential(h, nugget, sill, rang):
    return nugget + (sill - nugget) * (1 - np.exp(-3 * np.asarray(h) / rang))

def gaussian_vgm(h, nugget, sill, rang):
    return nugget + (sill - nugget) * (1 - np.exp(-3 * (np.asarray(h) / rang) ** 2))

vgm_results = []
vgm_curves = {}
for t in TASKS:
    lag, sv = empirical_semivariance(coords, df[t].to_numpy())
    vgm_curves[t] = (lag, sv)
    for name, fn in [('Spherical', spherical), ('Exponential', exponential), ('Gaussian', gaussian_vgm)]:
        try:
            p0 = [sv.min(), sv.max(), lag.max() / 2]
            popt, _ = curve_fit(fn, lag, sv, p0=p0, maxfev=8000,
                                 bounds=([0, 0, 1], [sv.max(), sv.max() * 3, lag.max() * 3]))
            pred = fn(lag, *popt)
            ss_res = ((sv - pred) ** 2).sum()
            ss_tot = ((sv - sv.mean()) ** 2).sum()
            r2 = 1 - ss_res / ss_tot
            vgm_results.append({'task': t, 'model': name, 'nugget': popt[0], 'sill': popt[1],
                                 'range_m': popt[2], 'R2_fit': r2})
        except Exception as e:
            vgm_results.append({'task': t, 'model': name, 'nugget': np.nan, 'sill': np.nan,
                                 'range_m': np.nan, 'R2_fit': np.nan})

vgm_df = pd.DataFrame(vgm_results)
vgm_df.to_csv(f'{OUT}/variogram_model_comparison.csv', index=False)
np.save(f'{OUT}/vgm_curves.npy', vgm_curves, allow_pickle=True)
print("\n=== Variogram model comparison (spherical/exponential/gaussian) ===")
print(vgm_df.round(3))

# =====================================================================
# 3. Sensitivity analysis: radius bank x number of scales -> mean R2 (SRK, all tasks)
# =====================================================================
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msrk_engine import singularity_index_from_train, ordinary_kriging_predict
from sklearn.metrics import r2_score

radius_options = {
    'fine': (25, 50, 100, 200),
    'medium': (50, 100, 200, 400),
    'coarse': (100, 200, 400, 800),
}
nscale_options = [2, 3, 4]

sens_rows = []
df_small = df.sample(n=500, random_state=1).reset_index(drop=True)  # tractable for a scan
coords_s = df_small[['x', 'y']].to_numpy()
x_n = (df_small['x'] - df_small['x'].mean()) / df_small['x'].std()
y_n = (df_small['y'] - df_small['y'].mean()) / df_small['y'].std()
base_xy = np.column_stack([x_n, y_n])

kf = KFold(n_splits=3, shuffle=True, random_state=2)
for rname, radii_full in radius_options.items():
    for ns in nscale_options:
        radii = radii_full[-ns:] if ns <= len(radii_full) else radii_full
        r2s = []
        for task in TASKS:
            y = df_small[task].to_numpy()
            other = [c for c in TASKS if c != task]
            preds = np.zeros(len(df_small))
            for tr, te in kf.split(df_small):
                sing_tr, sing_all = {}, {}
                for tcol in [task] + other:
                    v_tr = df_small[tcol].to_numpy()[tr]
                    sing_tr[tcol] = singularity_index_from_train(coords_s[tr], v_tr, coords_s[tr], radii)
                    sing_all[tcol] = singularity_index_from_train(coords_s[tr], v_tr, coords_s, radii)
                Xtr = np.column_stack([base_xy[tr]] + [sing_tr[c] for c in [task] + other])
                Xte = np.column_stack([base_xy[te]] + [sing_all[c][te] for c in [task] + other])
                rf = RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=3,
                                            random_state=1, n_jobs=-1).fit(Xtr, y[tr])
                mu_tr, mu_te = rf.predict(Xtr), rf.predict(Xte)
                res_tr = y[tr] - mu_tr
                z_res, _ = ordinary_kriging_predict(coords_s[tr], res_tr, coords_s[te])
                preds[te] = mu_te + z_res
            r2s.append(r2_score(y, preds))
        sens_rows.append({'radius_set': rname, 'n_scales': ns, 'mean_R2': np.mean(r2s)})

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(f'{OUT}/sensitivity_radii.csv', index=False)
print('\n=== Sensitivity analysis: radius bank x number of scales (mean R2 across 4 tasks) ===')
print(sens_df.round(3))

print('\nDeep analysis part 1 complete.')
