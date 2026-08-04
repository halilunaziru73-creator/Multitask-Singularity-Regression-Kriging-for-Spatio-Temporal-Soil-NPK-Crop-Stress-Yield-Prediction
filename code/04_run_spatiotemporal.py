import sys
sys.path.insert(0, '/home/claude/msrk/code')
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pykrige.ok import OrdinaryKriging
import warnings
warnings.filterwarnings('ignore')

st = pd.read_csv('/home/claude/msrk/data/brix_st_wide.csv')
date_cols = ['Jul_15', 'Jul_30', 'Aug_06', 'Aug_15', 'Aug_30']
doy_map = {'Jul_15': 196, 'Jul_30': 211, 'Aug_06': 218, 'Aug_15': 227, 'Aug_30': 242}

xy = st[['x', 'y']].to_numpy()


# ---- space-time singularity index: nested space-time cylinders A(s,t,r,tau) ----
def st_singularity_index(xy, t, v, radii, taus):
    n = len(xy)
    logC = []
    logvols = []
    for r in radii:
        for tau in taus:
            content = np.zeros(n)
            for i in range(n):
                dx = xy[:, 0] - xy[i, 0]
                dy = xy[:, 1] - xy[i, 1]
                dt = t - t[i]
                mask = (dx ** 2 + dy ** 2 <= r ** 2) & (np.abs(dt) <= tau)
                vol = np.pi * r ** 2 * (2 * tau + 1e-6)
                content[i] = (v[mask] - v.min() + 1e-6).sum() / vol
            logC.append(np.log(content + 1e-9))
            logvols.append(np.log(r) + np.log(2 * tau + 1e-6))
    logC = np.array(logC).T  # (n, n_scales)
    logvols = np.array(logvols)
    alpha = np.zeros(n)
    for i in range(n):
        slope, intercept = np.polyfit(logvols, logC[i], 1)
        alpha[i] = slope + 3.0  # 3-D (2 space + 1 time) support
    return alpha


long = pd.read_csv('/home/claude/msrk/data/brix_st_long.csv')
xyt = long[['x', 'y']].to_numpy()
t = long['t'].to_numpy()
v = long['brix'].to_numpy()

print('Computing space-time singularity index for Brix maturation field...')
alpha_st = st_singularity_index(xyt, t, v, radii=(150, 300), taus=(7, 15))
long['alpha_brix_st'] = alpha_st
long.to_csv('/home/claude/msrk/data/brix_st_long_with_singularity.csv', index=False)
print(pd.Series(alpha_st).describe())

# ---- multitask spatio-temporal trend + residual kriging (per-date OK on residuals) ----
long['x_n'] = (long['x'] - long['x'].mean()) / long['x'].std()
long['y_n'] = (long['y'] - long['y'].mean()) / long['y'].std()
long['t_n'] = (long['t'] - long['t'].mean()) / long['t'].std()

X = long[['x_n', 'y_n', 't_n', 'alpha_brix_st', 'ndvi']].to_numpy()
y = long['brix'].to_numpy()

kf = KFold(n_splits=5, shuffle=True, random_state=3)
methods = ['LM_st', 'RF_st', 'STRK']  # spatio-temporal RF+kriging = STRK (space-time SRK)
from sklearn.linear_model import LinearRegression
preds = {m: np.zeros(len(long)) for m in methods}
for tr, te in kf.split(long):
    lm = LinearRegression().fit(X[tr], y[tr])
    preds['LM_st'][te] = lm.predict(X[te])

    rf = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=3,
                                random_state=3, n_jobs=-1).fit(X[tr], y[tr])
    mu_tr, mu_te = rf.predict(X[tr]), rf.predict(X[te])
    preds['RF_st'][te] = mu_te

    # residual kriging per date using only that date's training residuals
    res_tr = y[tr] - mu_tr
    tr_dates = long['date'].to_numpy()[tr]
    te_dates = long['date'].to_numpy()[te]
    pred_strk = mu_te.copy()
    for dc in date_cols:
        m_tr = tr_dates == dc
        m_te = te_dates == dc
        if m_tr.sum() < 4 or m_te.sum() == 0:
            continue
        xy_tr_d = xyt[tr][m_tr]
        xy_te_d = xyt[te][m_te]
        try:
            ok = OrdinaryKriging(xy_tr_d[:, 0], xy_tr_d[:, 1], res_tr[m_tr],
                                  variogram_model='spherical', nlags=6, enable_plotting=False)
            z, _ = ok.execute('points', xy_te_d[:, 0], xy_te_d[:, 1])
            pred_strk[np.where(m_te)[0]] += np.asarray(z)
        except Exception:
            pass
    preds['STRK'][te] = pred_strk

rows = []
for m, p in preds.items():
    rows.append({'model': m, 'R2': r2_score(y, p), 'RMSE': np.sqrt(mean_squared_error(y, p)),
                 'MAE': mean_absolute_error(y, p)})
cv_st = pd.DataFrame(rows)
cv_st.to_csv('/home/claude/msrk/out/cv_results_spatiotemporal.csv', index=False)
print('\n=== Spatio-temporal Brix CV (5-fold) ===')
print(cv_st.round(3))

# feature importance from a full-data RF fit (for reporting)
rf_full = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=3,
                                 random_state=3, n_jobs=-1).fit(X, y)
imp = pd.Series(rf_full.feature_importances_, index=['x', 'y', 't', 'alpha_brix_st', 'ndvi'])
imp.to_csv('/home/claude/msrk/out/feature_importance_st.csv')
print('\nFeature importances (space-time RF trend):')
print(imp.sort_values(ascending=False))
