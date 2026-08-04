"""
02_msrk_engine.py
Core GeoAI engine for Multitask Singularity Regression Kriging (MSRK).

Pipeline
--------
1. Multiscale singularity-index feature engineering
   For each target variable v and each point s, embed nested isotropic
   neighbourhoods A(s,r) at radii r in R = {r_1< ... < r_m}. The local
   density-normalised cumulative content C_v(s,r) = sum_{s_i in A(s,r)} v(s_i) / |A(s,r)|
   scales with r as a power law under multifractal singularity theory:
       C_v(s,r) ~ r^(alpha_v(s) - 2)               (2-D support)
   alpha_v(s) is estimated by ordinary least squares on log C_v(s,r) vs log r.
   alpha_v(s) < 2 flags local enrichment / hot-spot behaviour (e.g. nutrient or
   sugar-loading anomalies); alpha_v(s) > 2 flags depletion/smooth background.

2. Multitask trend estimation
   A single Random-Forest (multi-output) regressor is trained on
   [x, y, singularity indices alpha_k(s) for all tasks, cross-task covariates]
   to predict all tasks jointly, sharing structure across correlated
   agronomic variables (yield, NPK, pH, CaCO3 / Brix, NDVI).

3. Residual kriging
   Residuals epsilon_k(s) = z_k(s) - mu_hat_k(s) are modelled with an
   experimental variogram per task and interpolated with ordinary kriging;
   final prediction Z_hat_k(s0) = mu_hat_k(s0) + eps_hat_k(s0) (Singularity
   Regression Kriging, SRK). Baselines: OK, IDW, LM, RF, RFK (RF+kriging,
   no singularity features).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pykrige.ok import OrdinaryKriging
import warnings
warnings.filterwarnings('ignore')

RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------
# 1. Singularity index
# ----------------------------------------------------------------------
def singularity_index(coords, values, radii):
    """
    coords : (n,2) array of UTM x,y
    values : (n,) array, the covariate/task to profile
    radii  : list of neighbourhood radii (m), increasing
    returns alpha : (n,) local singularity exponent
    """
    tree = cKDTree(coords)
    n = len(coords)
    logC = np.zeros((n, len(radii)))
    logR = np.log(radii)
    v = values - values.min() + 1e-6  # positivity for log-content
    for j, r in enumerate(radii):
        idx_lists = tree.query_ball_point(coords, r)
        for i, idxs in enumerate(idx_lists):
            area = np.pi * r ** 2
            content = v[idxs].sum() / area
            logC[i, j] = np.log(content + 1e-9)
    alpha = np.zeros(n)
    for i in range(n):
        # OLS slope of logC vs logR  ->  content ~ r^(alpha-2)
        slope, intercept = np.polyfit(logR, logC[i], 1)
        alpha[i] = slope + 2.0
    return alpha


def build_singularity_features(df, task_cols, coords_cols=('x', 'y'),
                                radii=(50, 100, 200, 400)):
    coords = df[list(coords_cols)].to_numpy()
    feats = {}
    for t in task_cols:
        feats[f'alpha_{t}'] = singularity_index(coords, df[t].to_numpy(), radii)
    return pd.DataFrame(feats, index=df.index)


def singularity_index_from_train(train_xy, train_v, query_xy, radii):
    """
    Leakage-free variant: the local singularity exponent at each query location
    (train or held-out test) is estimated using ONLY the training observations
    that fall in its nested neighbourhoods. Required for valid cross-validation;
    mirrors how the feature would be computed operationally (interpolate from
    known sample stations to any new location).
    """
    tree = cKDTree(train_xy)
    v = train_v - train_v.min() + 1e-6
    logR = np.log(radii)
    nq = len(query_xy)
    logC = np.zeros((nq, len(radii)))
    for j, r in enumerate(radii):
        idx_lists = tree.query_ball_point(query_xy, r)
        area = np.pi * r ** 2
        for i, idxs in enumerate(idx_lists):
            if len(idxs) == 0:
                logC[i, j] = np.log(1e-9)
            else:
                logC[i, j] = np.log(v[idxs].sum() / area + 1e-9)
    alpha = np.zeros(nq)
    for i in range(nq):
        slope, intercept = np.polyfit(logR, logC[i], 1)
        alpha[i] = slope + 2.0
    return alpha


# ----------------------------------------------------------------------
# 2 & 3. Multitask SRK fit / predict with spatial K-fold CV
# ----------------------------------------------------------------------
def idw_predict(train_xy, train_z, query_xy, power=2, k=12):
    tree = cKDTree(train_xy)
    dist, idx = tree.query(query_xy, k=min(k, len(train_xy)))
    dist = np.clip(dist, 1e-6, None)
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    w = 1.0 / dist ** power
    w /= w.sum(axis=1, keepdims=True)
    return (w * train_z[idx]).sum(axis=1)


def ordinary_kriging_predict(train_xy, train_z, query_xy):
    try:
        ok = OrdinaryKriging(train_xy[:, 0], train_xy[:, 1], train_z,
                              variogram_model='spherical', verbose=False,
                              enable_plotting=False, nlags=12)
        z, ss = ok.execute('points', query_xy[:, 0], query_xy[:, 1])
        return np.asarray(z), np.asarray(ss)
    except Exception:
        # fallback to IDW if variogram fit fails on a fold
        z = idw_predict(train_xy, train_z, query_xy)
        return z, np.full(len(z), np.var(train_z))


def run_cv_all_methods(df, task_cols, other_task_cols, radii=(50, 100, 200, 400),
                        n_splits=5, seed=11):
    """
    Leakage-controlled spatial K-fold CV comparing LM, IDW, OK, RF, RFK, SRK for
    every task. All engineered features (own-task and cross-task singularity
    indices) are recomputed from the TRAINING fold only inside every split, then
    evaluated at held-out coordinates -- the same information a real deployment
    would have. Raw cross-task values are never used directly (several task
    columns in this dataset are near-collinear estimates of one another,
    r ~ -0.99, so raw cross-task features would leak the target).
    """
    results = []
    coords = df[['x', 'y']].to_numpy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for task in task_cols:
        y = df[task].to_numpy()
        base_xy = df[['x_n', 'y_n']].to_numpy()

        preds = {m: np.zeros(len(df)) for m in ['LM', 'IDW', 'OK', 'RF', 'RFK', 'SRK']}

        for tr_idx, te_idx in kf.split(df):
            xy_tr, xy_te = coords[tr_idx], coords[te_idx]
            y_tr = y[tr_idx]

            sing_feats_tr, sing_feats_all = {}, {}
            for t in [task] + other_task_cols:
                v_tr = df[t].to_numpy()[tr_idx]
                sing_feats_tr[f'alpha_{t}'] = singularity_index_from_train(xy_tr, v_tr, xy_tr, radii)
                sing_feats_all[f'alpha_{t}'] = singularity_index_from_train(xy_tr, v_tr, coords, radii)

            X_base_tr, X_base_te = base_xy[tr_idx], base_xy[te_idx]
            X_sing_tr = np.column_stack([base_xy[tr_idx]] +
                                         [sing_feats_tr[f'alpha_{t}'] for t in [task] + other_task_cols])
            X_sing_te = np.column_stack([base_xy[te_idx]] +
                                         [sing_feats_all[f'alpha_{t}'][te_idx] for t in [task] + other_task_cols])

            lm = LinearRegression().fit(X_base_tr, y_tr)
            preds['LM'][te_idx] = lm.predict(X_base_te)

            preds['IDW'][te_idx] = idw_predict(xy_tr, y_tr, xy_te)

            z_ok, _ = ordinary_kriging_predict(xy_tr, y_tr, xy_te)
            preds['OK'][te_idx] = z_ok

            rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4,
                                        random_state=seed, n_jobs=-1).fit(X_base_tr, y_tr)
            mu_rf_tr, mu_rf_te = rf.predict(X_base_tr), rf.predict(X_base_te)
            preds['RF'][te_idx] = mu_rf_te

            res_tr = y_tr - mu_rf_tr
            z_res, _ = ordinary_kriging_predict(xy_tr, res_tr, xy_te)
            preds['RFK'][te_idx] = mu_rf_te + z_res

            rfs = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4,
                                         random_state=seed, n_jobs=-1).fit(X_sing_tr, y_tr)
            mu_s_tr, mu_s_te = rfs.predict(X_sing_tr), rfs.predict(X_sing_te)
            res_s_tr = y_tr - mu_s_tr
            z_res_s, _ = ordinary_kriging_predict(xy_tr, res_s_tr, xy_te)
            preds['SRK'][te_idx] = mu_s_te + z_res_s

        for m, p in preds.items():
            r2 = r2_score(y, p)
            rmse = np.sqrt(mean_squared_error(y, p))
            mae = mean_absolute_error(y, p)
            results.append({'task': task, 'model': m, 'R2': r2, 'RMSE': rmse, 'MAE': mae})

    return pd.DataFrame(results)
