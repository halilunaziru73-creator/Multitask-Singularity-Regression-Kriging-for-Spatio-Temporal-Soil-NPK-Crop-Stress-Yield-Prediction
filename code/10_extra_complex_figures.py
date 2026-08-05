import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from pykrige.ok import OrdinaryKriging
from msrk_engine import singularity_index_from_train, ordinary_kriging_predict
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
FIG = os.path.join(THIS_DIR, '..', 'figures')
OUT = os.path.join(THIS_DIR, '..', 'outputs_data')
TASKS = ['yield_kgm2', 'npk_ppm', 'soil_ph', 'caco3_ppm']
TASK_LABELS = {'yield_kgm2': 'Corn yield', 'npk_ppm': 'NPK (est.)',
               'soil_ph': 'Soil pH', 'caco3_ppm': 'CaCO$_3$ (est.)'}

df = pd.read_csv(os.path.join(THIS_DIR, '..', 'data', 'corn_multitask_with_singularity.csv'))
coords = df[['x', 'y']].to_numpy()
x_n = (df['x'] - df['x'].mean()) / df['x'].std()
y_n = (df['y'] - df['y'].mean()) / df['y'].std()
base_xy = np.column_stack([x_n, y_n])
radii = (50, 100, 200, 400)

# =====================================================================
# Fig 19: Directional (anisotropic) empirical semivariograms, 4 azimuths
# =====================================================================
def directional_semivariance(coords, values, azimuth_deg, tol_deg=22.5, n_lags=10, max_dist_frac=0.5):
    n = len(coords)
    rng = np.random.default_rng(1)
    idx_pairs = rng.choice(n, size=(min(300000, n * 60), 2))
    d_vec = coords[idx_pairs[:, 1]] - coords[idx_pairs[:, 0]]
    dist = np.sqrt((d_vec ** 2).sum(axis=1))
    ang = (np.degrees(np.arctan2(d_vec[:, 1], d_vec[:, 0])) + 360) % 180  # 0-180 (undirected)
    az = azimuth_deg % 180
    diff = np.minimum(np.abs(ang - az), 180 - np.abs(ang - az))
    max_dist = np.quantile(dist, max_dist_frac)
    mask = (dist > 0) & (dist <= max_dist) & (diff <= tol_deg)
    d, pv = dist[mask], idx_pairs[mask]
    gamma = 0.5 * (values[pv[:, 0]] - values[pv[:, 1]]) ** 2
    bins = np.linspace(0, max_dist, n_lags + 1)
    lag_c, sv = [], []
    for i in range(n_lags):
        m = (d >= bins[i]) & (d < bins[i + 1])
        if m.sum() > 15:
            lag_c.append(0.5 * (bins[i] + bins[i + 1]))
            sv.append(gamma[m].mean())
    return np.array(lag_c), np.array(sv)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
azimuths = [0, 45, 90, 135]
colors = ['#2b6cb0', '#c05621', '#2f855a', '#805ad5']
aniso_summary = []
for ax, task in zip(axes, ['yield_kgm2', 'npk_ppm']):
    v = df[task].to_numpy()
    for az, col in zip(azimuths, colors):
        lag, sv = directional_semivariance(coords, v, az)
        ax.plot(lag, sv, 'o-', color=col, label=f'{az} deg', ms=4)
        if len(sv) > 3:
            # crude range proxy: lag at which semivariance reaches 95% of its own max
            thresh = 0.95 * sv.max()
            over = lag[sv >= thresh]
            rng_est = over.min() if len(over) else np.nan
            aniso_summary.append({'task': task, 'azimuth': az, 'range_proxy_m': rng_est, 'sill_proxy': sv.max()})
    ax.set_title(TASK_LABELS[task])
    ax.set_xlabel('Lag distance h (m)'); ax.set_ylabel('Semivariance $\\gamma(h)$')
    ax.legend(fontsize=8, title='Azimuth')
plt.tight_layout()
plt.savefig(f'{FIG}/fig19_directional_variograms.png'); plt.close()
pd.DataFrame(aniso_summary).to_csv(f'{OUT}/anisotropy_summary.csv', index=False)
print('Fig 19 (directional variograms) done.')
print(pd.DataFrame(aniso_summary).round(1))

# =====================================================================
# Fig 20: Multi-metric radar charts (one per task), all 6 methods
# =====================================================================
cv = pd.read_csv(f'{OUT}/cv_results_spatial.csv')

def normalise_for_radar(sub):
    # Higher = better for all three axes: R2 as-is; RMSE, MAE inverted-normalised to [0,1]
    r2 = sub['R2'].clip(lower=0)
    rmse_score = 1 - (sub['RMSE'] - sub['RMSE'].min()) / (sub['RMSE'].max() - sub['RMSE'].min() + 1e-9)
    mae_score = 1 - (sub['MAE'] - sub['MAE'].min()) / (sub['MAE'].max() - sub['MAE'].min() + 1e-9)
    return r2.to_numpy(), rmse_score.to_numpy(), mae_score.to_numpy()

methods = ['LM', 'IDW', 'OK', 'RF', 'RFK', 'SRK']
method_colors = {'LM': '#4a5568', 'IDW': '#805ad5', 'OK': '#c05621', 'RF': '#2f855a',
                  'RFK': '#4299e1', 'SRK': '#e53e3e'}
method_styles = {'LM': '--', 'IDW': '-', 'OK': '-', 'RF': '-', 'RFK': '-', 'SRK': '-'}
method_widths = {'LM': 2.4, 'IDW': 1.6, 'OK': 1.6, 'RF': 1.6, 'RFK': 1.6, 'SRK': 2.2}
axes_labels = ['R$^2$', '1 - norm. RMSE', '1 - norm. MAE']
angles = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False).tolist()
angles += angles[:1]

fig, axs = plt.subplots(2, 2, figsize=(10, 10), subplot_kw=dict(polar=True))
for ax, task in zip(axs.ravel(), TASKS):
    sub = cv[cv['task'] == task].set_index('model').loc[methods].reset_index()
    r2v, rmv, mav = normalise_for_radar(sub)
    for i, m in enumerate(methods):
        vals = [r2v[i], rmv[i], mav[i]]
        vals += vals[:1]
        ax.plot(angles, vals, color=method_colors[m], lw=method_widths[m], ls=method_styles[m],
                 label=m, marker='o', ms=4, zorder=6 if m in ('LM', 'SRK') else 3)
        if m == 'SRK':
            ax.fill(angles, vals, color=method_colors[m], alpha=0.08)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(axes_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title(TASK_LABELS[task], fontsize=11, pad=14)
axs[0, 0].legend(loc='upper right', bbox_to_anchor=(1.35, 1.25), fontsize=8, ncol=1)
plt.tight_layout()
plt.savefig(f'{FIG}/fig20_radar_multimetric.png'); plt.close()
print('Fig 20 (radar charts) done.')

# =====================================================================
# Fig 21: 3-D SRK trend + kriged surface for corn yield
# =====================================================================
task = 'yield_kgm2'
other = [c for c in TASKS if c != task]
y = df[task].to_numpy()

gx = np.linspace(df['x'].min(), df['x'].max(), 45)
gy = np.linspace(df['y'].min(), df['y'].max(), 45)
GX, GY = np.meshgrid(gx, gy)
grid_xy = np.column_stack([GX.ravel(), GY.ravel()])
grid_xn = (gx - df['x'].mean()) / df['x'].std()
grid_yn = (gy - df['y'].mean()) / df['y'].std()
GXN, GYN = np.meshgrid(grid_xn, grid_yn)

sing_full, sing_grid = {}, {}
for tcol in [task] + other:
    v = df[tcol].to_numpy()
    sing_full[tcol] = singularity_index_from_train(coords, v, coords, radii)
    sing_grid[tcol] = singularity_index_from_train(coords, v, grid_xy, radii)
Xfull = np.column_stack([base_xy] + [sing_full[c] for c in [task] + other])
Xgrid = np.column_stack([np.column_stack([GXN.ravel(), GYN.ravel()])] + [sing_grid[c] for c in [task] + other])
rfs = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4,
                             random_state=1, n_jobs=-1).fit(Xfull, y)
mu_grid = rfs.predict(Xgrid)
res = y - rfs.predict(Xfull)
z_res, _ = ordinary_kriging_predict(coords, res, grid_xy)
Z_srk = (mu_grid + z_res).reshape(GX.shape)

fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(GX, GY, Z_srk, cmap='YlGnBu', linewidth=0, antialiased=True, rstride=1, cstride=1)
ax1.set_xlabel('Easting (m)', labelpad=8); ax1.set_ylabel('Northing (m)', labelpad=8)
ax1.set_zlabel('Predicted yield (kg m$^{-2}$)')
ax1.set_title('SRK prediction surface')
ax1.view_init(elev=32, azim=-60)
fig.colorbar(surf, ax=ax1, shrink=0.55, pad=0.08)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_wireframe(GX, GY, Z_srk, color='#2b6cb0', linewidth=0.4, rstride=2, cstride=2)
sc = ax2.scatter(df['x'], df['y'], df[task], c=df[task], cmap='YlOrRd', s=4, alpha=0.5)
ax2.set_xlabel('Easting (m)', labelpad=8); ax2.set_ylabel('Northing (m)', labelpad=8)
ax2.set_zlabel('Yield (kg m$^{-2}$)')
ax2.set_title('SRK surface (wireframe) vs. observations')
ax2.view_init(elev=25, azim=-45)
plt.tight_layout()
plt.savefig(f'{FIG}/fig21_3d_trend_surface.png'); plt.close()
print('Fig 21 (3D trend surface) done.')

# =====================================================================
# Fig 22: Fold-level CV variability (boxplot/strip) for corn yield, all 6 methods
# =====================================================================
task = 'yield_kgm2'
other = [c for c in TASKS if c != task]
kf = KFold(n_splits=5, shuffle=True, random_state=11)
fold_records = []
y = df[task].to_numpy()
from sklearn.linear_model import LinearRegression
from msrk_engine import idw_predict

for fold_i, (tr, te) in enumerate(kf.split(df)):
    xy_tr, xy_te = coords[tr], coords[te]
    y_tr = y[tr]

    lm = LinearRegression().fit(base_xy[tr], y_tr)
    p_lm = lm.predict(base_xy[te])

    p_idw = idw_predict(xy_tr, y_tr, xy_te)

    p_ok, _ = ordinary_kriging_predict(xy_tr, y_tr, xy_te)

    rf = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=4,
                                random_state=1, n_jobs=-1).fit(base_xy[tr], y_tr)
    mu_tr, mu_te = rf.predict(base_xy[tr]), rf.predict(base_xy[te])
    p_rf = mu_te
    res_tr = y_tr - mu_tr
    z_res, _ = ordinary_kriging_predict(xy_tr, res_tr, xy_te)
    p_rfk = mu_te + z_res

    sing_tr, sing_te = {}, {}
    for tcol in [task] + other:
        v_tr = df[tcol].to_numpy()[tr]
        sing_tr[tcol] = singularity_index_from_train(xy_tr, v_tr, xy_tr, radii)
        sing_te[tcol] = singularity_index_from_train(xy_tr, v_tr, xy_te, radii)
    Xtr = np.column_stack([base_xy[tr]] + [sing_tr[c] for c in [task] + other])
    Xte = np.column_stack([base_xy[te]] + [sing_te[c] for c in [task] + other])
    rfs = RandomForestRegressor(n_estimators=250, max_depth=10, min_samples_leaf=4,
                                 random_state=1, n_jobs=-1).fit(Xtr, y_tr)
    mu_s_tr, mu_s_te = rfs.predict(Xtr), rfs.predict(Xte)
    res_s_tr = y_tr - mu_s_tr
    z_res_s, _ = ordinary_kriging_predict(xy_tr, res_s_tr, xy_te)
    p_srk = mu_s_te + z_res_s

    for name, pred in [('LM', p_lm), ('IDW', p_idw), ('OK', p_ok), ('RF', p_rf), ('RFK', p_rfk), ('SRK', p_srk)]:
        fold_records.append({'fold': fold_i, 'model': name, 'R2': r2_score(y[te], pred)})

fold_df = pd.DataFrame(fold_records)
fold_df.to_csv(f'{OUT}/fold_level_r2_yield.csv', index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
box_data = [fold_df[fold_df['model'] == m]['R2'].to_numpy() for m in methods]
bp = ax.boxplot(box_data, labels=methods, patch_artist=True, widths=0.55)
for patch, m in zip(bp['boxes'], methods):
    patch.set_facecolor(method_colors[m]); patch.set_alpha(0.5)
for i, m in enumerate(methods):
    yv = fold_df[fold_df['model'] == m]['R2'].to_numpy()
    xv = np.random.default_rng(0).normal(i + 1, 0.045, size=len(yv))
    ax.scatter(xv, yv, color=method_colors[m], edgecolor='k', s=28, zorder=5)
ax.set_ylabel('$R^2$ per fold'); ax.set_xlabel('Model')
plt.tight_layout()
plt.savefig(f'{FIG}/fig22_fold_variability.png'); plt.close()
print('Fig 22 (fold-level CV variability boxplot) done.')

print('\\nAll complex figures (19-22) written to', FIG)
