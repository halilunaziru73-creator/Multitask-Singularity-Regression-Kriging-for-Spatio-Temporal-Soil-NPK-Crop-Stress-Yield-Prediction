import sys
sys.path.insert(0, '/home/claude/msrk/code')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
FIGDIR = '/home/claude/msrk/figs'
TASKS = ['yield_kgm2', 'npk_ppm', 'soil_ph', 'caco3_ppm']
TASK_LABELS = {'yield_kgm2': 'Corn yield (kg m$^{-2}$)', 'npk_ppm': 'NPK (est., ppm)',
               'soil_ph': 'Soil pH', 'caco3_ppm': 'CaCO$_3$ (est., ppm)'}

df = pd.read_csv('/home/claude/msrk/data/corn_multitask_with_singularity.csv')

# ---------------- Fig 1: correlation heatmap ----------------
cols = TASKS + [f'alpha_{t}' for t in TASKS]
corr = df[cols].corr(method='spearman')
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha='right')
ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=7,
                 color='white' if abs(corr.iloc[i, j]) > 0.6 else 'black')
plt.colorbar(im, label="Spearman's rho")

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig1_correlation.png'); plt.close()

# ---------------- Fig 2: singularity index maps (4 tasks) ----------------
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, t in zip(axes, TASKS):
    sc = ax.scatter(df['x'], df['y'], c=df[f'alpha_{t}'], cmap='viridis', s=6)
    ax.set_title(f'$\\alpha$({TASK_LABELS[t]})', fontsize=9)
    ax.set_xlabel('Easting (m)'); ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, fraction=0.046)
axes[0].set_ylabel('Northing (m)')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig2_singularity_maps.png'); plt.close()

# ---------------- Fig 3: CV comparison (R2 heatmap like screenshot) ----------------
cv = pd.read_csv('/home/claude/msrk/out/cv_results_spatial.csv')
piv = cv.pivot(index='model', columns='task', values='R2')[TASKS].loc[['LM', 'IDW', 'OK', 'RF', 'RFK', 'SRK']]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
im = ax.imshow(piv.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(TASKS))); ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], rotation=20, ha='right')
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f'{piv.values[i,j]:.3f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, label='$R^2$ (5-fold spatial CV)')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig3_cv_r2.png'); plt.close()

# ---------------- Fig 4: RMSE bar chart, normalised (%) ----------------
piv_rmse = cv.pivot(index='model', columns='task', values='RMSE')[TASKS].loc[['LM', 'IDW', 'OK', 'RF', 'RFK', 'SRK']]
piv_rmse_norm = piv_rmse / piv_rmse.loc['IDW']  # relative to IDW baseline
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(TASKS)); w = 0.13
for i, m in enumerate(piv_rmse_norm.index):
    ax.bar(x + i * w, piv_rmse_norm.loc[m], width=w, label=m)
ax.set_xticks(x + 2.5 * w); ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], rotation=15, ha='right')
ax.axhline(1.0, color='k', lw=0.6, ls='--')
ax.set_ylabel('RMSE relative to IDW baseline')

ax.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig4_rmse_relative.png'); plt.close()

# ---------------- Fig 5: Observed vs predicted (SRK) for yield & NPK, full-data fit ----------------
from sklearn.ensemble import RandomForestRegressor
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, t in zip(axes, ['yield_kgm2', 'npk_ppm']):
    other = [c for c in TASKS if c != t]
    X = df[['x', 'y'] + [f'alpha_{c}' for c in [t] + other]].to_numpy()
    y = df[t].to_numpy()
    rf = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=4,
                                random_state=1, n_jobs=-1).fit(X, y)
    pred = rf.predict(X)
    ax.scatter(y, pred, s=5, alpha=0.35, c='#2b6cb0')
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lims, lims, 'k--', lw=1)
    from sklearn.metrics import r2_score
    ax.set_title(f'{TASK_LABELS[t]}  (in-sample $R^2$={r2_score(y,pred):.3f})')
    ax.set_xlabel('Observed'); ax.set_ylabel('SRK-predicted')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig5_obs_vs_pred.png'); plt.close()

print('Spatial figures done.')

# ---------------- Spatio-temporal figures ----------------
st = pd.read_csv('/home/claude/msrk/data/brix_st_wide.csv')
long = pd.read_csv('/home/claude/msrk/data/brix_st_long_with_singularity.csv')
date_cols = ['Jul_15', 'Jul_30', 'Aug_06', 'Aug_15', 'Aug_30']

# Fig 6: Brix trajectories coloured by stress index
fig, ax = plt.subplots(figsize=(7, 4.5))
doy = [196, 211, 218, 227, 242]
cmap = plt.cm.RdYlBu
for _, r in st.iterrows():
    ax.plot(doy, r[date_cols].values, color=cmap(r['stress_index']), alpha=0.7, lw=1)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
plt.colorbar(sm, ax=ax, label='Stress index (0=fast sugar-loading, 1=slow/stressed)')
ax.set_xlabel('Day of year'); ax.set_ylabel('Brix (\u00b0Bx)')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig6_brix_trajectories.png'); plt.close()

# Fig 7: spatial map of stress index + NDVI + singularity(final date)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
sc0 = axes[0].scatter(st['x'], st['y'], c=st['stress_index'], cmap='RdYlBu', s=60, edgecolor='k', lw=0.3)
axes[0].set_title('Crop-stress index'); plt.colorbar(sc0, ax=axes[0], fraction=0.046)
sc1 = axes[1].scatter(st['x'], st['y'], c=st['NDVI1'], cmap='YlGn', s=60, edgecolor='k', lw=0.3)
axes[1].set_title('NDVI'); plt.colorbar(sc1, ax=axes[1], fraction=0.046)
final = long[long['date'] == 'Aug_30']
sc2 = axes[2].scatter(final['x'], final['y'], c=final['alpha_brix_st'], cmap='viridis', s=60, edgecolor='k', lw=0.3)
axes[2].set_title('Space-time singularity index $\\alpha_{st}$\n(Aug 30 slice)')
plt.colorbar(sc2, ax=axes[2], fraction=0.046)
for ax in axes: ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig7_st_maps.png'); plt.close()

# Fig 8: feature importance + CV bar for spatio-temporal model
imp = pd.read_csv('/home/claude/msrk/out/feature_importance_st.csv', index_col=0).iloc[:, 0]
cv_st = pd.read_csv('/home/claude/msrk/out/cv_results_spatiotemporal.csv')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
imp.sort_values().plot.barh(ax=axes[0], color='#2b6cb0')
axes[0].set_title('Space-time RF trend: feature importance')
axes[0].set_xlabel('Importance (impurity)')
axes[1].bar(cv_st['model'], cv_st['R2'], color=['#a0aec0', '#4299e1', '#2b6cb0'])
axes[1].set_ylim(0, 1); axes[1].set_ylabel('$R^2$ (5-fold CV)')
axes[1].set_title('Spatio-temporal Brix prediction (STRK vs. baselines)')

plt.tight_layout()
plt.savefig(f'{FIGDIR}/fig8_st_importance_cv.png'); plt.close()

print('All figures written to', FIGDIR)
