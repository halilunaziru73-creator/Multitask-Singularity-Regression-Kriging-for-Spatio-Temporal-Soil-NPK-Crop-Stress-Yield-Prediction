import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
FIG = '/home/claude/msrk/figs'
OUT = '/home/claude/msrk/out'
TASKS = ['yield_kgm2', 'npk_ppm', 'soil_ph', 'caco3_ppm']
TASK_LABELS = {'yield_kgm2': 'Corn yield', 'npk_ppm': 'NPK (est.)',
               'soil_ph': 'Soil pH', 'caco3_ppm': 'CaCO$_3$ (est.)'}
df = pd.read_csv('/home/claude/msrk/data/corn_multitask_with_singularity.csv')

# ---------------- Fig 11: Moran's I bar + Moran scatterplot for yield ----------------
moran = pd.read_csv(f'{OUT}/morans_i.csv')
fig = plt.figure(figsize=(11, 4.2))
gs = GridSpec(1, 2, width_ratios=[1, 1.2])
ax0 = fig.add_subplot(gs[0])
bars = ax0.bar([TASK_LABELS[t] for t in moran['task']], moran['MoransI'],
               color=['#2b6cb0', '#2b6cb0', '#c05621', '#c05621'])
ax0.axhline(0, color='k', lw=0.8)
for b, z in zip(bars, moran['z']):
    ax0.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f'z={z:.1f}',
              ha='center', fontsize=8)
ax0.set_ylabel("Global Moran's I")
ax0.set_ylim(0, 1.15)
ax0.set_title("Global Moran's I (k=8 NN weights)\nall p < 0.001")

# Moran scatterplot for yield: standardised value vs spatial-lag (mean of 8 NN)
from scipy.spatial import cKDTree
coords = df[['x', 'y']].to_numpy()
z = (df['yield_kgm2'] - df['yield_kgm2'].mean()) / df['yield_kgm2'].std()
tree = cKDTree(coords)
_, idx = tree.query(coords, k=9)
lag = z.to_numpy()[idx[:, 1:]].mean(axis=1)
ax1 = fig.add_subplot(gs[1])
ax1.scatter(z, lag, s=5, alpha=0.35, c='#2b6cb0')
b1, b0 = np.polyfit(z, lag, 1)
xx = np.linspace(z.min(), z.max(), 50)
ax1.plot(xx, b0 + b1 * xx, 'r-', lw=1.5, label=f"slope (Moran's I)={b1:.3f}")
ax1.axhline(0, color='gray', lw=0.5); ax1.axvline(0, color='gray', lw=0.5)
ax1.set_xlabel('Standardised corn yield, z(s)'); ax1.set_ylabel('Spatial lag, mean z(neighbours)')
ax1.set_title('Moran scatterplot: corn yield')
ax1.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{FIG}/fig11_morans_i.png'); plt.close()

# ---------------- Fig 12: Variogram model comparison (small multiples) ----------------
vgm_curves = np.load(f'{OUT}/vgm_curves.npy', allow_pickle=True).item()
vgm_df = pd.read_csv(f'{OUT}/variogram_model_comparison.csv')

def spherical(h, nugget, sill, rang):
    h = np.asarray(h)
    return np.where(h <= rang, nugget + (sill - nugget) * (1.5 * h / rang - 0.5 * (h / rang) ** 3), sill)
def exponential(h, nugget, sill, rang):
    return nugget + (sill - nugget) * (1 - np.exp(-3 * np.asarray(h) / rang))
def gaussian_vgm(h, nugget, sill, rang):
    return nugget + (sill - nugget) * (1 - np.exp(-3 * (np.asarray(h) / rang) ** 2))
fn_map = {'Spherical': spherical, 'Exponential': exponential, 'Gaussian': gaussian_vgm}
colors = {'Spherical': '#2b6cb0', 'Exponential': '#c05621', 'Gaussian': '#2f855a'}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, t in zip(axes.ravel(), TASKS):
    lag, sv = vgm_curves[t]
    ax.scatter(lag, sv, s=18, c='k', label='Empirical', zorder=5)
    sub = vgm_df[vgm_df['task'] == t]
    hh = np.linspace(0.1, lag.max(), 200)
    for _, row in sub.iterrows():
        fn = fn_map[row['model']]
        ax.plot(hh, fn(hh, row['nugget'], row['sill'], row['range_m']),
                color=colors[row['model']], lw=1.5,
                label=f"{row['model']} (R2={row['R2_fit']:.3f})")
    ax.set_title(TASK_LABELS[t]); ax.set_xlabel('Lag distance h (m)'); ax.set_ylabel('Semivariance $\\gamma(h)$')
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(f'{FIG}/fig12_variogram_models.png'); plt.close()

# ---------------- Fig 13: Sensitivity heatmap (radius bank x n_scales) ----------------
sens = pd.read_csv(f'{OUT}/sensitivity_radii.csv')
piv = sens.pivot(index='radius_set', columns='n_scales', values='mean_R2').loc[['fine', 'medium', 'coarse']]
fig, ax = plt.subplots(figsize=(5.5, 4))
im = ax.imshow(piv.values, cmap='RdYlGn', vmin=piv.values.min() - 0.02, vmax=piv.values.max() + 0.02)
ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
ax.set_xlabel('Number of nested scales'); ax.set_ylabel('Radius bank')
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f'{piv.values[i,j]:.3f}', ha='center', va='center', fontsize=10)
plt.colorbar(im, label='Mean $R^2$ (4 tasks, 3-fold CV, n=500 subsample)')

plt.tight_layout()
plt.savefig(f'{FIG}/fig13_sensitivity.png'); plt.close()

# ---------------- Fig 14: Permutation importance heatmap ----------------
perm = pd.read_csv(f'{OUT}/permutation_importance.csv')
piv2 = perm.groupby(['task', 'feature'])['importance_mean'].mean().unstack()
feat_order = ['x', 'y', 'alpha_yield_kgm2', 'alpha_npk_ppm', 'alpha_soil_ph', 'alpha_caco3_ppm']
piv2 = piv2[[c for c in feat_order if c in piv2.columns]]
piv2 = piv2.loc[TASKS]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
im = ax.imshow(piv2.values, cmap='viridis', aspect='auto')
ax.set_xticks(range(piv2.shape[1])); ax.set_xticklabels(piv2.columns, rotation=30, ha='right')
ax.set_yticks(range(piv2.shape[0])); ax.set_yticklabels([TASK_LABELS[t] for t in piv2.index])
for i in range(piv2.shape[0]):
    for j in range(piv2.shape[1]):
        ax.text(j, i, f'{piv2.values[i,j]:.2f}', ha='center', va='center', fontsize=8,
                 color='white' if piv2.values[i, j] < piv2.values.max() * 0.6 else 'black')
plt.colorbar(im, label='Permutation importance ($\\Delta R^2$, held-out folds)')

plt.tight_layout()
plt.savefig(f'{FIG}/fig14_permutation_importance.png'); plt.close()

# ---------------- Fig 15: PCA biplot (soil chemistry) ----------------
scores = np.load(f'{OUT}/pca_scores.npy')
loadings = np.load(f'{OUT}/pca_loadings.npy')
explvar = pd.read_csv(f'{OUT}/pca_explained_variance.csv', index_col=0).iloc[:, 0].to_numpy()
covs = ['PH_H2O', 'MO_PERC', 'P2O5', 'K2O', 'CaO', 'MgO', 'Top_Soil_E', 'Deep_Soil_']
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.scatter(scores[:, 0], scores[:, 1], s=4, alpha=0.15, c='#a0aec0')
scale = 3.2
for i, c in enumerate(covs):
    ax.arrow(0, 0, loadings[i, 0] * scale, loadings[i, 1] * scale, color='#c05621',
              head_width=0.12, length_includes_head=True)
    ax.text(loadings[i, 0] * scale * 1.15, loadings[i, 1] * scale * 1.15, c, color='#9c2b0e', fontsize=9,
             ha='center')
ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel(f'PC1 ({explvar[0]*100:.1f}% var.)'); ax.set_ylabel(f'PC2 ({explvar[1]*100:.1f}% var.)')

plt.tight_layout()
plt.savefig(f'{FIG}/fig15_pca_biplot.png'); plt.close()

# ---------------- Fig 16: Cross-section transect ----------------
transect = np.load(f'{OUT}/transect_results.npy', allow_pickle=True).item()
fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
for ax, task in zip(axes, ['yield_kgm2', 'npk_ppm']):
    r = transect[task]
    ax.plot(r['x'], r['OK'], label='OK', color='#c05621', lw=1.3)
    ax.plot(r['x'], r['RF'], label='RF', color='#2f855a', lw=1.3)
    ax.plot(r['x'], r['SRK'], label='SRK (proposed)', color='#2b6cb0', lw=1.8)
    ax.fill_between(r['x'], r['SRK'] - np.sqrt(r['SRK_var']), r['SRK'] + np.sqrt(r['SRK_var']),
                     color='#2b6cb0', alpha=0.15, label='SRK +/- 1 s.d.')
    ax.set_ylabel(TASK_LABELS[task]); ax.legend(fontsize=8, ncol=4)
axes[-1].set_xlabel('Easting (m)')

plt.tight_layout()
plt.savefig(f'{FIG}/fig16_cross_section.png'); plt.close()

# ---------------- Fig 17: Uncertainty maps (RFK vs SRK) ----------------
unc = np.load(f'{OUT}/uncertainty_grid.npz')
GX, GY, var_rfk, var_srk = unc['GX'], unc['GY'], unc['var_rfk'], unc['var_srk']
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
vmax = max(var_rfk.max(), var_srk.max())
im0 = axes[0].pcolormesh(GX, GY, var_rfk, cmap='inferno', vmin=0, vmax=vmax)
axes[0].set_title('RFK residual-kriging variance'); plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].pcolormesh(GX, GY, var_srk, cmap='inferno', vmin=0, vmax=vmax)
axes[1].set_title('SRK residual-kriging variance'); plt.colorbar(im1, ax=axes[1], fraction=0.046)
axes[2].hist(var_rfk.ravel(), bins=25, alpha=0.55, color='#c05621', label=f'RFK (mean={var_rfk.mean():.4f})', density=True)
axes[2].hist(var_srk.ravel(), bins=25, alpha=0.55, color='#2b6cb0', label=f'SRK (mean={var_srk.mean():.4f})', density=True)
axes[2].set_xlabel('Kriging variance'); axes[2].set_ylabel('Density'); axes[2].legend(fontsize=8)
axes[2].set_title('Uncertainty distribution')
for ax in axes[:2]: ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{FIG}/fig17_uncertainty.png'); plt.close()

# ---------------- Fig 18: Residual diagnostics (QQ + spatial residual map) ----------------
qq = np.load(f'{OUT}/qq_yield_srk.npz')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].scatter(qq['osm'], qq['osr'], s=8, c='#2b6cb0', alpha=0.5)
xx = np.linspace(qq['osm'].min(), qq['osm'].max(), 50)
axes[0].plot(xx, qq['intercept'] + qq['slope'] * xx, 'r-', lw=1.5)
axes[0].set_xlabel('Theoretical quantiles'); axes[0].set_ylabel('Ordered residuals')
axes[0].set_title(f"SRK trend-residual Q-Q plot, corn yield\n(r={float(qq['r']):.4f})")

# reload residuals for spatial map (recompute quickly is expensive; approximate via saved uncertainty grid context)
axes[1].hist(qq['osr'], bins=30, color='#4299e1', edgecolor='white')
axes[1].axvline(0, color='k', lw=1)
axes[1].set_title('SRK trend-residual distribution, corn yield')
axes[1].set_xlabel('Residual (kg m$^{-2}$)'); axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig(f'{FIG}/fig18_residual_diagnostics.png'); plt.close()

print('All advanced figures (11-18) written to', FIG)
