"""
01_data_prep.py
Multitask Singularity Regression Kriging (MSRK) - Data Preparation
Builds:
  (A) Spatial multitask dataset: corn yield (CORN_KG_M2), NPK_est, soil pH (ph_1),
      CaCO3_Est  -- all georeferenced (UTM), n=17,864 grid samples, Coimbra corn field.
  (B) Spatio-temporal dataset: canopy Brix (proxy for photosynthate / ripening stress)
      measured on 5 dates (Jul_15 -> Aug_30) at 68 fixed georeferenced stations, with
      companion NDVI (crop-stress proxy).
"""
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ---------- (A) Spatial multitask corn dataset ----------
npk = pd.read_csv('../raw_data/Corn_NPK.csv')
ph  = pd.read_csv('../raw_data/Corn_rise_pH.csv')

npk.columns = [c.strip() for c in npk.columns]
ph.columns  = [c.strip() for c in ph.columns]

df = npk.merge(ph[['SAMPLE', 'ph_1', 'CaCO3_Est']], on='SAMPLE', how='inner')
df = df.rename(columns={'XX_UTM': 'x', 'YY_UTM': 'y',
                         'CORN_KG_M2': 'yield_kgm2',
                         'NPK_est': 'npk_ppm',
                         'ph_1': 'soil_ph',
                         'CaCO3_Est': 'caco3_ppm'})
df = df.dropna(subset=['x', 'y', 'yield_kgm2', 'npk_ppm', 'soil_ph', 'caco3_ppm']).reset_index(drop=True)

# de-duplicate identical coordinates (grid overlaps) and subsample for tractable
# variogram/kriging (O(n^2)-O(n^3) cost) while preserving full data for RF trend fitting.
df = df.drop_duplicates(subset=['x', 'y']).reset_index(drop=True)

TASKS = ['yield_kgm2', 'npk_ppm', 'soil_ph', 'caco3_ppm']

df.to_csv(os.path.join(THIS_DIR, '..', 'data', 'corn_multitask_full.csv'), index=False)

# Kriging-tractable subsample (spatially thinned via random subsample of a fine grid)
N_SUB = 1400
if len(df) > N_SUB:
    idx = RNG.choice(len(df), size=N_SUB, replace=False)
    df_sub = df.iloc[idx].reset_index(drop=True)
else:
    df_sub = df.copy()
df_sub.to_csv(os.path.join(THIS_DIR, '..', 'data', 'corn_multitask_sub.csv'), index=False)

print('Full corn multitask dataset:', df.shape)
print('Subsampled (kriging-tractable):', df_sub.shape)
print(df[TASKS].describe().T[['mean', 'std', 'min', 'max']])

# ---------- (B) Spatio-temporal Brix / NDVI dataset ----------
brix = pd.read_excel('../raw_data/Sample_brix_ndvi.xlsx', sheet_name=0)
brix.columns = [c.strip() for c in brix.columns]
coords = pd.read_csv('../raw_data/BRIX_AMT.csv')
coords.columns = [c.strip() for c in coords.columns]
coords = coords.rename(columns={'Sample': 'Id'})[['Id', 'XX_UTM', 'YY_UTM']]

brix_st = brix.merge(coords, on='Id', how='inner').rename(columns={'XX_UTM': 'x', 'YY_UTM': 'y'})
date_cols = ['Jul_15', 'Jul_30', 'Aug_06', 'Aug_15', 'Aug_30']

# long (spatio-temporal) format: one row per station x date
doy_map = {'Jul_15': 196, 'Jul_30': 211, 'Aug_06': 218, 'Aug_15': 227, 'Aug_30': 242}
long_rows = []
for _, r in brix_st.iterrows():
    for dc in date_cols:
        long_rows.append({'Id': r['Id'], 'x': r['x'], 'y': r['y'],
                           'date': dc, 't': doy_map[dc], 'brix': r[dc], 'ndvi': r['NDVI1']})
brix_long = pd.DataFrame(long_rows)

# Ripening rate = slope of Brix over the season per station -> crop-stress proxy
# (slow Brix accumulation despite normal NDVI => sugar-loading/water stress signature)
slopes = brix_st.apply(lambda r: np.polyfit([doy_map[d] for d in date_cols],
                                             [r[d] for d in date_cols], 1)[0], axis=1)
brix_st['brix_rate'] = slopes
brix_st['stress_index'] = (brix_st['brix_rate'].max() - brix_st['brix_rate']) / \
                           (brix_st['brix_rate'].max() - brix_st['brix_rate'].min())

brix_st.to_csv(os.path.join(THIS_DIR, '..', 'data', 'brix_st_wide.csv'), index=False)
brix_long.to_csv(os.path.join(THIS_DIR, '..', 'data', 'brix_st_long.csv'), index=False)

print('\nSpatio-temporal Brix/NDVI dataset:', brix_st.shape, '(', len(brix_long), 'space-time observations)')
print(brix_st[['NDVI1', 'brix_rate', 'stress_index']].describe().T[['mean', 'std', 'min', 'max']])
