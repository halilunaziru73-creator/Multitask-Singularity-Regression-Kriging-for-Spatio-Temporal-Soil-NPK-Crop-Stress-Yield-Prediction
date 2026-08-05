import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from msrk_engine import build_singularity_features, run_cv_all_methods, \
    ordinary_kriging_predict
import importlib

df = pd.read_csv(os.path.join(THIS_DIR, '..', 'data', 'corn_multitask_sub.csv'))
TASKS = ['yield_kgm2', 'npk_ppm', 'soil_ph', 'caco3_ppm']

# normalise coords (helps RF/LM numerics)
df['x_n'] = (df['x'] - df['x'].mean()) / df['x'].std()
df['y_n'] = (df['y'] - df['y'].mean()) / df['y'].std()

print('Building multiscale singularity-index features (this profiles each of the',
      len(TASKS), 'tasks over 4 nested neighbourhoods)...')
sing = build_singularity_features(df, TASKS, radii=(50, 100, 200, 400))
df = pd.concat([df, sing], axis=1)
df.to_csv(os.path.join(THIS_DIR, '..', 'data', 'corn_multitask_with_singularity.csv'), index=False)
print(df[[f'alpha_{t}' for t in TASKS]].describe().T[['mean', 'std', 'min', 'max']])

# NOTE: yield_kgm2 <-> npk_ppm (r=-0.99) and soil_ph <-> caco3_ppm (r=-1.00) are
# near-deterministic estimate pairs in this dataset, so raw cross-task values are
# EXCLUDED as features everywhere below (see msrk_engine.run_cv_all_methods) --
# only training-fold-derived singularity indices and coordinates are used.
results_all = []
for task in TASKS:
    other_tasks = [t for t in TASKS if t != task]
    res = run_cv_all_methods(df, [task], other_tasks, radii=(50, 100, 200, 400),
                              n_splits=5, seed=11)
    results_all.append(res)

cv_results = pd.concat(results_all, ignore_index=True)
cv_results.to_csv(os.path.join(THIS_DIR, '..', 'outputs_data', 'cv_results_spatial.csv'), index=False)
print('\n=== 5-fold spatial CV results (raw units) ===')
print(cv_results.pivot(index='model', columns='task', values='R2').round(3))
print('\nRMSE:')
print(cv_results.pivot(index='model', columns='task', values='RMSE').round(3))
