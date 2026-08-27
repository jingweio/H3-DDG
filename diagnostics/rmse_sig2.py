"""Same fold fingerprint, but testing BOTH RMSE conventions and the full 25-assay set.

Convention A: per-DMS equal-weight mean of per-assay std        (what A.3 says: 'per-DMS metrics')
Convention B: row-pooled std over the whole slice               (a plain global RMSE)
Candidate sets: each of the 5 folds, and all 25 assays (i.e. the full OOF, all folds merged).
Target from Table 2: RMSE(>=3)/RMSE(ALL) ~ 2.08-2.35 for every method; <3/ALL ~ 0.87-0.95.
"""
import os
import ast
import numpy as np, pandas as pd

repo = '/home/guoj0f/repos/H3-DDG/.claude/worktrees/reproduce'
# BindingGYM's raw data lives in the shared store (see bindinggym_dataset.py). Same env var,
# same fallback, so this script keeps working wherever the data actually is.
INPUT = os.environ.get('BINDINGGYM_INPUT', f'{repo}/data/input')
folds = pd.read_csv(f'{repo}/data_splits/inter_assay_folds.tsv', sep='\t', comment='#')
MIN = 100

def nmut(cell):
    d = ast.literal_eval(cell) if isinstance(cell, str) and cell.strip() else {}
    return sum(len([t for t in str(v).split(':') if t]) for v in d.values())

cache = {}
for _, r in folds.iterrows():
    src = pd.read_csv(f'{INPUT}/Binding_substitutions_DMS/{r.DMS_id}.csv')
    cache[r.DMS_id] = (src['mutant_pdb'].fillna('{}').apply(nmut).values,
                       src['DMS_score'].values.astype(np.float64), r.test_fold)

def signature(dms_ids, label):
    res = {}
    for name, sel in (('ALL', lambda nm: np.ones(len(nm), bool)),
                      ('<3', lambda nm: nm < 3), ('>=3', lambda nm: nm >= 3)):
        per, pooled = [], []
        for d in dms_ids:
            nm, y, _ = cache[d]
            m = sel(nm)
            if m.sum() >= MIN:
                per.append(y[m].std()); pooled.append(y[m])
        res[name] = (np.mean(per) if per else np.nan,
                     np.concatenate(pooled).std() if pooled else np.nan, len(per))
    a = f"{res['>=3'][0]/res['ALL'][0]:.2f}" if res['ALL'][0] and not np.isnan(res['>=3'][0]) else '  na'
    b = f"{res['>=3'][1]/res['ALL'][1]:.2f}" if res['ALL'][1] and not np.isnan(res['>=3'][1]) else '  na'
    print(f"{label:>14} | perDMS ALL {res['ALL'][0]:7.4f} <3 {res['<3'][0]:7.4f} >=3 {res['>=3'][0]:7.4f} "
          f"ratio {a:>5} | pooled ALL {res['ALL'][1]:7.4f} <3 {res['<3'][1]:7.4f} >=3 {res['>=3'][1]:7.4f} "
          f"ratio {b:>5} | assays {res['ALL'][2]}/{res['<3'][2]}/{res['>=3'][2]}")

for f in sorted(folds.test_fold.unique()):
    signature(list(folds[folds.test_fold == f].DMS_id), f'fold {f}')
signature(list(folds.DMS_id), 'ALL 25 assays')
print('\nTarget (Table 2, every method):  ratio >=3/ALL = 2.08-2.35')
