"""Question (1): is DMS_score's DIRECTION consistent across the 25 assays, and does it match
H3-DDG's ddG convention?

Two external anchors first:
  * BindingGYM curates DMS_score as `raw_phenotype * DMS_directionality`
    (BindingGYM utils/data_utils.py:25), so direction is normalised at curation time.
  * BindingGYM's own training code writes the label as
    `-ddg if 'DMS_score' not in columns else DMS_score` (training/dataset.py),
    i.e. it treats `-ddg` and `DMS_score` as the same quantity  =>  ddg = -DMS_score.
    That is exactly bindinggym.py's `label_sign = -1`.

This script checks the claim against the data, independently, two ways:
  A. WT percentile. In a loss-of-function scan most mutations hurt, so the wild type should sit
     NEAR THE TOP of its assay's DMS_score distribution. A flipped assay would put WT near the
     bottom. (Assays with no WT row are reported as such.)
  B. Correlation between mutation COUNT and DMS_score. More mutations should mean worse binding
     if higher-is-better holds and the library is a loss-of-function scan; a strong POSITIVE
     correlation means the library is gain-of-function, where the wild type is the worst binder.
Both are direction diagnostics, but B also flags the gain-of-function assays -- where an
inverse-folding model that rewards "looks natural" is inverted BY CONSTRUCTION.
"""
import ast
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.abspath(__file__)) + '/..'
# BindingGYM's raw data lives in the shared store (see bindinggym_dataset.py). Same env var,
# same fallback, so this script keeps working wherever the data actually is.
INPUT = os.environ.get('BINDINGGYM_INPUT', f'{REPO}/data/input')
def nmut(cell):
    d = ast.literal_eval(cell) if isinstance(cell, str) and cell.strip() else {}
    return sum(len([t for t in str(v).split(':') if t]) for v in d.values())


folds = pd.read_csv(f'{REPO}/data_splits/inter_assay_folds.tsv', sep='\t', comment='#')
rows = []
for _, r in folds.iterrows():
    d = pd.read_csv(f'{INPUT}/Binding_substitutions_DMS/{r.DMS_id}.csv')
    nm = d['mutant_pdb'].fillna('{}').apply(nmut).values
    y = d['DMS_score'].values.astype(np.float64)
    wt = np.where(nm == 0)[0]
    wt_pct = float((y < y[wt[0]]).mean() * 100) if len(wt) else np.nan
    wt_val = float(y[wt[0]]) if len(wt) else np.nan
    sub = nm >= 1
    rho = spearmanr(nm[sub], y[sub])[0] if sub.sum() > 2 and len(np.unique(nm[sub])) > 1 else np.nan
    rows.append(dict(fold=r.test_fold, DMS_id=r.DMS_id, n=len(d),
                     wt_pct=wt_pct, wt_val=wt_val, y_min=y.min(), y_max=y.max(),
                     rho_nmut_score=rho))

t = pd.DataFrame(rows).sort_values(['fold', 'DMS_id'])
pd.set_option('display.width', 220)
print(t.to_string(index=False, float_format=lambda v: f'{v:9.3f}'))

have = t[t.wt_pct.notna()]
print(f'\nassays with a WT row: {len(have)}/25')
print(f'  WT percentile >= 50 (loss-of-function, direction as expected): {(have.wt_pct >= 50).sum()}')
print(f'  WT percentile <  50 (GAIN-of-function: wild type is a POOR binder): {(have.wt_pct < 50).sum()}')
gof = have[have.wt_pct < 50]
if len(gof):
    print('  -> ' + ', '.join(f'{r.DMS_id[:32]} (WT at {r.wt_pct:.0f}%, fold {int(r.fold)})'
                              for _, r in gof.iterrows()))
print(f'\nassays where MORE mutations correlate with HIGHER score (rho > +0.2): '
      f'{(t.rho_nmut_score > 0.2).sum()}')
for _, r in t[t.rho_nmut_score > 0.2].iterrows():
    print(f'  {r.DMS_id[:34]:34} fold {int(r.fold)}  rho {r.rho_nmut_score:+.3f}  n {r.n}')
print('\nNo assay should have WT near 0% AND rho strongly negative -- that would be a flipped sign.')
flipped = have[(have.wt_pct < 20) & (have.rho_nmut_score < -0.2)]
print('flipped-sign candidates:', list(flipped.DMS_id) if len(flipped) else 'NONE')
