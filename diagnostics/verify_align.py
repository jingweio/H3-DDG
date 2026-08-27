"""Decisive free check: does each OOF row's label match the SOURCE csv at that row_index?

collect_results() indexes `id` with i (position in the batch) but `ddG`/`ddG_pred` with k
(position in complex_row_indices).  If those two ever disagree, every reported number is noise
by construction.  This verifies the id <-> label half of that pairing end-to-end against the
untouched BindingGYM source files, for all 119,200 evaluated rows.
"""
import os
import glob, os
import numpy as np, pandas as pd

repo = '/home/guoj0f/repos/H3-DDG/.claude/worktrees/reproduce'
# BindingGYM's raw data lives in the shared store (see bindinggym_dataset.py). Same env var,
# same fallback, so this script keeps working wherever the data actually is.
INPUT = os.environ.get('BINDINGGYM_INPUT', f'{repo}/data/input')
sp = os.path.dirname(os.path.abspath(__file__))
oof = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(sp, 'oof', 'oof_fold*.csv')))],
                ignore_index=True)

bad_total = 0
print(f'{"DMS_id":38} {"n_oof":>7} {"src_rows":>9} {"mismatch":>9} {"max|delta|":>11}')
for dms, g in oof.groupby('DMS_id'):
    src = pd.read_csv(os.path.join(INPUT, 'Binding_substitutions_DMS', f'{dms}.csv'))
    # row_index was assigned as the positional index into this very csv
    want = src['DMS_score'].values.astype(np.float64)[g.row_index.values]
    got = g.DMS_score.values.astype(np.float64)
    d = np.abs(want - got)
    bad = int((d > 1e-4).sum())
    bad_total += bad
    print(f'{dms[:38]:38} {len(g):7d} {len(src):9d} {bad:9d} {d.max():11.3e}')

print(f'\nTOTAL label/row_index mismatches: {bad_total} / {len(oof)}')
print('=> id <-> label pairing VERIFIED' if bad_total == 0 else '=> ALIGNMENT IS BROKEN')

# Also: are the 14 assays' full row counts fully covered (no dropped/duplicated rows)?
print('\ncoverage check (every source row evaluated exactly once?):')
for dms, g in oof.groupby('DMS_id'):
    src_n = len(pd.read_csv(os.path.join(INPUT, 'Binding_substitutions_DMS', f'{dms}.csv')))
    dup = int(g.row_index.duplicated().sum())
    print(f'  {dms[:38]:38} oof {len(g):6d} / src {src_n:6d}  dup {dup}  '
          f'{"OK" if len(g)==src_n and dup==0 else "<-- CHECK"}')
