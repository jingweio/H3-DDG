"""Pin down WHICH single fold the paper reports, using two independent signatures.

S4.1: 'the fold with the most multi-point mutations for testing'.  Ambiguous ('multi' could be
>=2 or >=3), so cross-check against a second signature that does not depend on that reading:
Table 2's RMSE pattern.  For EVERY method in Table 2, ALL ~ <3 << >=3
  ProteinMPNN 3.4974 / 3.2328 / 5.5921    BA-Cycle 1.2419 / 1.0925 / 2.5822
  Prompt-DDG  1.5216 / 1.3499 / 3.5747    BA-DDG   1.1182 / 0.9716 / 2.5191
  H3-DDG      1.1294 / 1.0758 / 2.4976
Metrics are per-DMS equal-weight averages, so ALL ~ <3 requires that, within each surviving
assay, the <3 rows OUTNUMBER the >=3 rows.  Only some folds can do that.
Also checks which folds can even populate all three columns under BindingGYM's >=100-row filter.
"""
import ast
import pandas as pd

repo = '/home/guoj0f/repos/H3-DDG/.claude/worktrees/reproduce'
folds = pd.read_csv(f'{repo}/data_splits/inter_assay_folds.tsv', sep='\t', comment='#')
MIN = 100

def nmut(cell):
    d = ast.literal_eval(cell) if isinstance(cell, str) and cell.strip() else {}
    return sum(len([t for t in str(v).split(':') if t]) for v in d.values())

recs = []
for _, r in folds.iterrows():
    src = pd.read_csv(f'{repo}/data/input/Binding_substitutions_DMS/{r.DMS_id}.csv')
    nm = src['mutant_pdb'].fillna('{}').apply(nmut)
    recs.append(dict(fold=r.test_fold, DMS_id=r.DMS_id, n_all=len(src),
                     n_lt3=int(((nm >= 1) & (nm < 3)).sum()), n_ge3=int((nm >= 3).sum())))
t = pd.DataFrame(recs)

print(f'{"fold":>4} {"assays: ALL":>11} {"<3":>4} {">=3":>4}   {"assays where <3 outnumber >=3":>30}')
for f, g in t.groupby('fold'):
    a_all = int((g.n_all >= MIN).sum())
    a_lt3 = int((g.n_lt3 >= MIN).sum())
    a_ge3 = int((g.n_ge3 >= MIN).sum())
    surv = g[g.n_all >= MIN]
    dom = int((surv.n_lt3 > surv.n_ge3).sum())
    flag = ''
    if a_lt3 == 0 or a_ge3 == 0:
        flag = '  <- cannot fill all 3 Table-2 columns'
    elif dom == a_all:
        flag = '  <- MATCHES the ALL~<3 RMSE pattern'
    print(f'{f:>4} {a_all:>11} {a_lt3:>4} {a_ge3:>4}   {f"{dom}/{a_all}":>30}{flag}')

print('\n=== per-assay detail ===')
for f, g in t.groupby('fold'):
    print(f'-- f{f}')
    for _, r in g.iterrows():
        print(f'   {r.DMS_id[:34]:34} all {r.n_all:6d}  <3 {r.n_lt3:6d}  >=3 {r.n_ge3:6d}'
              f'  {"<3 dominates" if r.n_lt3 > r.n_ge3 else ">=3 dominates"}')
